/// Visual range editor: 13×13 grid where each cell can hold a per-class
/// weight (0.0 to 1.0). Click/drag paints with the brush weight; brush
/// has quick-set buttons (0/25/50/75/100) plus a slider.
///
/// Round-trips through PokerStove syntax via `range_parser.dart` —
/// existing range strings load into the grid; the live serialization
/// preview is shown below the brush controls.
library;

import 'package:flutter/material.dart';

import '../preflop_presets.dart';
import '../range_parser.dart';

class RangeEditorDialog extends StatefulWidget {
  /// Title shown at the top — typically "Edit OOP range" or "Edit IP range".
  final String title;
  /// Initial range string (parsed on open). Empty is fine.
  final String initial;
  /// Called with the serialized range string when the user clicks Apply.
  final void Function(String) onApply;

  const RangeEditorDialog({
    super.key,
    required this.title,
    required this.initial,
    required this.onApply,
  });

  @override
  State<RangeEditorDialog> createState() => _RangeEditorDialogState();
}

class _RangeEditorDialogState extends State<RangeEditorDialog> {
  late Map<String, double> _weights;
  /// Brush weight (0..1). Click-painting sets a cell to this value.
  double _brush = 1.0;
  /// Drag-paint state — set on drag start, applied to every cell entered.
  double? _dragValue;
  /// Last cell coordinates we wrote to during a drag, to avoid redundant
  /// setStates and repeated paint of the same cell.
  int? _lastDragRow;
  int? _lastDragCol;
  /// True if `onTapDown` already opened this gesture. `onPanStart` checks
  /// this and skips re-toggling — otherwise the same cell would flip
  /// twice on a click-then-drift gesture and net-cancel.
  bool _gestureStartedByTap = false;
  /// Selected preset (resets on edit).
  String? _activePreset;

  @override
  void initState() {
    super.initState();
    _weights = parseRangeString(widget.initial);
  }

  /// Cell label for (row, col) per the chart layout.
  String _label(int row, int col) {
    const ranks = ['A','K','Q','J','T','9','8','7','6','5','4','3','2'];
    if (row == col) return '${ranks[row]}${ranks[row]}';
    if (row < col) return '${ranks[row]}${ranks[col]}s';
    return '${ranks[col]}${ranks[row]}o';
  }

  void _paintCell(int row, int col, double value) {
    final label = _label(row, col);
    setState(() {
      if (value <= 1e-4) {
        _weights.remove(label);
      } else {
        _weights[label] = value;
      }
      _activePreset = null; // edit invalidates preset selection
    });
  }

  void _onPanStart(int row, int col) {
    // Always paint at the current brush value. Brush 0% erases (the footer
    // hint documents this). The previous "toggle if cell already matches
    // brush" behaviour broke drag-paint: the first cell's toggle would set
    // _dragValue to 0, and the rest of the drag would silently erase
    // everything it crossed — exactly the "brush does nothing" symptom.
    _dragValue = _brush;
    _lastDragRow = row;
    _lastDragCol = col;
    _paintCell(row, col, _brush);
  }

  void _onPanContinue(int row, int col) {
    if (_dragValue == null) return;
    if (_lastDragRow == row && _lastDragCol == col) return;
    _lastDragRow = row;
    _lastDragCol = col;
    _paintCell(row, col, _dragValue!);
  }

  @override
  Widget build(BuildContext context) {
    final screen = MediaQuery.of(context).size;
    // Cap at 92% of the screen so the dialog never spills off small monitors.
    // Otherwise let it grow to fit the 13×13 grid + controls without inner
    // scrolling at the training tab's 1.32× text scale.
    final maxW = screen.width * 0.92 < 1040 ? screen.width * 0.92 : 1040.0;
    final maxH = screen.height * 0.92 < 1040 ? screen.height * 0.92 : 1040.0;
    return MediaQuery(
      data: MediaQuery.of(context).copyWith(
        textScaler: const TextScaler.linear(1.32),
      ),
      child: Dialog(
        backgroundColor: const Color(0xFF272A2D),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10),
          side: const BorderSide(color: Color(0xFF324E7A), width: 1.4),
        ),
        child: ConstrainedBox(
          constraints: BoxConstraints(maxWidth: maxW, maxHeight: maxH),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _header(),
              const Divider(height: 1, color: Color(0xFF4A4E52)),
              Flexible(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.fromLTRB(20, 14, 20, 8),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      _presetSelector(),
                      const SizedBox(height: 14),
                      _grid(),
                      const SizedBox(height: 14),
                      _brushControls(),
                      const SizedBox(height: 12),
                      _statsAndPreview(),
                    ],
                  ),
                ),
              ),
              const Divider(height: 1, color: Color(0xFF4A4E52)),
              _footer(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _header() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 14, 12, 10),
      child: Row(
        children: [
          const Icon(Icons.grid_view, size: 20, color: Color(0xFF6FB3DC)),
          const SizedBox(width: 10),
          Text(
            widget.title,
            style: const TextStyle(
              color: Color(0xFFEAE6D9),
              fontSize: 18,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.4,
            ),
          ),
          const Spacer(),
          IconButton(
            icon: const Icon(Icons.close, size: 20),
            onPressed: () => Navigator.of(context).pop(),
          ),
        ],
      ),
    );
  }

  Widget _presetSelector() {
    final groups = presetsByGroup();
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        const Text(
          'PRESET',
          style: TextStyle(
            color: Color(0x99EAE6D9),
            fontSize: 11,
            letterSpacing: 1.6,
            fontWeight: FontWeight.w800,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              isExpanded: true,
              value: _activePreset,
              hint: const Text(
                'Pick a predefined range…',
                style: TextStyle(color: Color(0x88EAE6D9), fontSize: 13),
              ),
              dropdownColor: const Color(0xFF1B1F22),
              style: const TextStyle(color: Color(0xFFEAE6D9), fontSize: 13),
              items: [
                for (final entry in groups.entries) ...[
                  DropdownMenuItem<String>(
                    enabled: false,
                    value: '__group_${entry.key}',
                    child: Padding(
                      padding: const EdgeInsets.symmetric(vertical: 4),
                      child: Text(
                        entry.key.toUpperCase(),
                        style: const TextStyle(
                          color: Color(0xFF6FB3DC),
                          fontSize: 11,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 1.4,
                        ),
                      ),
                    ),
                  ),
                  for (final p in entry.value)
                    DropdownMenuItem<String>(
                      value: p.label,
                      child: Padding(
                        padding: const EdgeInsets.only(left: 8),
                        child: Tooltip(
                          message: p.description,
                          child: Text(
                            p.label,
                            style: const TextStyle(fontSize: 13),
                          ),
                        ),
                      ),
                    ),
                ],
              ],
              onChanged: (v) {
                if (v == null || v.startsWith('__group_')) return;
                final preset = preflopPresets.firstWhere((p) => p.label == v);
                setState(() {
                  _weights = parseRangeString(preset.range);
                  _activePreset = v;
                });
              },
            ),
          ),
        ),
        const SizedBox(width: 12),
        TextButton.icon(
          icon: const Icon(Icons.delete_outline, size: 16),
          label: const Text('Clear all'),
          style: TextButton.styleFrom(foregroundColor: const Color(0xCCEAE6D9)),
          onPressed: () => setState(() {
            _weights = {};
            _activePreset = null;
          }),
        ),
      ],
    );
  }

  Widget _grid() {
    // Cell math: each cell is `cellSize` wide PLUS `cellGap` total margin
    // (split as 0.5 left + 0.5 right). The hit-test slot per cell is the
    // sum, so the parent SizedBox must be sized at `slot * 13` exactly to
    // avoid a Row overflow.
    const cellSize = 48.0;
    const cellGap = 1.0;
    const slot = cellSize + cellGap;
    const gridDim = slot * 13;

    return Center(
      child: SizedBox(
        width: gridDim,
        height: gridDim,
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onPanStart: (d) {
            // Clicking always fires onTapDown first; if a click drifts into
            // a drag, onPanStart fires a moment later. Without this guard,
            // onPanStart would re-run the toggle and cancel the click.
            if (_gestureStartedByTap) {
              _gestureStartedByTap = false;
              return;
            }
            final row = (d.localPosition.dy / slot).floor();
            final col = (d.localPosition.dx / slot).floor();
            if (row < 0 || row > 12 || col < 0 || col > 12) return;
            _onPanStart(row, col);
          },
          onPanUpdate: (d) {
            final row = (d.localPosition.dy / slot).floor();
            final col = (d.localPosition.dx / slot).floor();
            if (row < 0 || row > 12 || col < 0 || col > 12) return;
            _onPanContinue(row, col);
          },
          onPanEnd: (_) {
            _dragValue = null;
            _lastDragRow = null;
            _lastDragCol = null;
            _gestureStartedByTap = false;
          },
          onTapDown: (d) {
            final row = (d.localPosition.dy / slot).floor();
            final col = (d.localPosition.dx / slot).floor();
            if (row < 0 || row > 12 || col < 0 || col > 12) return;
            _onPanStart(row, col);
            _gestureStartedByTap = true;
          },
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              for (int r = 0; r < 13; r++)
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    for (int c = 0; c < 13; c++)
                      _gridCell(r, c, cellSize),
                  ],
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _gridCell(int row, int col, double size) {
    final label = _label(row, col);
    final w = _weights[label] ?? 0.0;
    final isPair = row == col;
    final isSuited = row < col;
    // Background color shifts with weight; type tint hints at category.
    final Color baseTint = isPair
        ? const Color(0xFF6F7CA0) // pair — cool
        : (isSuited
            ? const Color(0xFF5FB37A) // suited — green
            : const Color(0xFFD47B3F)); // offsuit — orange
    final fillAlpha = (w * 0.92).clamp(0.0, 1.0);
    final fill = Color.lerp(const Color(0xFF34383B), baseTint, fillAlpha)!;
    return Container(
      width: size,
      height: size,
      margin: const EdgeInsets.all(0.5),
      decoration: BoxDecoration(
        color: fill,
        border: Border.all(
          color: w > 0.001
              ? const Color(0xFFEAE6D9).withValues(alpha: 0.35)
              : const Color(0xFF4A4E52),
          width: 0.8,
        ),
        borderRadius: BorderRadius.circular(3),
      ),
      child: Stack(
        children: [
          Positioned.fill(
            child: Center(
              child: Text(
                label,
                style: TextStyle(
                  color: w > 0.5
                      ? Colors.white
                      : const Color(0xFFEAE6D9).withValues(alpha: 0.85),
                  fontSize: 12,
                  fontFamily: 'monospace',
                  fontWeight: FontWeight.w700,
                  shadows: const [
                    Shadow(
                      color: Color(0x77000000),
                      blurRadius: 2,
                      offset: Offset(0, 1),
                    ),
                  ],
                ),
              ),
            ),
          ),
          if (w > 0.001 && w < 0.999)
            Positioned(
              right: 3,
              bottom: 2,
              child: Text(
                '${(w * 100).round()}',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 9,
                  fontFamily: 'monospace',
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _brushControls() {
    Widget chip(double v, String label) {
      final selected = (_brush - v).abs() < 1e-3;
      return Padding(
        padding: const EdgeInsets.only(right: 6),
        child: InkWell(
          onTap: () => setState(() => _brush = v),
          borderRadius: BorderRadius.circular(4),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: selected ? const Color(0xFF1F3A56) : const Color(0xFF2F3236),
              border: Border.all(
                color: selected ? const Color(0xFF6FB3DC) : const Color(0xFF4A4E52),
                width: 1.2,
              ),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              label,
              style: const TextStyle(
                color: Color(0xFFEAE6D9),
                fontFamily: 'monospace',
                fontSize: 13,
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
        ),
      );
    }

    return Row(
      children: [
        const Text(
          'BRUSH',
          style: TextStyle(
            color: Color(0x99EAE6D9),
            fontSize: 11,
            letterSpacing: 1.6,
            fontWeight: FontWeight.w800,
          ),
        ),
        const SizedBox(width: 12),
        chip(0.0, '0%'),
        chip(0.25, '25%'),
        chip(0.50, '50%'),
        chip(0.75, '75%'),
        chip(1.00, '100%'),
        const SizedBox(width: 14),
        Expanded(
          child: SliderTheme(
            data: SliderTheme.of(context).copyWith(
              activeTrackColor: const Color(0xFF6FB3DC),
              inactiveTrackColor: const Color(0xFF4A4E52),
              thumbColor: const Color(0xFFEAE6D9),
            ),
            child: Slider(
              value: _brush,
              min: 0.0,
              max: 1.0,
              divisions: 20,
              label: '${(_brush * 100).round()}%',
              onChanged: (v) => setState(() => _brush = v),
            ),
          ),
        ),
        Text(
          '${(_brush * 100).round()}%',
          style: const TextStyle(
            color: Color(0xFFEAE6D9),
            fontFamily: 'monospace',
            fontSize: 13,
            fontWeight: FontWeight.w800,
          ),
        ),
      ],
    );
  }

  Widget _statsAndPreview() {
    final pct = _rangePercent();
    final preview = serializeClassWeights(_weights);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            const Text(
              'PREVIEW',
              style: TextStyle(
                color: Color(0x99EAE6D9),
                fontSize: 11,
                letterSpacing: 1.6,
                fontWeight: FontWeight.w800,
              ),
            ),
            const Spacer(),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: const Color(0x336FB3DC),
                border: Border.all(color: const Color(0xFF6FB3DC), width: 1),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                '${pct.toStringAsFixed(1)}% of all hands',
                style: const TextStyle(
                  color: Color(0xFF6FB3DC),
                  fontSize: 12,
                  fontWeight: FontWeight.w800,
                  fontFamily: 'monospace',
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        Container(
          padding: const EdgeInsets.all(10),
          constraints: const BoxConstraints(minHeight: 60, maxHeight: 100),
          decoration: BoxDecoration(
            color: const Color(0xFF1B1F22),
            border: Border.all(color: const Color(0xFF4A4E52)),
            borderRadius: BorderRadius.circular(4),
          ),
          child: SingleChildScrollView(
            child: SelectableText(
              preview.isEmpty ? '(empty range)' : preview,
              style: const TextStyle(
                color: Color(0xCCEAE6D9),
                fontFamily: 'monospace',
                fontSize: 12,
                height: 1.4,
              ),
            ),
          ),
        ),
      ],
    );
  }

  /// Approx % of the 1326 total NLHE preflop combos covered by the current
  /// weighted range. Combos per class:
  ///   pair   = 6 (e.g. AA = AhAs/AhAd/AhAc/AsAd/AsAc/AdAc)
  ///   suited = 4
  ///   offsuit = 12
  double _rangePercent() {
    var combos = 0.0;
    _weights.forEach((cls, w) {
      final n = cls.length == 2 ? 6 : (cls.endsWith('s') ? 4 : 12);
      combos += n * w;
    });
    return 100.0 * combos / 1326.0;
  }

  Widget _footer() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 10, 20, 14),
      child: Row(
        children: [
          const Text(
            'Click & drag to paint cells. Use brush 0% to erase.',
            style: TextStyle(
              color: Color(0x88EAE6D9),
              fontSize: 12,
              fontStyle: FontStyle.italic,
            ),
          ),
          const Spacer(),
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel'),
          ),
          const SizedBox(width: 8),
          FilledButton.icon(
            icon: const Icon(Icons.check, size: 18),
            label: const Text('Apply'),
            onPressed: () {
              widget.onApply(serializeClassWeights(_weights));
              Navigator.of(context).pop();
            },
          ),
        ],
      ),
    );
  }
}
