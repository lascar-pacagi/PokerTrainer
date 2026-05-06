/// Modal "configure & solve a new spot" dialog.
///
/// Sections (top to bottom):
///   1. Ranges     — multi-line text fields (PokerStove syntax).
///   2. Board      — flop string + optional turn/river.
///   3. Pot/stack  — int fields.
///   4. Bet sizes  — bet/raise strings per street.
///   5. Solver     — iterations + target exploitability.
///   6. Output     — Estimate panel + Solve / Cancel buttons.
///
/// Range text validation is delegated to pt-solver; we surface its error
/// message in red rather than re-implementing PokerStove parsing in Dart.
library;

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../solver_runner.dart';
import 'range_editor_dialog.dart';

class SpotConfigDialog extends StatefulWidget {
  final SolverRunner runner;
  final SpotConfig initial;
  /// Called with the resulting JSON path after a successful solve.
  final void Function(String path) onSolved;

  const SpotConfigDialog({
    super.key,
    required this.runner,
    required this.initial,
    required this.onSolved,
  });

  @override
  State<SpotConfigDialog> createState() => _SpotConfigDialogState();
}

class _SpotConfigDialogState extends State<SpotConfigDialog> {
  late final SpotConfig _cfg;
  // Controllers for fields that aren't simple int → spinner.
  late final TextEditingController _oop;
  late final TextEditingController _ip;
  late final TextEditingController _flop;
  late final TextEditingController _turn;
  late final TextEditingController _river;
  late final TextEditingController _pot;
  late final TextEditingController _stack;
  late final TextEditingController _flopBet;
  late final TextEditingController _flopRaise;
  late final TextEditingController _turnBet;
  late final TextEditingController _turnRaise;
  late final TextEditingController _riverBet;
  late final TextEditingController _riverRaise;
  late final TextEditingController _iters;
  late final TextEditingController _target;

  Timer? _debounceEstimate;

  @override
  void initState() {
    super.initState();
    _cfg = widget.initial;
    _oop = TextEditingController(text: _cfg.oopRange);
    _ip = TextEditingController(text: _cfg.ipRange);
    _flop = TextEditingController(text: _cfg.flop);
    _turn = TextEditingController(text: _cfg.turn ?? '');
    _river = TextEditingController(text: _cfg.river ?? '');
    _pot = TextEditingController(text: '${_cfg.startingPot}');
    _stack = TextEditingController(text: '${_cfg.effectiveStack}');
    _flopBet = TextEditingController(text: _cfg.flopBet);
    _flopRaise = TextEditingController(text: _cfg.flopRaise);
    _turnBet = TextEditingController(text: _cfg.turnBet);
    _turnRaise = TextEditingController(text: _cfg.turnRaise);
    _riverBet = TextEditingController(text: _cfg.riverBet);
    _riverRaise = TextEditingController(text: _cfg.riverRaise);
    _iters = TextEditingController(text: '${_cfg.maxIterations}');
    _target = TextEditingController(text: '${_cfg.targetExploitabilityPctPot}');

    widget.runner.addListener(_runnerChanged);
    // Auto-estimate on open so the user sees memory immediately.
    WidgetsBinding.instance.addPostFrameCallback((_) => _scheduleEstimate(immediate: true));
  }

  @override
  void dispose() {
    _debounceEstimate?.cancel();
    widget.runner.removeListener(_runnerChanged);
    for (final c in [
      _oop, _ip, _flop, _turn, _river, _pot, _stack,
      _flopBet, _flopRaise, _turnBet, _turnRaise, _riverBet, _riverRaise,
      _iters, _target,
    ]) {
      c.dispose();
    }
    super.dispose();
  }

  void _runnerChanged() {
    if (mounted) setState(() {});
  }

  /// Sync controller values back into _cfg.
  void _flush() {
    _cfg.oopRange = _oop.text.trim();
    _cfg.ipRange = _ip.text.trim();
    _cfg.flop = _flop.text.trim();
    _cfg.turn = _turn.text.trim().isEmpty ? null : _turn.text.trim();
    _cfg.river = _river.text.trim().isEmpty ? null : _river.text.trim();
    _cfg.startingPot = int.tryParse(_pot.text.trim()) ?? _cfg.startingPot;
    _cfg.effectiveStack = int.tryParse(_stack.text.trim()) ?? _cfg.effectiveStack;
    _cfg.flopBet = _flopBet.text.trim();
    _cfg.flopRaise = _flopRaise.text.trim();
    _cfg.turnBet = _turnBet.text.trim();
    _cfg.turnRaise = _turnRaise.text.trim();
    _cfg.riverBet = _riverBet.text.trim();
    _cfg.riverRaise = _riverRaise.text.trim();
    _cfg.maxIterations = int.tryParse(_iters.text.trim()) ?? _cfg.maxIterations;
    _cfg.targetExploitabilityPctPot =
        double.tryParse(_target.text.trim()) ?? _cfg.targetExploitabilityPctPot;
  }

  /// Debounce-friendly estimate. Most edits are character-by-character, and
  /// each one would launch pt-solver. Wait 500ms idle before launching.
  void _scheduleEstimate({bool immediate = false}) {
    _debounceEstimate?.cancel();
    _flush();
    if (immediate) {
      widget.runner.estimate(_cfg);
    } else {
      _debounceEstimate = Timer(const Duration(milliseconds: 500), () {
        widget.runner.estimate(_cfg);
      });
    }
  }

  Future<void> _solve() async {
    _flush();
    final path = await widget.runner.solve(_cfg);
    if (path != null && mounted) {
      widget.onSolved(path);
      Navigator.of(context).pop();
    }
  }

  @override
  Widget build(BuildContext context) {
    final r = widget.runner;
    // The dialog mounts under the MaterialApp's root Overlay, so it inherits
    // the root MediaQuery (scaler 1.0) — *not* the training tab's 1.32.
    // Override here so the dialog matches the rest of the Training tab.
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
        constraints: const BoxConstraints(maxWidth: 760, maxHeight: 800),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _header(),
            const Divider(height: 1, color: Color(0xFF4A4E52)),
            Flexible(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _section('RANGES'),
                    // OOP/IP is the data label; BB/SB matches the positional
                    // names HU players think in (BB acts first postflop = OOP,
                    // SB acts last postflop = IP).
                    _rangeFieldWithEditor('OOP / BB', _oop),
                    const SizedBox(height: 8),
                    _rangeFieldWithEditor('IP / SB', _ip),
                    const SizedBox(height: 16),
                    _section('BOARD'),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(child: _textField('Flop (e.g. Td9d6h)', _flop)),
                        const SizedBox(width: 10),
                        SizedBox(width: 110, child: _textField('Turn (opt.)', _turn)),
                        const SizedBox(width: 10),
                        SizedBox(width: 110, child: _textField('River (opt.)', _river)),
                      ],
                    ),
                    const SizedBox(height: 16),
                    _section('POT & STACK (chips)'),
                    Row(
                      children: [
                        Expanded(child: _textField('Starting pot', _pot, intOnly: true)),
                        const SizedBox(width: 10),
                        Expanded(child: _textField('Effective stack', _stack, intOnly: true)),
                      ],
                    ),
                    const SizedBox(height: 16),
                    _section('BET SIZES'),
                    _betSizeHelp(),
                    const SizedBox(height: 8),
                    _streetSizeRow('Flop', _flopBet, _flopRaise),
                    const SizedBox(height: 8),
                    _streetSizeRow('Turn', _turnBet, _turnRaise),
                    const SizedBox(height: 8),
                    _streetSizeRow('River', _riverBet, _riverRaise),
                    const SizedBox(height: 16),
                    _section('SOLVER'),
                    Row(
                      children: [
                        Expanded(child: _textField('Max iterations', _iters, intOnly: true)),
                        const SizedBox(width: 10),
                        Expanded(child: _textField('Target expl. (% pot)', _target)),
                      ],
                    ),
                    const SizedBox(height: 12),
                    _section('TREE DEPTH'),
                    _depthSelector(),
                  ],
                ),
              ),
            ),
            const Divider(height: 1, color: Color(0xFF4A4E52)),
            _estimatePanel(r),
            _footerButtons(r),
          ],
        ),
      ),
      ),
    );
  }

  Widget _header() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 16, 12, 12),
      child: Row(
        children: [
          const Icon(Icons.tune, size: 20, color: Color(0xFF6FB3DC)),
          const SizedBox(width: 10),
          const Text(
            'New solve',
            style: TextStyle(
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

  Widget _section(String label) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Text(
        label,
        style: const TextStyle(
          color: Color(0x99EAE6D9),
          fontSize: 11,
          letterSpacing: 1.6,
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }

  /// Range field plus an "Edit visually" button that pops the 13×13 grid
  /// editor. The editor takes the field's current text as its starting state
  /// (parsed best-effort) and writes back the serialized result on Apply.
  Widget _rangeFieldWithEditor(String label, TextEditingController c) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Expanded(child: _rangeField(label, c)),
        const SizedBox(width: 8),
        Tooltip(
          message: 'Open visual range editor (13×13 grid + presets)',
          child: OutlinedButton.icon(
            icon: const Icon(Icons.grid_view, size: 16),
            label: const Text('Edit'),
            style: OutlinedButton.styleFrom(
              foregroundColor: const Color(0xFFEAE6D9),
              side: const BorderSide(color: Color(0xFF6FB3DC), width: 1.2),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
            ),
            onPressed: () {
              showDialog(
                context: context,
                barrierDismissible: false,
                builder: (_) => RangeEditorDialog(
                  title: 'Edit $label range',
                  initial: c.text,
                  onApply: (str) {
                    setState(() => c.text = str);
                    _scheduleEstimate();
                  },
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _rangeField(String label, TextEditingController c) {
    return TextField(
      controller: c,
      onChanged: (_) => _scheduleEstimate(),
      maxLines: 2,
      style: const TextStyle(
        color: Color(0xFFEAE6D9),
        fontFamily: 'monospace',
        fontSize: 13,
      ),
      decoration: InputDecoration(
        labelText: label,
        labelStyle: const TextStyle(color: Color(0x88EAE6D9), fontSize: 13),
        filled: true,
        fillColor: const Color(0xFF2F3236),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(6),
          borderSide: const BorderSide(color: Color(0xFF4A4E52)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(6),
          borderSide: const BorderSide(color: Color(0xFF6FB3DC), width: 1.4),
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
      ),
    );
  }

  Widget _textField(String label, TextEditingController c, {bool intOnly = false}) {
    return TextField(
      controller: c,
      onChanged: (_) => _scheduleEstimate(),
      keyboardType: intOnly ? TextInputType.number : TextInputType.text,
      inputFormatters: intOnly
          ? [FilteringTextInputFormatter.digitsOnly]
          : null,
      style: const TextStyle(
        color: Color(0xFFEAE6D9),
        fontFamily: 'monospace',
        fontSize: 13,
      ),
      decoration: InputDecoration(
        labelText: label,
        labelStyle: const TextStyle(color: Color(0x88EAE6D9), fontSize: 12),
        filled: true,
        fillColor: const Color(0xFF2F3236),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(6),
          borderSide: const BorderSide(color: Color(0xFF4A4E52)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(6),
          borderSide: const BorderSide(color: Color(0xFF6FB3DC), width: 1.4),
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      ),
    );
  }

  Widget _depthSelector() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        for (final d in SolveDepth.values) _depthRow(d),
      ],
    );
  }

  Widget _depthRow(SolveDepth d) {
    final selected = _cfg.depth == d;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: InkWell(
        onTap: () => setState(() => _cfg.depth = d),
        borderRadius: BorderRadius.circular(6),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
          decoration: BoxDecoration(
            color: selected ? const Color(0xFF1F3A56) : const Color(0xFF2F3236),
            border: Border.all(
              color: selected ? const Color(0xFF6FB3DC) : const Color(0xFF4A4E52),
              width: 1.2,
            ),
            borderRadius: BorderRadius.circular(6),
          ),
          child: Row(
            children: [
              Icon(
                selected ? Icons.radio_button_checked : Icons.radio_button_off,
                color: selected ? const Color(0xFF6FB3DC) : const Color(0x88EAE6D9),
                size: 18,
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      d.human,
                      style: const TextStyle(
                        color: Color(0xFFEAE6D9),
                        fontSize: 13,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _depthBlurb(d),
                      style: const TextStyle(
                        color: Color(0x99EAE6D9),
                        fontSize: 11,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  String _depthBlurb(SolveDepth d) {
    switch (d) {
      case SolveDepth.flop:
        return 'Stop at the turn deal. Smallest file (~150 KB). Fastest dump.';
      case SolveDepth.turn:
        return 'Walk every turn card. ~5–50 MB depending on bet sizes.';
      case SolveDepth.river:
        return 'Walk both turn and river. Hundreds of MB — narrow ranges only.';
    }
  }

  /// Inline cheat sheet explaining the BetSizeOptions DSL. Always visible —
  /// most users won't hover a help icon, but they will read text that's
  /// already there. Synced with `validation/.../bet_size.rs` doc-comment.
  Widget _betSizeHelp() {
    const labelStyle = TextStyle(
      color: Color(0xFFEAE6D9),
      fontSize: 13,
      fontWeight: FontWeight.w800,
      fontFamily: 'monospace',
    );
    const helpStyle = TextStyle(
      color: Color(0xCCEAE6D9),
      fontSize: 12.5,
      height: 1.45,
    );
    const noteStyle = TextStyle(
      color: Color(0x99EAE6D9),
      fontSize: 12,
      height: 1.45,
      fontStyle: FontStyle.italic,
    );

    Widget kv(String k, String v) => Padding(
          padding: const EdgeInsets.only(bottom: 3),
          child: RichText(
            text: TextSpan(
              children: [
                TextSpan(text: '$k  ', style: labelStyle),
                TextSpan(text: v, style: helpStyle),
              ],
            ),
          ),
        );

    return Container(
      padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
      decoration: BoxDecoration(
        color: const Color(0xFF1B1F22),
        border: Border.all(color: const Color(0xFF4A4E52)),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text(
            'How to specify sizings',
            style: TextStyle(
              color: Color(0xCCEAE6D9),
              fontSize: 12.5,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.6,
            ),
          ),
          const SizedBox(height: 6),
          const Text(
            'Each field is a comma-separated list of sizes. The solver tries '
            'all of them at every node and learns the best mix. Suffixes:',
            style: helpStyle,
          ),
          const SizedBox(height: 6),
          kv('50%', '— 50% of the current pot. Any percent works (33%, 75%, 125%).'),
          kv('a', '— all-in.'),
          kv('100c', '— flat 100-chip bet (constant size, ignores pot).'),
          kv('2.5x', '— RAISE only: 2.5× the previous bet.'),
          kv('e / 2e / 3e', '— geometric multi-street sizing (auto-sizes to '
              'go all-in over N streets evenly).'),
          const SizedBox(height: 8),
          const Text(
            'Examples:',
            style: TextStyle(
              color: Color(0xCCEAE6D9),
              fontSize: 12,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            '  Bet "33%, 75%"        — small or big c-bet, mixed by the solver.\n'
            '  Bet "50%, a"          — half-pot or all-in (river polar).\n'
            '  Bet "33%, 75%, 150%"  — three sizes, full polar tree.\n'
            '  Raise "2.5x"          — single re-raise size.\n'
            '  Raise "2x, 3.5x"      — small or large 3-bet.',
            style: TextStyle(
              color: Color(0xCCEAE6D9),
              fontFamily: 'monospace',
              fontSize: 12,
              height: 1.5,
            ),
          ),
          const SizedBox(height: 6),
          const Text(
            'More sizes ⇒ richer strategy but exponentially bigger tree. '
            'Start with one size per street; add more if the solve is fast '
            'enough.',
            style: noteStyle,
          ),
        ],
      ),
    );
  }

  Widget _streetSizeRow(String street, TextEditingController bet, TextEditingController raise) {
    return Row(
      children: [
        SizedBox(
          width: 60,
          child: Text(
            street,
            style: const TextStyle(
              color: Color(0xCCEAE6D9),
              fontSize: 13,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
        Expanded(child: _textField('bet', bet)),
        const SizedBox(width: 10),
        Expanded(child: _textField('raise', raise)),
      ],
    );
  }

  Widget _estimatePanel(SolverRunner r) {
    final est = r.lastEstimate;
    final err = r.lastError;
    final running = r.running;
    final logs = r.stderrLines;

    final Widget body;
    if (err != null) {
      body = Text(
        err,
        style: const TextStyle(
          color: Color(0xFFD47B7B),
          fontFamily: 'monospace',
          fontSize: 12,
        ),
      );
    } else if (est == null && running) {
      body = const Row(
        children: [
          SizedBox(
            width: 14,
            height: 14,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          SizedBox(width: 10),
          Text(
            'Estimating tree size…',
            style: TextStyle(color: Color(0x99EAE6D9), fontSize: 12),
          ),
        ],
      );
    } else if (est != null) {
      final tooLarge = est.tooLarge;
      body = Row(
        children: [
          Icon(
            tooLarge ? Icons.error : Icons.check_circle,
            size: 16,
            color: tooLarge ? const Color(0xFFD47B7B) : const Color(0xFF5FB37A),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              tooLarge
                  ? 'Tree too large: ${est.memoryHuman} > ${est.capHuman} cap. '
                    'Reduce sizings or stacks.'
                  : 'Tree size: ${est.memoryHuman}  '
                    '(${est.memoryCompressedHuman} compressed) — ok.',
              style: TextStyle(
                color: tooLarge
                    ? const Color(0xFFD47B7B)
                    : const Color(0xFFEAE6D9),
                fontSize: 12,
                fontFamily: 'monospace',
              ),
            ),
          ),
        ],
      );
    } else {
      body = const Text(
        'Edit any field to estimate the tree size.',
        style: TextStyle(
          color: Color(0x66EAE6D9),
          fontStyle: FontStyle.italic,
          fontSize: 12,
        ),
      );
    }

    final showLogs = running && logs.isNotEmpty;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      color: const Color(0xFF24282B),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          body,
          if (showLogs) ...[
            const SizedBox(height: 6),
            Container(
              constraints: const BoxConstraints(maxHeight: 80),
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: const Color(0xFF2F3236),
                borderRadius: BorderRadius.circular(4),
                border: Border.all(color: const Color(0xFF4A4E52)),
              ),
              child: SingleChildScrollView(
                reverse: true,
                child: Text(
                  logs.join('\n'),
                  style: const TextStyle(
                    color: Color(0xAAEAE6D9),
                    fontFamily: 'monospace',
                    fontSize: 11,
                    height: 1.4,
                  ),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _footerButtons(SolverRunner r) {
    final est = r.lastEstimate;
    final canSolve = !r.running && est != null && !est.tooLarge;
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 16),
      child: Row(
        children: [
          OutlinedButton.icon(
            icon: const Icon(Icons.refresh, size: 16),
            label: const Text('Estimate now'),
            onPressed: r.running ? null : () => _scheduleEstimate(immediate: true),
          ),
          const Spacer(),
          if (r.running)
            TextButton.icon(
              icon: const Icon(Icons.stop, size: 16),
              label: const Text('Cancel'),
              onPressed: r.cancel,
            )
          else
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Close'),
            ),
          const SizedBox(width: 8),
          FilledButton.icon(
            icon: const Icon(Icons.play_arrow, size: 18),
            label: const Text('Solve'),
            onPressed: canSolve ? _solve : null,
          ),
        ],
      ),
    );
  }
}
