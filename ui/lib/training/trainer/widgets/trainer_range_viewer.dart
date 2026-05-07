/// Modal range viewer — 13×13 hand grids for OOP and IP.
///
/// Data source is the active scenario's ROOT weights. For a top-level
/// scenario those are the preflop ranges (the solver's input); for a
/// subgame they're the conditional ranges. We surface that distinction
/// with a banner so the user knows what they're looking at.
///
/// Optionally renders a hand-context strip at the top: board cards, your
/// hole cards, pot. Lets users study ranges without losing sight of the
/// situation they're deciding in.
library;

import 'package:flutter/material.dart';

import '../../../ffi/actions.dart';
import '../../../widgets/card_view.dart';
import '../../hand_class.dart';
import '../../scenario.dart';

/// Optional decision context for the range viewer.
///
/// When supplied, the modal renders the board, your hole cards, and pot at
/// the top so you can think about your decision while studying the ranges
/// — no need to close the dialog and re-open it to glance at the table.
class TrainerRangeContext {
  /// Cards on the board (3, 4, or 5 strings like "Td"). Empty = preflop.
  final List<String> board;
  /// Your hole cards as a 4-char combo string ("AsKh"). Empty hides the slot.
  final String yourHand;
  /// 'oop' or 'ip' — used for the seat label next to your hand.
  final String yourSeat;
  /// Current pot in chips.
  final int pot;

  const TrainerRangeContext({
    required this.board,
    required this.yourHand,
    required this.yourSeat,
    required this.pot,
  });
}

class TrainerRangeViewer extends StatelessWidget {
  final Scenario scenario;
  final TrainerRangeContext? handContext;

  const TrainerRangeViewer({
    super.key,
    required this.scenario,
    this.handContext,
  });

  /// Open as a modal dialog. Convenience helper so the screen layer just
  /// calls `TrainerRangeViewer.show(context, scenario, handContext: ctx)`.
  static Future<void> show(
    BuildContext ctx,
    Scenario s, {
    TrainerRangeContext? handContext,
  }) =>
      showDialog<void>(
        context: ctx,
        builder: (_) => Dialog(
          backgroundColor: const Color(0xFF1B1E20),
          insetPadding: const EdgeInsets.all(24),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: TrainerRangeViewer(scenario: s, handContext: handContext),
          ),
        ),
      );

  @override
  Widget build(BuildContext context) {
    final root = scenario.root;
    final oopWeights =
        root.player == 'oop' ? root.weights : root.weightsOpp;
    final ipWeights =
        root.player == 'ip' ? root.weights : root.weightsOpp;

    return ConstrainedBox(
      // Slightly wider + taller: width gives each cell ~3px more so the
      // corner-percentage stays legible without crowding the centre label;
      // height accommodates the new context strip on top.
      constraints: const BoxConstraints(maxWidth: 1080, maxHeight: 820),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          if (handContext != null) ...[
            _ContextStrip(ctx: handContext!),
            const SizedBox(height: 12),
          ],
          _banner(),
          const SizedBox(height: 12),
          Expanded(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: _RangeChart(
                    title: 'OOP / BB range',
                    accent: const Color(0xFF6FB3DC),
                    combos: scenario.oopCombos,
                    weights: oopWeights,
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: _RangeChart(
                    title: 'IP / SB range',
                    accent: const Color(0xFFDCBE6F),
                    combos: scenario.ipCombos,
                    weights: ipWeights,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.centerRight,
            child: TextButton.icon(
              onPressed: () => Navigator.of(context).maybePop(),
              icon: const Icon(Icons.close, size: 16),
              label: const Text('Close'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _banner() {
    final isSubgame = scenario.parent != null;
    if (isSubgame) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: const Color(0x33D4B43F),
          border: Border.all(color: const Color(0xFFD4B43F), width: 1),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Row(
          children: [
            const Icon(Icons.alt_route,
                color: Color(0xFFD4B43F), size: 18),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                'CURRENT (CONDITIONAL) RANGES — '
                'subgame from ${scenario.parent!.parentLine.join(" › ")} → '
                '${scenario.parent!.pickedCard}. '
                'Different from preflop ranges.',
                style: const TextStyle(
                  color: Color(0xFFEAE6D9),
                  fontSize: 12,
                  height: 1.4,
                ),
              ),
            ),
          ],
        ),
      );
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF1F3A56),
        border: Border.all(color: const Color(0xFF6FB3DC), width: 1),
        borderRadius: BorderRadius.circular(6),
      ),
      child: const Row(
        children: [
          Icon(Icons.account_tree_outlined,
              color: Color(0xFF6FB3DC), size: 18),
          SizedBox(width: 10),
          Expanded(
            child: Text(
              'PREFLOP RANGES — input to the solver. These are the '
              'unconditional starting ranges for both seats.',
              style: TextStyle(
                color: Color(0xFFEAE6D9),
                fontSize: 12,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _RangeChart extends StatelessWidget {
  final String title;
  final Color accent;
  final List<String> combos;
  final List<double> weights;

  const _RangeChart({
    required this.title,
    required this.accent,
    required this.combos,
    required this.weights,
  });

  @override
  Widget build(BuildContext context) {
    final cells = _aggregate(combos, weights);
    // Find max weight for normalisation — colour intensity = w / maxW.
    var maxW = 0.0;
    for (final w in cells) {
      if (w > maxW) maxW = w;
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          title,
          style: TextStyle(
            color: accent,
            fontSize: 13,
            fontWeight: FontWeight.w800,
            letterSpacing: 1.0,
          ),
        ),
        const SizedBox(height: 6),
        AspectRatio(
          aspectRatio: 1,
          child: LayoutBuilder(
            builder: (ctx, cs) {
              final cellSize = cs.maxWidth / 13;
              return Stack(
                children: [
                  for (int r = 0; r < 13; r++)
                    for (int c = 0; c < 13; c++)
                      Positioned(
                        left: c * cellSize,
                        top: r * cellSize,
                        width: cellSize,
                        height: cellSize,
                        child: _cell(r, c, cells, maxW),
                      ),
                ],
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _cell(int r, int c, List<double> cells, double maxW) {
    final idx = r * 13 + c;
    final w = cells[idx];
    final intensity = maxW > 0 ? (w / maxW).clamp(0.0, 1.0) : 0.0;
    final faded = w <= 1e-6;
    final label = _cellLabel(r, c);
    // Combo-count semantics: pair = 6, suited = 4, offsuit = 12. Used both
    // for the "% of class" tooltip line and for the absolute-combos line.
    final isPair = r == c;
    final isSuited = r < c;
    final maxCombos = isPair ? 6 : (isSuited ? 4 : 12);
    final pctOfClass = (w / maxCombos).clamp(0.0, 1.0) * 100;
    final tooltipMessage = faded
        ? '$label  ·  not in range'
        : '$label  ·  ${pctOfClass.toStringAsFixed(0)}% in range '
            '(${w.toStringAsFixed(2)} / $maxCombos combos)';
    // Background gets darker for low weights, more saturated for high.
    // Label is always white on a coloured background for max legibility.
    final bg = faded
        ? const Color(0xFF24282B)
        : accent.withValues(alpha: 0.22 + 0.62 * intensity);
    return Tooltip(
      message: tooltipMessage,
      waitDuration: const Duration(milliseconds: 250),
      child: Container(
      decoration: BoxDecoration(
        color: bg,
        border: Border.all(
          color: faded ? const Color(0xFF3A3F42) : accent.withValues(alpha: 0.75),
          width: 0.5,
        ),
      ),
      child: Stack(
        children: [
          // Centered class label — primary identifier.
          Positioned.fill(
            child: Center(
              child: FittedBox(
                fit: BoxFit.scaleDown,
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 2),
                  child: Text(
                    label,
                    style: TextStyle(
                      color: faded
                          ? const Color(0x66EAE6D9)
                          : Colors.white,
                      fontFamily: 'monospace',
                      fontSize: 12,
                      fontWeight: FontWeight.w800,
                      // Drop shadow keeps the label readable when the
                      // background gets bright at high weights.
                      shadows: const [
                        Shadow(
                          color: Color(0xCC000000),
                          blurRadius: 2,
                          offset: Offset(0, 1),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
          // Weight percentage in the bottom-right corner — only shown for
          // partial weights (skip pure 0% and 100% cells; their state is
          // already obvious from the colour). Matches the PokerCoaching
          // PDF chart convention.
          if (w > 1e-3 && w < 0.999)
            Positioned(
              right: 2,
              bottom: 1,
              child: Text(
                _formatWeight(w),
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 9,
                  fontFamily: 'monospace',
                  fontWeight: FontWeight.w900,
                  shadows: [
                    Shadow(
                      color: Color(0xDD000000),
                      blurRadius: 2,
                      offset: Offset(0, 1),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
      ),
    );
  }

  /// Compact weight format: "50" for 0.5, "75" for 0.75, "5" for 0.05, etc.
  /// Two digits max so the corner stays unobtrusive next to the centred
  /// hand label.
  static String _formatWeight(double w) {
    final pct = (w * 100).round();
    return '$pct';
  }

  /// 169-cell weight aggregation. Reuses classifyCombo from hand_class.dart.
  List<double> _aggregate(List<String> combos, List<double> weights) {
    final out = List<double>.filled(169, 0.0);
    for (var i = 0; i < combos.length; i++) {
      if (i >= weights.length) break;
      final w = weights[i];
      if (w <= 0) continue;
      final cls = classifyCombo(combos[i]);
      out[cls.row * 13 + cls.col] += w;
    }
    return out;
  }

  static String _cellLabel(int r, int c) {
    const ranks = 'AKQJT98765432';
    final hi = ranks[r];
    final lo = ranks[c];
    if (r == c) return '$hi$hi';
    if (r < c) return '$hi${lo}s';
    return '$lo${hi}o';
  }
}

/// Compact "what's on the table right now" strip for the range viewer.
/// Renders board + your hole cards + pot in a single horizontal row so the
/// modal stays compact while showing all the context needed to think
/// through a decision.
class _ContextStrip extends StatelessWidget {
  final TrainerRangeContext ctx;
  const _ContextStrip({required this.ctx});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF1A2D27),
        border: Border.all(color: const Color(0xFF345A4A), width: 1),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          // Board first — left-aligned, with a tiny "BOARD" label.
          _section(
            label: ctx.board.isEmpty ? 'PREFLOP' : 'BOARD',
            child: ctx.board.isEmpty
                ? const Text(
                    '— no board yet —',
                    style: TextStyle(
                      color: Color(0x99EAE6D9),
                      fontStyle: FontStyle.italic,
                      fontSize: 12,
                    ),
                  )
                : Wrap(
                    spacing: 5,
                    children: [
                      for (final c in ctx.board)
                        CardView(card: EngineCard.parse(c), width: 38),
                    ],
                  ),
          ),
          const SizedBox(width: 18),
          const _Sep(),
          const SizedBox(width: 18),
          // Pot — small fixed-width chip.
          _section(
            label: 'POT',
            child: Text(
              '${ctx.pot}',
              style: const TextStyle(
                color: Color(0xFFEAE6D9),
                fontFamily: 'monospace',
                fontSize: 16,
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
          const Spacer(),
          const _Sep(),
          const SizedBox(width: 18),
          // Your hand — right-aligned with the seat colour the rest of the
          // trainer uses (cyan for OOP/BB, amber for IP/SB).
          _section(
            label: 'YOU '
                '(${ctx.yourSeat.toUpperCase()}/'
                '${ctx.yourSeat == 'oop' ? 'BB' : 'SB'})',
            accent: ctx.yourSeat == 'oop'
                ? const Color(0xFF6FB3DC)
                : const Color(0xFFDCBE6F),
            child: ctx.yourHand.length == 4
                ? Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      CardView(card: _parseSafe(ctx.yourHand.substring(0, 2)),
                          width: 38),
                      const SizedBox(width: 5),
                      CardView(card: _parseSafe(ctx.yourHand.substring(2, 4)),
                          width: 38),
                    ],
                  )
                : const Text(
                    '— not dealt —',
                    style: TextStyle(
                      color: Color(0x99EAE6D9),
                      fontStyle: FontStyle.italic,
                      fontSize: 12,
                    ),
                  ),
          ),
        ],
      ),
    );
  }

  Widget _section({
    required String label,
    required Widget child,
    Color? accent,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label,
          style: TextStyle(
            color: accent ?? const Color(0x99EAE6D9),
            fontSize: 10,
            letterSpacing: 1.4,
            fontWeight: FontWeight.w800,
          ),
        ),
        const SizedBox(height: 4),
        child,
      ],
    );
  }

  /// EngineCard.parse can throw on malformed input (e.g. partially-filled
  /// combo strings during state transitions). Fall back to noCard so the
  /// modal renders an empty slot rather than crashing.
  static int _parseSafe(String s) {
    try {
      return EngineCard.parse(s);
    } catch (_) {
      return EngineCard.noCard;
    }
  }
}

/// Vertical hairline divider between sections of the context strip.
class _Sep extends StatelessWidget {
  const _Sep();
  @override
  Widget build(BuildContext context) {
    return Container(
      width: 1,
      height: 48,
      color: const Color(0xFF345A4A),
    );
  }
}
