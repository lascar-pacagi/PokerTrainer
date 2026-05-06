/// Modal range viewer — 13×13 hand grids for OOP and IP.
///
/// Data source is the active scenario's ROOT weights. For a top-level
/// scenario those are the preflop ranges (the solver's input); for a
/// subgame they're the conditional ranges. We surface that distinction
/// with a banner so the user knows what they're looking at.
library;

import 'package:flutter/material.dart';

import '../../hand_class.dart';
import '../../scenario.dart';

class TrainerRangeViewer extends StatelessWidget {
  final Scenario scenario;
  const TrainerRangeViewer({super.key, required this.scenario});

  /// Open as a modal dialog. Convenience helper so the screen layer just
  /// calls `TrainerRangeViewer.show(context, scenario)`.
  static Future<void> show(BuildContext ctx, Scenario s) =>
      showDialog<void>(
        context: ctx,
        builder: (_) => Dialog(
          backgroundColor: const Color(0xFF1B1E20),
          insetPadding: const EdgeInsets.all(24),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: TrainerRangeViewer(scenario: s),
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
      constraints: const BoxConstraints(maxWidth: 980, maxHeight: 720),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
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
    return Container(
      decoration: BoxDecoration(
        color: faded
            ? const Color(0xFF24282B)
            : accent.withValues(alpha: 0.18 + 0.55 * intensity),
        border: Border.all(
          color: faded ? const Color(0xFF3A3F42) : accent.withValues(alpha: 0.7),
          width: 0.5,
        ),
      ),
      child: FittedBox(
        fit: BoxFit.scaleDown,
        child: Padding(
          padding: const EdgeInsets.all(2),
          child: Text(
            label,
            style: TextStyle(
              color: faded
                  ? const Color(0x55EAE6D9)
                  : const Color(0xFFEAE6D9),
              fontFamily: 'monospace',
              fontSize: 10,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ),
    );
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
