/// One button per legal action. Labels are derived from the engine's snapped
/// sizing so what the user sees is what step() will apply.
library;

import 'package:flutter/material.dart';

import '../ffi/actions.dart';
import '../ffi/engine.dart';
import '../game/game_session.dart';

class ActionBar extends StatelessWidget {
  final GameSession session;
  const ActionBar({super.key, required this.session});

  @override
  Widget build(BuildContext context) {
    final obs   = session.observation;
    final table = session.tableState;

    if (obs == null) {
      return _disabledBar('Hand is over — deal a new hand.');
    }

    final agent = session.agentFor(table.toAct!);
    final isModelTurn = agent is ModelAgent;

    // If a model is to act AND auto-step is on, the model has either already
    // stepped (and we're in a transitional frame) or it will step before the
    // next paint. Show a "thinking" banner.
    if (isModelTurn && session.autoStep) {
      return _disabledBar('${table.toAct!.shortLabel}: ${agent.label} acting…');
    }

    final argmax = session.strategy?.argmaxLegalIdx;
    final qs     = session.strategy?.qValues;
    // Auto-detect probability mode (CFR PolicyNet shim) so we can display
    // labels as percentages in the action buttons too.
    final isProbability = qs != null && _isProbabilityDistribution(qs);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF34383B),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: isModelTurn
              ? const Color(0xFFFFD24A)        // gold border = "model's turn,
              : const Color(0xFF4A4E52),       //               you're driving"
          width: isModelTurn ? 1.5 : 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (isModelTurn) ...[
            // Manual-step mode: tell the user what's happening + give a
            // "take model's pick" shortcut.
            Padding(
              padding: const EdgeInsets.fromLTRB(2, 2, 2, 8),
              child: Row(
                children: [
                  const Icon(Icons.pause_circle,
                      size: 18, color: Color(0xFFFFD24A)),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      '${table.toAct!.shortLabel}: ${agent.label} '
                      'is to act — choose any action below, or take the '
                      'model\'s pick.',
                      style: const TextStyle(
                        color: Color(0xCCEAE6D9),
                        fontSize: 13,
                        fontStyle: FontStyle.italic,
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  FilledButton.icon(
                    icon: const Icon(Icons.play_arrow, size: 16),
                    label: const Text("Step model's pick"),
                    onPressed: () => session.stepModelOnce(),
                  ),
                ],
              ),
            ),
          ],
          Wrap(
            spacing: 8,
            runSpacing: 8,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              for (int i = 0; i < obs.legal.length; i++)
                _ActionButton(
                  label: _labelFor(obs.legal[i], obs.sizingForLegal(i), table),
                  subLabel: qs != null
                      ? (isProbability
                          ? '${(qs[i] * 100).toStringAsFixed(qs[i] >= 0.10 ? 0 : 1)}%'
                          : 'Q ${qs[i].toStringAsFixed(2)}')
                      : null,
                  isArgmax: argmax == i,
                  isFold: obs.legal[i] == ActionType.fold,
                  isAllIn: obs.legal[i] == ActionType.allIn,
                  onTap: () => session.applyLegal(i),
                ),
            ],
          ),
        ],
      ),
    );
  }

  /// Mirror of `_isProbabilityDistribution` in main.dart — kept private here
  /// so widgets/action_bar doesn't need to import the app shell. Heuristic:
  /// values in [-eps, 1+eps] AND sum ≈ 1.
  static bool _isProbabilityDistribution(dynamic vs) {
    final n = vs.length as int;
    if (n == 0) return false;
    const eps = 1e-3;
    double sum = 0.0;
    for (var i = 0; i < n; i++) {
      final v = (vs[i] as num).toDouble();
      if (v < -eps || v > 1.0 + eps) return false;
      sum += v;
    }
    return (sum - 1.0).abs() <= 1e-2;
  }

  Widget _disabledBar(String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 22),
      decoration: BoxDecoration(
        color: const Color(0xFF34383B),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Center(
        child: Text(
          text,
          style: const TextStyle(
            color: Color(0xAAEAE6D9),
            fontStyle: FontStyle.italic,
            fontSize: 15,
          ),
        ),
      ),
    );
  }

  /// "Fold" / "Call X bb" / "R 25% (to Y bb)" / "All-in (Z bb)".
  String _labelFor(ActionType type, int sizingChips, TableState table) {
    if (type == ActionType.fold) return 'Fold';
    if (type == ActionType.checkCall) {
      return table.toCallChips == 0
          ? 'Check'
          : 'Call ${(table.toCallChips / TableState.chipsPerBb).toStringAsFixed(1)} bb';
    }
    final toBb = sizingChips / TableState.chipsPerBb;
    if (type == ActionType.allIn) {
      return 'All-in (${toBb.toStringAsFixed(1)} bb)';
    }
    return '${type.shortLabel} (${toBb.toStringAsFixed(1)} bb)';
  }
}

class _ActionButton extends StatelessWidget {
  final String label;
  final String? subLabel;     // optional Q-value, shown small
  final bool isArgmax;
  final bool isFold;
  final bool isAllIn;
  final VoidCallback onTap;

  const _ActionButton({
    required this.label,
    required this.onTap,
    this.subLabel,
    this.isArgmax = false,
    this.isFold = false,
    this.isAllIn = false,
  });

  @override
  Widget build(BuildContext context) {
    final base = isFold
        ? const Color(0xFF7A2A2A)
        : isAllIn
            ? const Color(0xFFA37212)
            : const Color(0xFF2D5B7C);

    return Material(
      color: base,
      borderRadius: BorderRadius.circular(7),
      elevation: 1,
      child: InkWell(
        borderRadius: BorderRadius.circular(7),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
          decoration: isArgmax
              ? BoxDecoration(
                  borderRadius: BorderRadius.circular(7),
                  border: Border.all(color: const Color(0xFFFFD24A), width: 2.5),
                )
              : null,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Text(
                label,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                  fontSize: 16,
                ),
              ),
              if (subLabel != null) ...[
                const SizedBox(height: 3),
                Text(
                  subLabel!,
                  style: const TextStyle(
                    color: Color(0xEEFFFFFF),
                    fontFamily: 'monospace',
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
