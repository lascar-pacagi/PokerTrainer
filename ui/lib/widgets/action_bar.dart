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
    if (agent is ModelAgent) {
      // Auto-stepping happens in GameSession; this just paints a banner so
      // the user knows the model is "thinking" / has stepped.
      return _disabledBar('${table.toAct!.shortLabel}: ${agent.label} acting…');
    }

    final argmax = session.strategy?.argmaxLegalIdx;
    final qs     = session.strategy?.qValues;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF1B1B1B),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          for (int i = 0; i < obs.legal.length; i++)
            _ActionButton(
              label: _labelFor(obs.legal[i], obs.sizingForLegal(i), table),
              subLabel: qs != null
                  ? 'Q ${qs[i].toStringAsFixed(2)}'
                  : null,
              isArgmax: argmax == i,
              isFold: obs.legal[i] == ActionType.fold,
              isAllIn: obs.legal[i] == ActionType.allIn,
              onTap: () => session.applyLegal(i),
            ),
        ],
      ),
    );
  }

  Widget _disabledBar(String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 22),
      decoration: BoxDecoration(
        color: const Color(0xFF1B1B1B),
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
