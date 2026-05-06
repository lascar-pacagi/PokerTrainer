/// End-of-hand recap. One row per user decision, highlighting EV gap so the
/// user can scan for the costly mistakes at a glance.
library;

import 'package:flutter/material.dart';

import '../trainer_session.dart';

class TrainerDecisionSummary extends StatelessWidget {
  final TrainerSession session;
  final VoidCallback onReplay;
  final VoidCallback onNewHand;
  final VoidCallback onQuit;

  const TrainerDecisionSummary({
    super.key,
    required this.session,
    required this.onReplay,
    required this.onNewHand,
    required this.onQuit,
  });

  @override
  Widget build(BuildContext context) {
    final log = session.log;
    final totalGap = log.fold<double>(0, (s, r) => s + r.evGap);

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF24282B),
        border: Border.all(color: const Color(0xFFD4B43F), width: 1.4),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              const Icon(Icons.flag, size: 20, color: Color(0xFFD4B43F)),
              const SizedBox(width: 8),
              const Text(
                'HAND RECAP',
                style: TextStyle(
                  color: Color(0xFFEAE6D9),
                  fontSize: 13,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 1.4,
                ),
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: _gapColor(totalGap).withValues(alpha: 0.22),
                  border: Border.all(color: _gapColor(totalGap), width: 1),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  'total EV gap ${totalGap.toStringAsFixed(2)} chips',
                  style: TextStyle(
                    color: _gapColor(totalGap),
                    fontFamily: 'monospace',
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            'You played ${log.length} decision${log.length == 1 ? "" : "s"} '
            'as ${session.userSeat.toUpperCase()} '
            '(${session.userSeat == 'oop' ? 'BB' : 'SB'}). '
            'Hand: ${session.userHandStr} · opp: ${session.oppHandStrRevealed}.',
            style: const TextStyle(
              color: Color(0xCCEAE6D9),
              fontSize: 12,
              height: 1.4,
            ),
          ),
          const SizedBox(height: 10),
          if (log.isEmpty)
            const Text(
              'Hand ended before any user decision (opp action only).',
              style: TextStyle(
                color: Color(0x99EAE6D9),
                fontStyle: FontStyle.italic,
                fontSize: 12,
              ),
            )
          else
            _table(log),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              FilledButton.icon(
                onPressed: onReplay,
                icon: const Icon(Icons.replay, size: 16),
                label: const Text('Replay this hand'),
              ),
              OutlinedButton.icon(
                onPressed: onNewHand,
                icon: const Icon(Icons.casino, size: 16),
                label: const Text('Deal new hand'),
              ),
              OutlinedButton.icon(
                onPressed: onQuit,
                icon: const Icon(Icons.close, size: 16),
                label: const Text('Quit trainer'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _table(List<DecisionRecord> log) {
    return Column(
      children: [
        _headerRow(),
        for (var i = 0; i < log.length; i++) _row(i, log[i]),
      ],
    );
  }

  Widget _headerRow() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 5),
      decoration: const BoxDecoration(
        color: Color(0xFF1B1E20),
        borderRadius: BorderRadius.vertical(top: Radius.circular(4)),
      ),
      child: const Row(
        children: [
          SizedBox(
            width: 26,
            child: Text('#',
                style: TextStyle(
                  color: Color(0x99EAE6D9),
                  fontFamily: 'monospace',
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                )),
          ),
          Expanded(
            child: Text('LINE',
                style: TextStyle(
                  color: Color(0x99EAE6D9),
                  fontFamily: 'monospace',
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                )),
          ),
          SizedBox(
            width: 110,
            child: Text('YOU',
                style: TextStyle(
                  color: Color(0x99EAE6D9),
                  fontFamily: 'monospace',
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                )),
          ),
          SizedBox(
            width: 110,
            child: Text('GTO BEST',
                style: TextStyle(
                  color: Color(0x99EAE6D9),
                  fontFamily: 'monospace',
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                )),
          ),
          SizedBox(
            width: 80,
            child: Text('EV GAP',
                textAlign: TextAlign.right,
                style: TextStyle(
                  color: Color(0x99EAE6D9),
                  fontFamily: 'monospace',
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                )),
          ),
        ],
      ),
    );
  }

  Widget _row(int i, DecisionRecord r) {
    final col = _gapColor(r.evGap);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 5),
      decoration: BoxDecoration(
        color: i.isEven
            ? const Color(0xFF1F2225)
            : const Color(0xFF24282B),
        border: const Border(
          bottom: BorderSide(color: Color(0xFF3A3F42), width: 0.5),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 26,
            child: Text(
              '${i + 1}',
              style: const TextStyle(
                color: Color(0x99EAE6D9),
                fontFamily: 'monospace',
                fontSize: 12,
              ),
            ),
          ),
          Expanded(
            child: Text(
              r.line.isEmpty ? '(root)' : r.line.join(' › '),
              style: const TextStyle(
                color: Color(0xCCEAE6D9),
                fontFamily: 'monospace',
                fontSize: 12,
              ),
            ),
          ),
          SizedBox(
            width: 110,
            child: Text(
              '${r.yourAction}\n${r.yourEv.toStringAsFixed(2)}',
              style: TextStyle(
                color: r.yourActionIdx == r.bestActionIdx
                    ? const Color(0xFF6FDC84)
                    : const Color(0xFFEAE6D9),
                fontFamily: 'monospace',
                fontSize: 12,
                fontWeight: FontWeight.w700,
                height: 1.3,
              ),
            ),
          ),
          SizedBox(
            width: 110,
            child: Text(
              '${r.bestAction}\n${r.bestEv.toStringAsFixed(2)}',
              style: const TextStyle(
                color: Color(0xFFD4B43F),
                fontFamily: 'monospace',
                fontSize: 12,
                fontWeight: FontWeight.w700,
                height: 1.3,
              ),
            ),
          ),
          SizedBox(
            width: 80,
            child: Text(
              r.evGap.toStringAsFixed(2),
              textAlign: TextAlign.right,
              style: TextStyle(
                color: col,
                fontFamily: 'monospace',
                fontSize: 12,
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// Same thresholds as the in-decision feedback so the user has a single
  /// mental scale across screens. Chip-denominated; see trainer_action_panel.
  static Color _gapColor(double gap) => gap < 0.5
      ? const Color(0xFF6FDC84)
      : gap < 3.0
          ? const Color(0xFFD4B43F)
          : const Color(0xFFDC6F6F);
}
