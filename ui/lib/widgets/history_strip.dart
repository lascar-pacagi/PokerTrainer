/// Vertical history rail. Section header per street; one row per applied
/// action. Scrolls when the hand gets long. Lives in the left column of
/// the inspector layout.
library;

import 'package:flutter/material.dart';

import '../ffi/actions.dart';
import '../ffi/engine.dart';

class HistoryStrip extends StatelessWidget {
  final List<HistoryEntry> history;

  const HistoryStrip({super.key, required this.history});

  static const _streetOrder = [
    Street.preflop,
    Street.flop,
    Street.turn,
    Street.river,
  ];

  @override
  Widget build(BuildContext context) {
    final byStreet = <Street, List<HistoryEntry>>{
      for (final s in _streetOrder) s: const [],
    };
    for (final e in history) {
      byStreet[e.street] = [...byStreet[e.street]!, e];
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
      decoration: BoxDecoration(
        color: const Color(0xFF101010),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF2A2A2A), width: 1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text(
            'HISTORY',
            style: TextStyle(
              color: Color(0xCCEAE6D9),
              fontSize: 13,
              letterSpacing: 1.6,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 12),
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                mainAxisSize: MainAxisSize.min,
                children: [
                  for (final s in _streetOrder)
                    _StreetSection(street: s, entries: byStreet[s]!),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _StreetSection extends StatelessWidget {
  final Street street;
  final List<HistoryEntry> entries;
  const _StreetSection({required this.street, required this.entries});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Street header — small caps over a thin underline.
          Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Row(
              children: [
                Text(
                  _streetLabel(street),
                  style: const TextStyle(
                    color: Color(0xAAEAE6D9),
                    fontWeight: FontWeight.w800,
                    fontSize: 12,
                    letterSpacing: 1.4,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Container(
                    height: 1,
                    color: const Color(0x33EAE6D9),
                  ),
                ),
              ],
            ),
          ),
          if (entries.isEmpty)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 2),
              child: Text(
                '—',
                style: TextStyle(
                  color: Color(0x55EAE6D9),
                  fontSize: 14,
                ),
              ),
            )
          else
            for (int i = 0; i < entries.length; i++)
              _actionRow(i, entries[i]),
        ],
      ),
    );
  }

  Widget _actionRow(int idx, HistoryEntry e) {
    final actor = e.actor;
    final actorColor = actor == Player.sb
        ? const Color(0xFF6FB3DC)
        : const Color(0xFFDCBE6F);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          // Step number.
          SizedBox(
            width: 22,
            child: Text(
              '${idx + 1}.',
              style: const TextStyle(
                color: Color(0x66EAE6D9),
                fontFamily: 'monospace',
                fontSize: 12,
              ),
            ),
          ),
          // Actor pill.
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
            decoration: BoxDecoration(
              color: actorColor.withValues(alpha: 0.18),
              borderRadius: BorderRadius.circular(4),
              border: Border.all(color: actorColor, width: 1),
            ),
            child: Text(
              actor.shortLabel,
              style: TextStyle(
                color: actorColor,
                fontFamily: 'monospace',
                fontWeight: FontWeight.w700,
                fontSize: 12,
              ),
            ),
          ),
          const SizedBox(width: 8),
          // Action description.
          Expanded(
            child: Text(
              _renderAction(e),
              style: const TextStyle(
                color: Color(0xFFEAE6D9),
                fontFamily: 'monospace',
                fontSize: 14,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  String _streetLabel(Street s) {
    switch (s) {
      case Street.preflop:  return 'PREFLOP';
      case Street.flop:     return 'FLOP';
      case Street.turn:     return 'TURN';
      case Street.river:    return 'RIVER';
      case Street.showdown: return 'SHOWDOWN';
    }
  }

  String _renderAction(HistoryEntry e) {
    if (e.type == ActionType.fold)      return 'fold';
    if (e.type == ActionType.checkCall) {
      return e.betToChips == 0
          ? 'check'
          : 'call ${e.betToBb.toStringAsFixed(2)} bb';
    }
    if (e.type == ActionType.allIn) {
      return 'all-in to ${e.betToBb.toStringAsFixed(2)} bb';
    }
    return '${e.type.shortLabel.toLowerCase()} → ${e.betToBb.toStringAsFixed(2)} bb';
  }
}
