/// One row per street with the actions taken on it. Read-only for now;
/// click-to-replay will be added when the GameSession learns to rewind.
library;

import 'package:flutter/material.dart';

import '../ffi/actions.dart';
import '../ffi/engine.dart';

class HistoryStrip extends StatelessWidget {
  final List<HistoryEntry> history;

  const HistoryStrip({super.key, required this.history});

  static const _streetOrder = [Street.preflop, Street.flop, Street.turn, Street.river];

  @override
  Widget build(BuildContext context) {
    final byStreet = <Street, List<HistoryEntry>>{
      for (final s in _streetOrder) s: const [],
    };
    for (final e in history) {
      byStreet[e.street] = [...byStreet[e.street]!, e];
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF101010),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          for (final s in _streetOrder)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 3),
              child: _streetRow(s, byStreet[s]!),
            ),
        ],
      ),
    );
  }

  Widget _streetRow(Street s, List<HistoryEntry> entries) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 80,
          child: Text(
            _streetLabel(s),
            style: const TextStyle(
              color: Color(0xAAEAE6D9),
              fontWeight: FontWeight.w700,
              fontSize: 13,
              letterSpacing: 1.0,
            ),
          ),
        ),
        Expanded(
          child: entries.isEmpty
              ? const Text(
                  '—',
                  style: TextStyle(
                    color: Color(0x66EAE6D9),
                    fontSize: 14,
                  ),
                )
              : Wrap(
                  spacing: 6,
                  runSpacing: 4,
                  children: [
                    for (final e in entries) _actionChip(e),
                  ],
                ),
        ),
      ],
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

  Widget _actionChip(HistoryEntry e) {
    final txt = _renderAction(e);
    final color = e.actor == Player.sb
        ? const Color(0xFF2D5B7C)
        : const Color(0xFF7C5B2D);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        txt,
        style: const TextStyle(
          color: Colors.white,
          fontFamily: 'monospace',
          fontSize: 13,
        ),
      ),
    );
  }

  String _renderAction(HistoryEntry e) {
    final actor = e.actor.shortLabel;
    if (e.type == ActionType.fold)      return '$actor fold';
    if (e.type == ActionType.checkCall) {
      return e.betToChips == 0 ? '$actor check' : '$actor call ${e.betToBb.toStringAsFixed(1)}';
    }
    if (e.type == ActionType.allIn) {
      return '$actor all-in ${e.betToBb.toStringAsFixed(1)}';
    }
    return '$actor ${e.type.shortLabel.replaceAll(" ", "")} ${e.betToBb.toStringAsFixed(1)}';
  }
}
