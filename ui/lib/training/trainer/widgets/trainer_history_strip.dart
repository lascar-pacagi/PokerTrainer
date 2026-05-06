/// Right-rail chronological list of everything that happened in the hand.
/// Mirrors the inspector's HistoryStrip in spirit but reads from the trainer
/// session's history list (which mixes user/opp actions, dealt cards, and
/// street markers).
library;

import 'package:flutter/material.dart';

import '../trainer_session.dart';

class TrainerHistoryStrip extends StatelessWidget {
  final TrainerSession session;
  const TrainerHistoryStrip({super.key, required this.session});

  @override
  Widget build(BuildContext context) {
    final entries = session.history;
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF272A2D),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFF4A4E52), width: 1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Padding(
            padding: EdgeInsets.fromLTRB(12, 12, 12, 6),
            child: Text(
              'HAND HISTORY',
              style: TextStyle(
                color: Color(0xCCEAE6D9),
                fontSize: 11,
                letterSpacing: 1.6,
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
          const Divider(height: 1, color: Color(0xFF4A4E52)),
          Expanded(
            child: entries.isEmpty
                ? const Padding(
                    padding: EdgeInsets.all(12),
                    child: Text(
                      'no actions yet',
                      style: TextStyle(
                        color: Color(0x66EAE6D9),
                        fontStyle: FontStyle.italic,
                        fontSize: 11,
                      ),
                    ),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    itemCount: entries.length,
                    itemBuilder: (_, i) => _entryRow(entries[i]),
                  ),
          ),
        ],
      ),
    );
  }

  Widget _entryRow(TrainerHistoryEntry e) {
    if (e.kind == 'street') {
      // Section divider between streets.
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        child: Row(
          children: [
            const Expanded(
              child: Divider(color: Color(0xFF4A4E52), height: 1),
            ),
            const SizedBox(width: 8),
            Text(
              e.label,
              style: const TextStyle(
                color: Color(0xCCD4B43F),
                fontSize: 10,
                letterSpacing: 1.4,
                fontWeight: FontWeight.w800,
              ),
            ),
            const SizedBox(width: 8),
            const Expanded(
              child: Divider(color: Color(0xFF4A4E52), height: 1),
            ),
          ],
        ),
      );
    }
    if (e.kind == 'chance') {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        child: Text(
          e.label,
          style: const TextStyle(
            color: Color(0xCCD4B43F),
            fontFamily: 'monospace',
            fontSize: 11,
            fontStyle: FontStyle.italic,
          ),
        ),
      );
    }
    final actor = e.kind == 'user' ? 'YOU' : 'OPP';
    final accent = e.kind == 'user'
        ? const Color(0xFF6FB3DC)
        : const Color(0xFFDC8A6F);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
      child: Row(
        children: [
          SizedBox(
            width: 38,
            child: Text(
              actor,
              style: TextStyle(
                color: accent,
                fontFamily: 'monospace',
                fontSize: 11,
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
          Expanded(
            child: Text(
              e.label,
              style: const TextStyle(
                color: Color(0xCCEAE6D9),
                fontFamily: 'monospace',
                fontSize: 12,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}
