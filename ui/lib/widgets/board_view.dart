/// Center-of-table: pot label + 5-slot board, with unrevealed slots shown
/// as empty placeholders (slot count tied to current street).
library;

import 'package:flutter/material.dart';

import '../ffi/actions.dart';
import '../ffi/engine.dart';
import 'card_view.dart';

class BoardView extends StatelessWidget {
  final TableState table;
  final List<int> board;     // 5 slots, NO_CARD where unrevealed by board state

  const BoardView({super.key, required this.table, required this.board});

  @override
  Widget build(BuildContext context) {
    final visible = table.street.nVisibleBoard;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 18),
      decoration: BoxDecoration(
        color: const Color(0xFF0F2A1B),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFF1F4F36), width: 1.5),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text(
            'POT',
            style: TextStyle(
              color: Color(0xAAEAE6D9),
              letterSpacing: 2.0,
              fontSize: 12,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            '${table.potBb.toStringAsFixed(1)} bb',
            style: const TextStyle(
              color: Color(0xFFEAE6D9),
              fontFamily: 'monospace',
              fontSize: 30,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              for (int i = 0; i < 5; i++) ...[
                if (i > 0) const SizedBox(width: 8),
                CardView(
                  card: i < visible ? board[i] : EngineCard.noCard,
                  width: 64,
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }
}
