/// One player's seat: name, hole cards, stack, action chips committed
/// this street. Highlighted with a colored ring when this seat is to act.
library;

import 'package:flutter/material.dart';

import '../ffi/actions.dart';
import '../ffi/engine.dart';
import 'card_view.dart';

class SeatView extends StatelessWidget {
  final Player player;
  final TableState table;
  final List<int> holeCards;          // 2 entries (NO_CARD allowed)
  final bool faceDown;                 // hide opponent's cards
  final bool isToAct;
  final String? agentLabel;            // e.g. "Human" / "Model: cpu_long_50k"

  const SeatView({
    super.key,
    required this.player,
    required this.table,
    required this.holeCards,
    required this.isToAct,
    this.faceDown = false,
    this.agentLabel,
  });

  @override
  Widget build(BuildContext context) {
    final stackBb     = table.stackBb(player);
    final committedBb = table.investedThisStreetBb(player);
    final allIn       = table.allIn[player.index];
    final ringColor   = isToAct
        ? const Color(0xFFFFD24A)
        : const Color(0x00000000);

    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: const Color(0xFF112216),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: ringColor, width: 2.5),
        boxShadow: isToAct
            ? const [
                BoxShadow(color: Color(0x55FFD24A), blurRadius: 14, spreadRadius: 1),
              ]
            : null,
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Hole cards
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              CardView(card: holeCards[0], faceDown: faceDown, width: 42),
              const SizedBox(width: 4),
              CardView(card: holeCards[1], faceDown: faceDown, width: 42),
            ],
          ),
          const SizedBox(width: 12),
          // Name + stack column
          Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    player.shortLabel,
                    style: const TextStyle(
                      color: Color(0xFFEAE6D9),
                      fontWeight: FontWeight.w800,
                      letterSpacing: 0.6,
                      fontSize: 14,
                    ),
                  ),
                  if (agentLabel != null) ...[
                    const SizedBox(width: 6),
                    Text(
                      '· $agentLabel',
                      style: const TextStyle(
                        color: Color(0xAAEAE6D9),
                        fontSize: 11,
                      ),
                    ),
                  ],
                  if (allIn) ...[
                    const SizedBox(width: 6),
                    _badge('ALL-IN', const Color(0xFFC72D2D)),
                  ],
                ],
              ),
              const SizedBox(height: 2),
              Text(
                '${stackBb.toStringAsFixed(1)} bb',
                style: const TextStyle(
                  color: Color(0xFFEAE6D9),
                  fontFamily: 'monospace',
                  fontSize: 16,
                ),
              ),
              if (committedBb > 0)
                Padding(
                  padding: const EdgeInsets.only(top: 2),
                  child: Text(
                    'in: ${committedBb.toStringAsFixed(2)} bb',
                    style: const TextStyle(
                      color: Color(0xAAEAE6D9),
                      fontFamily: 'monospace',
                      fontSize: 11,
                    ),
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _badge(String text, Color bg) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        text,
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.w700,
          fontSize: 9,
          letterSpacing: 0.6,
        ),
      ),
    );
  }
}
