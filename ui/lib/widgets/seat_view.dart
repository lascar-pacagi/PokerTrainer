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
  /// Equity for this seat (win + 0.5*tie). Null when unknown.
  final double? equityFraction;

  const SeatView({
    super.key,
    required this.player,
    required this.table,
    required this.holeCards,
    required this.isToAct,
    this.faceDown = false,
    this.agentLabel,
    this.equityFraction,
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
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF112216),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: ringColor, width: 3),
        boxShadow: isToAct
            ? const [
                BoxShadow(color: Color(0x66FFD24A), blurRadius: 18, spreadRadius: 1),
              ]
            : null,
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Hole cards (bigger).
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              CardView(card: holeCards[0], faceDown: faceDown, width: 64),
              const SizedBox(width: 6),
              CardView(card: holeCards[1], faceDown: faceDown, width: 64),
            ],
          ),
          const SizedBox(width: 16),
          // Name + stack + equity column.
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
                      fontSize: 18,
                    ),
                  ),
                  if (agentLabel != null) ...[
                    const SizedBox(width: 8),
                    Text(
                      '· $agentLabel',
                      style: const TextStyle(
                        color: Color(0xAAEAE6D9),
                        fontSize: 13,
                      ),
                    ),
                  ],
                  if (allIn) ...[
                    const SizedBox(width: 8),
                    _badge('ALL-IN', const Color(0xFFC72D2D)),
                  ],
                ],
              ),
              const SizedBox(height: 4),
              Text(
                '${stackBb.toStringAsFixed(1)} bb',
                style: const TextStyle(
                  color: Color(0xFFEAE6D9),
                  fontFamily: 'monospace',
                  fontSize: 22,
                  fontWeight: FontWeight.w700,
                ),
              ),
              if (committedBb > 0)
                Padding(
                  padding: const EdgeInsets.only(top: 3),
                  child: Text(
                    'in: ${committedBb.toStringAsFixed(2)} bb',
                    style: const TextStyle(
                      color: Color(0xAAEAE6D9),
                      fontFamily: 'monospace',
                      fontSize: 13,
                    ),
                  ),
                ),
              if (equityFraction != null && !faceDown) ...[
                const SizedBox(height: 6),
                _equityChip(equityFraction!),
              ],
            ],
          ),
        ],
      ),
    );
  }

  Widget _equityChip(double eq) {
    final pct = (eq * 100).toStringAsFixed(1);
    // Color tilts green when ahead, red when behind.
    final tint = eq >= 0.5
        ? Color.lerp(const Color(0xFF1F4F36), const Color(0xFF2D7A4D),
            (eq - 0.5).clamp(0.0, 0.5) * 2)!
        : Color.lerp(const Color(0xFF4F1F1F), const Color(0xFF7A2D2D),
            (0.5 - eq).clamp(0.0, 0.5) * 2)!;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: tint,
        borderRadius: BorderRadius.circular(5),
      ),
      child: Text(
        'equity $pct%',
        style: const TextStyle(
          color: Colors.white,
          fontFamily: 'monospace',
          fontSize: 12,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  Widget _badge(String text, Color bg) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        text,
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.w700,
          fontSize: 11,
          letterSpacing: 0.6,
        ),
      ),
    );
  }
}
