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
  /// Transient "last action" caption shown as a pill above the seat
  /// ("Raise to 6 bb", "Check", "Call", "Fold"…), poker-client style.
  final String? actionBubble;
  /// Optional caption shown beneath face-down cards when the viewer asked to
  /// peek but this seat's cards aren't available to reveal yet (e.g. Slumbot,
  /// which only discloses its hand at showdown). Null → no caption.
  final String? revealHint;

  const SeatView({
    super.key,
    required this.player,
    required this.table,
    required this.holeCards,
    required this.isToAct,
    this.faceDown = false,
    this.agentLabel,
    this.equityFraction,
    this.actionBubble,
    this.revealHint,
  });

  @override
  Widget build(BuildContext context) {
    final stackBb     = table.stackBb(player);
    final committedBb = table.investedThisStreetBb(player);
    final allIn       = table.allIn[player.index];
    final ringColor   = isToAct
        ? const Color(0xFFFFD24A)
        : const Color(0x00000000);

    // Action bubble to the RIGHT of the seat (more room for "Raise to 6 bb",
    // no vertical shift). Fixed-width slot so the table doesn't jump.
    final bubble = SizedBox(
      width: 150,
      child: actionBubble == null
          ? null
          : Padding(
              padding: const EdgeInsets.only(left: 12),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: const Color(0xF2202830),
                    borderRadius: BorderRadius.circular(13),
                    border: Border.all(color: const Color(0xFFFFD24A), width: 1.5),
                  ),
                  child: Text(
                    actionBubble!,
                    style: const TextStyle(
                      color: Color(0xFFFFE8A8),
                      fontWeight: FontWeight.w800,
                      fontSize: 14,
                      letterSpacing: 0.3,
                    ),
                  ),
                ),
              ),
            ),
    );

    final box = AnimatedContainer(
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
          // Hole cards (bigger). A reveal hint sits beneath them when the eye
          // is on but these cards aren't available to peek (Slumbot pre-show).
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CardView(card: holeCards[0], faceDown: faceDown, width: 64),
                  const SizedBox(width: 6),
                  CardView(card: holeCards[1], faceDown: faceDown, width: 64),
                ],
              ),
              if (revealHint != null) ...[
                const SizedBox(height: 5),
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.lock_outline,
                        size: 12, color: Color(0x99EAE6D9)),
                    const SizedBox(width: 4),
                    Text(
                      revealHint!,
                      style: const TextStyle(
                        color: Color(0x99EAE6D9),
                        fontSize: 12,
                        fontStyle: FontStyle.italic,
                      ),
                    ),
                  ],
                ),
              ],
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
                        color: Color(0xCCEAE6D9),
                        fontSize: 15,
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
                      color: Color(0xCCEAE6D9),
                      fontFamily: 'monospace',
                      fontSize: 15,
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

    return Row(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [box, bubble],
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
          fontSize: 15,
          fontWeight: FontWeight.w700,
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
          fontWeight: FontWeight.w800,
          fontSize: 13,
          letterSpacing: 0.8,
        ),
      ),
    );
  }
}
