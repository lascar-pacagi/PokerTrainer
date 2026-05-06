/// Visual centerpiece for the trainer: board cards on top, two seats facing
/// each other, pot in the middle. Lighter than the inspector's TableView
/// because we don't have a live engine — everything reads from the
/// TrainerSession's current node.
library;

import 'package:flutter/material.dart';

import '../../../ffi/actions.dart';
import '../../../widgets/card_view.dart';
import '../../scenario.dart';
import '../trainer_session.dart';

class TrainerTable extends StatelessWidget {
  final TrainerSession session;
  /// True when the hand is at a showdown terminal — only then do we reveal
  /// opp hole cards. Until then they render face-down.
  final bool revealOpp;

  const TrainerTable({
    super.key,
    required this.session,
    required this.revealOpp,
  });

  @override
  Widget build(BuildContext context) {
    final n = session.currentNode;
    final isUserTurn = n.isAction && n.player == session.userSeat;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1A2D27),
        border: Border.all(color: const Color(0xFF345A4A), width: 1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          _seatRow(
            label: session.oppSeat.toUpperCase(),
            sublabel: _seatSublabel(session.oppSeat),
            cards: revealOpp
                ? _cardsFromCombo(session.oppHandStrRevealed)
                : null,
            stack: _stackFor(n, session.oppSeat),
            highlight: !isUserTurn && n.isAction,
            isOpp: true,
          ),
          const SizedBox(height: 16),
          _board(n.board, n.pot),
          const SizedBox(height: 16),
          _seatRow(
            label: 'YOU (${session.userSeat.toUpperCase()})',
            sublabel: _seatSublabel(session.userSeat),
            cards: _cardsFromCombo(session.userHandStr),
            stack: _stackFor(n, session.userSeat),
            highlight: isUserTurn,
            isOpp: false,
          ),
        ],
      ),
    );
  }

  /// HU postflop convention: BB acts first → OOP. SB acts second → IP.
  /// We surface the SB/BB label as a sublabel so the user's mental model
  /// (which is usually positional) lines up with the data label.
  String _seatSublabel(String seat) => seat == 'oop' ? 'BB' : 'SB';

  int _stackFor(ScenarioNode node, String seat) =>
      seat == 'oop' ? node.stacks[0] : node.stacks[1];

  List<int> _cardsFromCombo(String combo) {
    // combo is "AsKh" — 4 chars, two cards.
    return [
      _parseCard(combo.substring(0, 2)),
      _parseCard(combo.substring(2, 4)),
    ];
  }

  /// EngineCard.parse can throw on garbage; fall back to a "no card" sentinel
  /// rather than crashing the trainer on a malformed combo string.
  int _parseCard(String s) {
    try {
      return EngineCard.parse(s);
    } catch (_) {
      return EngineCard.noCard;
    }
  }

  Widget _seatRow({
    required String label,
    required String sublabel,
    required List<int>? cards,
    required int stack,
    required bool highlight,
    required bool isOpp,
  }) {
    final accent = isOpp ? const Color(0xFFDC8A6F) : const Color(0xFF6FB3DC);
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        // Cards (left of nameplate when bottom seat, right when top).
        if (!isOpp) _cardsBlock(cards, faceDown: false),
        if (!isOpp) const SizedBox(width: 14),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          decoration: BoxDecoration(
            color: highlight
                ? accent.withValues(alpha: 0.22)
                : const Color(0xFF24343A),
            border: Border.all(
              color: highlight ? accent : const Color(0xFF45525A),
              width: highlight ? 2 : 1,
            ),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Text(
                    label,
                    style: TextStyle(
                      color: accent,
                      fontWeight: FontWeight.w800,
                      fontSize: 13,
                      letterSpacing: 1.2,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    sublabel,
                    style: const TextStyle(
                      color: Color(0x88EAE6D9),
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 2),
              Text(
                'stack $stack',
                style: const TextStyle(
                  color: Color(0xCCEAE6D9),
                  fontFamily: 'monospace',
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ),
        if (isOpp) const SizedBox(width: 14),
        if (isOpp)
          _cardsBlock(cards, faceDown: !revealOpp || cards == null),
      ],
    );
  }

  Widget _cardsBlock(List<int>? cards, {required bool faceDown}) {
    if (faceDown || cards == null) {
      return const Row(
        children: [
          CardView(card: 0, faceDown: true, width: 50),
          SizedBox(width: 6),
          CardView(card: 0, faceDown: true, width: 50),
        ],
      );
    }
    return Row(
      children: [
        CardView(card: cards[0], width: 50),
        const SizedBox(width: 6),
        CardView(card: cards[1], width: 50),
      ],
    );
  }

  Widget _board(List<String> board, int pot) {
    return Column(
      children: [
        Text(
          'POT $pot',
          style: const TextStyle(
            color: Color(0xFFEAE6D9),
            fontFamily: 'monospace',
            fontSize: 16,
            fontWeight: FontWeight.w800,
            letterSpacing: 1.2,
          ),
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 6,
          children: [
            for (final c in board)
              CardView(card: EngineCard.parse(c), width: 50),
          ],
        ),
      ],
    );
  }
}

