/// The HU NLHE table: BB at the top, board in the middle, SB at the bottom.
/// Reads everything from GameSession.
library;

import 'package:flutter/material.dart';

import '../ffi/actions.dart';
import '../game/game_session.dart';
import 'board_view.dart';
import 'seat_view.dart';

class TableView extends StatelessWidget {
  final GameSession session;
  /// True = always show both hole hands face-up (inspector mode).
  /// False = hide the seat that's NOT to act (play-vs-bot feel).
  final bool revealAllHoles;

  /// Optional override for a seat's occupant label (e.g. the Play screen passes
  /// "You" / "Slumbot" / a model name). Null → fall back to the agent label.
  final String? Function(Player p)? labelFor;

  /// Optional override for whether a seat is rendered face-down (e.g. the Play
  /// screen hides Slumbot's unknown cards). Null → the default reveal logic.
  final bool Function(Player p)? faceDownOf;

  /// Optional transient "last action" caption per seat ("Raise to 6 bb"…).
  final String? Function(Player p)? bubbleFor;

  const TableView({
    super.key,
    required this.session,
    this.revealAllHoles = true,
    this.labelFor,
    this.faceDownOf,
    this.bubbleFor,
  });

  @override
  Widget build(BuildContext context) {
    final table   = session.tableState;
    final board   = session.env.getBoard();
    final sbHole  = session.env.getHole(Player.sb);
    final bbHole  = session.env.getHole(Player.bb);
    final equity  = session.equity;
    final summary = session.terminalSummary();

    Widget seat(Player p, List<int> hole) {
      // Face-down: a caller override wins (Play hides Slumbot's unknown cards);
      // otherwise, when not revealing all, hide the seat that is NOT to act.
      final hide = faceDownOf != null
          ? faceDownOf!(p)
          : (!revealAllHoles &&
              !table.isTerminal &&
              table.toAct != null &&
              p != table.toAct);
      double? eq;
      if (equity != null && !hide) {
        eq = p == Player.sb ? equity.sbEquity() : equity.bbEquity();
      }
      return SeatView(
        player: p,
        table: table,
        holeCards: hole,
        isToAct: !table.isTerminal && table.toAct == p,
        faceDown: hide,
        agentLabel: labelFor != null ? labelFor!(p) : _agentLabel(session.agentFor(p)),
        equityFraction: eq,
        actionBubble: bubbleFor != null ? bubbleFor!(p) : null,
      );
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 24),
      decoration: BoxDecoration(
        color: const Color(0xFF0A1A11),
        borderRadius: BorderRadius.circular(28),
        gradient: const RadialGradient(
          radius: 0.95,
          colors: [Color(0xFF143F26), Color(0xFF071913)],
          stops: [0.05, 1.0],
        ),
        border: Border.all(color: const Color(0xFF233F2C), width: 4),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // BB seat (top)
          Align(
            alignment: Alignment.topCenter,
            child: seat(Player.bb, bbHole),
          ),
          const SizedBox(height: 24),
          BoardView(table: table, board: board),
          const SizedBox(height: 24),
          // SB seat (bottom)
          Align(
            alignment: Alignment.bottomCenter,
            child: seat(Player.sb, sbHole),
          ),
          if (summary != null) ...[
            const SizedBox(height: 18),
            _TerminalCard(summary: summary),
          ],
        ],
      ),
    );
  }

  String? _agentLabel(SeatAgent a) {
    if (a is HumanAgent) return 'Human';
    if (a is ModelAgent) return 'Model · ${a.label}';
    return null;
  }
}

class _TerminalCard extends StatelessWidget {
  final TerminalSummary summary;
  const _TerminalCard({required this.summary});

  @override
  Widget build(BuildContext context) {
    final winner = summary.winner;
    final isChop = winner == null;
    final winnerColor = winner == Player.sb
        ? const Color(0xFF2D5B7C)
        : winner == Player.bb
            ? const Color(0xFF7C5B2D)
            : const Color(0xFF555555);

    final headline = isChop
        ? 'Hand chopped'
        : '${winner.shortLabel} wins ${summary.winAmountBb.toStringAsFixed(2)} bb';

    final reasonStr = summary.reason == Terminal.fold
        ? 'opponent folded'
        : 'showdown';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
      decoration: BoxDecoration(
        color: const Color(0xEE0A0A0A),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: winnerColor, width: 2),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                isChop
                    ? Icons.handshake
                    : (summary.reason == Terminal.fold
                        ? Icons.exit_to_app
                        : Icons.emoji_events),
                color: const Color(0xFFFFD24A),
                size: 22,
              ),
              const SizedBox(width: 8),
              Text(
                headline,
                style: const TextStyle(
                  color: Color(0xFFEAE6D9),
                  fontSize: 22,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 0.4,
                ),
              ),
              const SizedBox(width: 12),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: const Color(0x33FFFFFF),
                  borderRadius: BorderRadius.circular(5),
                ),
                child: Text(
                  reasonStr,
                  style: const TextStyle(
                    color: Color(0xEEEAE6D9),
                    fontSize: 13,
                    letterSpacing: 0.6,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          // Per-player chip delta + showdown hand category if available.
          Row(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _seatLine('SB',
                  summary.sbDeltaBb,
                  category: summary.sbHandCategory,
                  isWinner: winner == Player.sb),
              const SizedBox(width: 28),
              _seatLine('BB',
                  summary.bbDeltaBb,
                  category: summary.bbHandCategory,
                  isWinner: winner == Player.bb),
            ],
          ),
        ],
      ),
    );
  }

  Widget _seatLine(String label, double deltaBb,
      {String? category, required bool isWinner}) {
    final color = deltaBb > 0
        ? const Color(0xFF66C28F)
        : deltaBb < 0
            ? const Color(0xFFD0716F)
            : const Color(0xCCEAE6D9);
    final sign = deltaBb >= 0 ? '+' : '';
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              label,
              style: TextStyle(
                color: const Color(0xFFEAE6D9),
                fontSize: 15,
                letterSpacing: 0.8,
                fontWeight: isWinner ? FontWeight.w800 : FontWeight.w500,
              ),
            ),
            const SizedBox(width: 8),
            Text(
              '$sign${deltaBb.toStringAsFixed(2)} bb',
              style: TextStyle(
                color: color,
                fontFamily: 'monospace',
                fontSize: 17,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
        if (category != null)
          Padding(
            padding: const EdgeInsets.only(top: 2),
            child: Text(
              category,
              style: const TextStyle(
                color: Color(0xCCEAE6D9),
                fontSize: 14,
                fontStyle: FontStyle.italic,
              ),
            ),
          ),
      ],
    );
  }
}
