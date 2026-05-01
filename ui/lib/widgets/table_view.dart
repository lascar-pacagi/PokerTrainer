/// The HU NLHE table: BB at the top, board in the middle, SB at the bottom.
/// Reads everything from GameSession.
library;

import 'package:flutter/material.dart';

import '../ffi/actions.dart';
import '../ffi/engine.dart';
import '../game/game_session.dart';
import 'board_view.dart';
import 'seat_view.dart';

class TableView extends StatelessWidget {
  final GameSession session;
  /// True = always show both hole hands face-up (inspector mode).
  /// False = hide the seat that's NOT to act (play-vs-bot feel).
  final bool revealAllHoles;

  const TableView({
    super.key,
    required this.session,
    this.revealAllHoles = true,
  });

  @override
  Widget build(BuildContext context) {
    final table = session.tableState;
    final board = session.env.getBoard();
    final sbHole = session.env.getHole(Player.sb);
    final bbHole = session.env.getHole(Player.bb);

    Widget seat(Player p, List<int> hole) {
      // Hide-cards rule (when not in reveal-all): hide the seat that is
      // currently NOT to act. On terminal we always show both.
      final hide = !revealAllHoles &&
          !table.isTerminal &&
          table.toAct != null &&
          p != table.toAct;
      return SeatView(
        player: p,
        table: table,
        holeCards: hole,
        isToAct: !table.isTerminal && table.toAct == p,
        faceDown: hide,
        agentLabel: _agentLabel(session.agentFor(p)),
      );
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 18),
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
          const SizedBox(height: 20),
          BoardView(table: table, board: board),
          const SizedBox(height: 20),
          // SB seat (bottom)
          Align(
            alignment: Alignment.bottomCenter,
            child: seat(Player.sb, sbHole),
          ),
          if (table.isTerminal) ...[
            const SizedBox(height: 14),
            _terminalBanner(table),
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

  Widget _terminalBanner(TableState t) {
    final reasonStr = t.terminalReason == Terminal.fold ? 'fold' : 'showdown';
    final sbBb = t.payoffChips[0] / TableState.chipsPerBb;
    final bbBb = t.payoffChips[1] / TableState.chipsPerBb;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xCC000000),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        'Hand ended ($reasonStr) · SB ${sbBb >= 0 ? "+" : ""}${sbBb.toStringAsFixed(2)} bb · '
        'BB ${bbBb >= 0 ? "+" : ""}${bbBb.toStringAsFixed(2)} bb',
        style: const TextStyle(
          color: Color(0xFFEAE6D9),
          fontFamily: 'monospace',
          fontSize: 13,
        ),
      ),
    );
  }
}
