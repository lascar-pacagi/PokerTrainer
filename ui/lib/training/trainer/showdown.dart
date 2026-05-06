/// Showdown evaluation: ranks two 7-card hands via the engine's pt_eval_7
/// table and returns who won plus the chip distribution.
///
/// pt_eval_7 returns a numeric rank where LOWER is BETTER. Two hands tie
/// when their ranks compare equal.
library;

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../../ffi/actions.dart';
import '../../ffi/engine.dart';

enum ShowdownWinner { you, opp, tie }

class ShowdownResult {
  final ShowdownWinner winner;
  /// Engine rank for the user (lower = stronger). Negative if evaluation failed.
  final int yourRank;
  final int oppRank;
  /// Chips you net from this terminal vs. the start of the hand. Positive
  /// = profit. Computed assuming pot is split evenly on tie, full to
  /// winner otherwise; effective contribution from each side is
  /// `(starting_stack - current_stack)`.
  final int yourChipDelta;
  final int oppChipDelta;

  const ShowdownResult({
    required this.winner,
    required this.yourRank,
    required this.oppRank,
    required this.yourChipDelta,
    required this.oppChipDelta,
  });
}

/// Evaluate a 7-card showdown.
///
/// `userCards` and `oppCards` are 2-card lists (engine card bytes).
/// `board` is 5 cards (must be 5; throws on shorter — never call this on
/// a non-river terminal).
/// `pot` is the total pot at the terminal; `userInvested` and `oppInvested`
/// are how much each player has put into that pot. The chip-delta math
/// just follows: winner takes pot, payoff = pot - own_invested.
///
/// Returns null if [PokerEngine.instance] is null (engine never initialised
/// — defensive, shouldn't happen post-startup but allows the caller to
/// fall back to "showdown reached" without a winner reveal).
ShowdownResult? evaluateShowdown({
  required List<int> userCards,
  required List<int> oppCards,
  required List<String> board,
  required int pot,
  required int userInvested,
  required int oppInvested,
}) {
  final eng = PokerEngine.instance;
  if (eng == null) return null;
  if (userCards.length != 2 || oppCards.length != 2) return null;
  if (board.length != 5) {
    throw ArgumentError('evaluateShowdown requires a full 5-card board, '
        'got ${board.length}');
  }

  // FFI call needs two 7-byte buffers. Allocate, fill, eval, free.
  final youBuf = malloc<ffi.Uint8>(7);
  final oppBuf = malloc<ffi.Uint8>(7);
  try {
    youBuf[0] = userCards[0]; youBuf[1] = userCards[1];
    oppBuf[0] = oppCards[0];  oppBuf[1] = oppCards[1];
    for (int i = 0; i < 5; i++) {
      final c = EngineCard.parse(board[i]);
      youBuf[2 + i] = c;
      oppBuf[2 + i] = c;
    }
    final youRank = eng.native.ptEval7(youBuf);
    final oppRank = eng.native.ptEval7(oppBuf);

    final ShowdownWinner winner;
    if (youRank < oppRank) {
      winner = ShowdownWinner.you;
    } else if (oppRank < youRank) {
      winner = ShowdownWinner.opp;
    } else {
      winner = ShowdownWinner.tie;
    }

    final int yourDelta;
    final int oppDelta;
    switch (winner) {
      case ShowdownWinner.you:
        // Win the whole pot, having contributed `userInvested` to it.
        yourDelta = pot - userInvested;
        oppDelta = -oppInvested;
        break;
      case ShowdownWinner.opp:
        yourDelta = -userInvested;
        oppDelta = pot - oppInvested;
        break;
      case ShowdownWinner.tie:
        // Pot split — odd chip goes to first player by convention; we
        // distribute fairly using integer floor and dump remainder on user.
        // Edge case (ties) so the rounding policy doesn't really matter.
        final half = pot ~/ 2;
        final rem = pot - half * 2;
        yourDelta = (half + rem) - userInvested;
        oppDelta = half - oppInvested;
        break;
    }

    return ShowdownResult(
      winner: winner,
      yourRank: youRank,
      oppRank: oppRank,
      yourChipDelta: yourDelta,
      oppChipDelta: oppDelta,
    );
  } finally {
    malloc.free(youBuf);
    malloc.free(oppBuf);
  }
}
