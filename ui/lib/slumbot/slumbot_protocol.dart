/// Pure translation between the Slumbot API's wire format and the engine.
///
/// No I/O here — every function is a pure transform so it can be unit-tested
/// without the network. The HTTP layer lives in `slumbot_client.dart`; the
/// orchestration (applying these to a mirror env) lives in the Play screen's
/// `MatchController`.
///
/// Slumbot format (slumbot.com/api): 200 bb stacks = 20000 chips, blinds
/// 50/100 — so 1 bb = 100 chips, IDENTICAL to the engine
/// (`TableState.chipsPerBb == 100`). Bet sizes therefore map 1:1: a Slumbot
/// `b300` is `betToChips = 300`.
///
/// Action-string grammar (one hand): tokens concatenated, `/` separates
/// streets. `f`=fold, `k`=check, `c`=call, `b<N>`=bet/raise so the actor's
/// total wagered this street is N chips. Cards are 2-char strings ("Ah") with
/// ranks `23456789TJQKA` and suits `cdhs` — the same alphabet the engine's
/// `EngineCard.parse` already uses.
library;

import '../ffi/actions.dart';

/// One parsed Slumbot betting action (street boundaries are carried
/// separately, see [parseActionString]).
sealed class SbAction {
  const SbAction();
}

class SbFold extends SbAction {
  const SbFold();
}

class SbCheck extends SbAction {
  const SbCheck();
}

class SbCall extends SbAction {
  const SbCall();
}

class SbBet extends SbAction {
  /// Total chips the actor has wagered this street after the bet/raise.
  final int toChips;
  const SbBet(this.toChips);
}

/// A betting action tagged with the street index it occurred on (0=preflop,
/// 1=flop, 2=turn, 3=river), so a replayer can detect street transitions.
class SbStreetAction {
  final int street;
  final SbAction action;
  const SbStreetAction(this.street, this.action);
}

/// Parse a Slumbot action string into an ordered, street-tagged token list.
///
/// `"b200c/kk/b300c"` → preflop bet-to-200, call; flop check, check; turn
/// bet-to-300, call. Empty string → empty list. Throws [FormatException] on a
/// malformed token.
List<SbStreetAction> parseActionString(String s) {
  final out = <SbStreetAction>[];
  if (s.isEmpty) return out;
  final streets = s.split('/');
  for (int street = 0; street < streets.length; street++) {
    final seg = streets[street];
    int i = 0;
    while (i < seg.length) {
      final ch = seg[i];
      switch (ch) {
        case 'f':
          out.add(SbStreetAction(street, const SbFold()));
          i++;
        case 'k':
          out.add(SbStreetAction(street, const SbCheck()));
          i++;
        case 'c':
          out.add(SbStreetAction(street, const SbCall()));
          i++;
        case 'b':
          int j = i + 1;
          while (j < seg.length && _isDigit(seg[j])) {
            j++;
          }
          if (j == i + 1) {
            throw FormatException('bet token missing size at $i in "$s"');
          }
          out.add(SbStreetAction(street, SbBet(int.parse(seg.substring(i + 1, j)))));
          i = j;
        default:
          throw FormatException('bad action char "$ch" at $i in "$s"');
      }
    }
  }
  return out;
}

bool _isDigit(String c) {
  final u = c.codeUnitAt(0);
  return u >= 0x30 && u <= 0x39;
}

/// Encode the local player's chosen action as a Slumbot `incr` token.
///
/// `type` is the engine ActionType the local seat played; `betToChips` is its
/// absolute chip target (from `Observation.sizingForLegal` / `sizingsByAction
/// Type`) — required for raises/all-in, ignored otherwise. `toCallChips` of the
/// spot decides whether CHECK_CALL renders as `k` (check) or `c` (call).
String encodeAction(ActionType type, {required int toCallChips, int betToChips = 0}) {
  switch (type) {
    case ActionType.fold:
      return 'f';
    case ActionType.checkCall:
      return toCallChips == 0 ? 'k' : 'c';
    default:
      // RAISE_* or ALL_IN — Slumbot wants the total wagered this street.
      return 'b$betToChips';
  }
}

/// Parse a list of Slumbot card strings ("Ah") into engine card bytes.
List<int> parseCards(List<String> cards) =>
    [for (final c in cards) EngineCard.parse(c)];

/// Build a 5-slot board (NO_CARD for unrevealed) from Slumbot's `board` list.
List<int> boardSlots(List<String> board) {
  final slots = List<int>.filled(5, EngineCard.noCard);
  for (int i = 0; i < board.length && i < 5; i++) {
    slots[i] = EngineCard.parse(board[i]);
  }
  return slots;
}
