/// Engine-side enums mirrored on the Dart side.
///
/// Ordinal values are *load-bearing* — they must match `pt::ActionType`,
/// `pt::Player`, `pt::Street`, `pt::Terminal` exactly so we can `byValue`
/// across the FFI boundary without translation tables.
library;

enum ActionType {
  fold,        // 0
  checkCall,   // 1
  raise25,     // 2 — 0.25× pot (often snaps to min-raise on small pots)
  raise33,     // 3
  raise50,     // 4
  raise75,     // 5
  raise100,    // 6
  raise150,    // 7
  raise200,    // 8
  raise300,    // 9
  allIn;       // 10

  static const numActions = 11;

  static ActionType fromInt(int i) {
    if (i < 0 || i >= numActions) {
      throw ArgumentError('ActionType ordinal out of range: $i');
    }
    return ActionType.values[i];
  }

  /// Short label for buttons/tables. Sizing context is added by the caller.
  String get shortLabel {
    switch (this) {
      case ActionType.fold:      return 'Fold';
      case ActionType.checkCall: return 'Call';
      case ActionType.raise25:   return 'R 25%';
      case ActionType.raise33:   return 'R 33%';
      case ActionType.raise50:   return 'R 50%';
      case ActionType.raise75:   return 'R 75%';
      case ActionType.raise100:  return 'R pot';
      case ActionType.raise150:  return 'R 1.5x';
      case ActionType.raise200:  return 'R 2x';
      case ActionType.raise300:  return 'R 3x';
      case ActionType.allIn:     return 'All-in';
    }
  }

  bool get isRaise => index >= ActionType.raise25.index && index <= ActionType.allIn.index;
  bool get isFold  => this == ActionType.fold;
}

enum Player {
  sb,   // 0 — BTN/SB: first to act preflop, last (IP) postflop
  bb;   // 1 — BB: last to act preflop, first (OOP) postflop

  static Player fromInt(int i) => Player.values[i];

  String get shortLabel => this == Player.sb ? 'SB' : 'BB';

  Player get other => this == Player.sb ? Player.bb : Player.sb;
}

enum Street {
  preflop,    // 0
  flop,       // 1
  turn,       // 2
  river,      // 3
  showdown;   // 4 — terminal-only

  static Street fromInt(int i) => Street.values[i];

  /// Number of board cards visible on this street.
  int get nVisibleBoard {
    switch (this) {
      case Street.preflop:  return 0;
      case Street.flop:     return 3;
      case Street.turn:     return 4;
      case Street.river:    return 5;
      case Street.showdown: return 5;
    }
  }
}

enum Terminal {
  none,      // 0 — hand still in progress
  fold,      // 1 — opponent folded
  showdown;  // 2 — board ran out, compare hands

  static Terminal fromInt(int i) => Terminal.values[i];
}

/// Card index convention (matches engine/src/card.h):
///   card = rank * 4 + suit, range [0, 52). 0xFF = NO_CARD.
class Card {
  static const int noCard   = 0xFF;
  static const int numRanks = 13;
  static const int numSuits = 4;
  static const int numCards = 52;

  static const String rankChars = '23456789TJQKA';
  static const String suitChars = 'cdhs';

  /// rank: 0=2 .. 12=A. suit: 0=c, 1=d, 2=h, 3=s.
  static int make(int rank, int suit) {
    assert(rank >= 0 && rank < numRanks);
    assert(suit >= 0 && suit < numSuits);
    return (rank << 2) | suit;
  }

  static int rankOf(int c) => c >> 2;
  static int suitOf(int c) => c & 0x3;

  /// "AsKh" → returns the AS card, then KH. Throws on parse error.
  static int parse(String s) {
    if (s.length != 2) throw FormatException('card must be 2 chars: "$s"');
    final r = rankChars.indexOf(s[0].toUpperCase());
    final t = suitChars.indexOf(s[1].toLowerCase());
    if (r < 0 || t < 0) throw FormatException('bad card: "$s"');
    return make(r, t);
  }

  static String toReadable(int c) {
    if (c == noCard) return '--';
    return '${rankChars[rankOf(c)]}${suitChars[suitOf(c)]}';
  }
}
