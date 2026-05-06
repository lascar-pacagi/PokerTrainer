// One-shot tool: convert weighted 13x13 chart matrices into PokerStove range
// strings using the project's existing serializeClassWeights() helper.
//
// ignore_for_file: avoid_print, avoid_relative_lib_imports  // CLI tool, not app code
//
// The seven matrices below mirror the PokerCoaching 100bb HU NLHE preflop
// charts (BTN RFI, BB 3-Bet, BB Call, BTN 4-Bet, BTN Call vs 3-Bet, BB
// 5-Bet, BB Call vs 4-Bet). Each matrix is row-major with rows/cols ordered
// A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2.
//
// Run from the ui directory:
//     dart tool/build_pokercoaching_ranges.dart
//
// Output is the range string for each chart, ready to paste into
// preflop_presets.dart.
//
// NOTE: This file is a build aid, not part of the runtime app. It depends on
// `lib/training/range_parser.dart` for the serializer.

import '../lib/training/range_parser.dart';

const ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];

/// Produce the chart-cell label at (row, col). Diagonal = pair, above
/// diagonal = suited, below diagonal = offsuit (row=low rank, col=high rank).
String cellLabel(int row, int col) {
  if (row == col) return '${ranks[row]}${ranks[row]}';
  if (row < col) return '${ranks[row]}${ranks[col]}s';
  return '${ranks[col]}${ranks[row]}o';
}

/// Convert a 13x13 weighted matrix into a PokerStove range string.
String matrixToRange(List<List<double>> m) {
  final weights = <String, double>{};
  for (int r = 0; r < 13; r++) {
    for (int c = 0; c < 13; c++) {
      final w = m[r][c];
      if (w > 1e-4) weights[cellLabel(r, c)] = w;
    }
  }
  return serializeClassWeights(weights);
}

// ─── Matrices (cell weights from the PokerCoaching PDF) ─────────────────────

// 1. BTN RFI: 81% — pure 1/0, only the bottom-left "trash" zone is folded.
final btnRfi = <List<double>>[
  // A    K    Q    J    T    9    8    7    6    5    4    3    2
  [1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1], // A
  [1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1], // K
  [1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1], // Q
  [1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1], // J
  [1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1], // T
  [1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1], // 9
  [1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1], // 8
  [1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1], // 7
  [1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1], // 6
  [1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1], // 5
  [1,    1,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1], // 4
  [1,    1,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1], // 3
  [1,    1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1], // 2
];

// 2. BB 3-Bet: 20.4% — heavy on premium pairs/Broadways and select bluffs.
final bb3bet = <List<double>>[
  // A    K     Q     J     T     9     8     7     6     5     4     3     2
  [1,    1,    1,    1,    1,    0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5 ], // A
  [1,    1,    1,    1,    1,    0.75, 0.5,  0.5,  0.35, 0.35, 0.2,  0.2,  0.2 ], // K
  [1,    0.5,  1,    1,    1,    0.75, 0.5,  0.5,  0.35, 0.35, 0.2,  0,    0   ], // Q
  [0.5,  0.2,  0.2,  1,    1,    0.75, 0.5,  0.5,  0.35, 0.2,  0,    0,    0   ], // J
  [0.2,  0.1,  0.1,  0.1,  1,    1,    0.5,  0.2,  0.2,  0,    0,    0,    0   ], // T
  [0.1,  0.1,  0.1,  0.1,  0.1,  1,    1,    0.75, 0.5,  0.2,  0,    0,    0   ], // 9
  [0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  1,    1,    0.75, 0.5,  0.2,  0,    0   ], // 8
  [0,    0,    0,    0,    0,    0,    0.1,  0.5,  1,    0.75, 0.5,  0.2,  0   ], // 7
  [0,    0,    0,    0,    0,    0,    0,    0,    0.5,  1,    0.75, 0.5,  0   ], // 6
  [0,    0,    0,    0,    0,    0,    0,    0,    0.1,  0.5,  1,    0.75, 0   ], // 5
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0.1,  0.5,  0.5,  0.2 ], // 4
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.2 ], // 3
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.2 ], // 2
];

// 3. BB Call: 55.22% — broad continuing range vs SB open.
// Note: top-left premium cells are 0 here (3-bet not call).
final bbCall = <List<double>>[
  // A    K     Q     J     T     9     8     7     6     5     4     3     2
  [0,    0,    0,    0,    0,    0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5 ], // A
  [0,    0,    0,    0,    0,    0.25, 0.5,  0.5,  0.65, 0.65, 0.8,  0.8,  0.8 ], // K
  [0,    0.5,  0,    0,    0,    0.25, 0.5,  0.5,  0.65, 0.65, 0.8,  0,    0   ], // Q
  [0.5,  0.8,  0.8,  0,    0,    0.25, 0.5,  0.65, 0.65, 0.8,  1,    1,    1   ], // J
  [0.8,  0.9,  0.9,  0.9,  0,    0,    0.5,  0.8,  0.8,  1,    1,    1,    1   ], // T
  [0.9,  0.9,  0.9,  0.9,  0.9,  0,    0,    0.25, 0.5,  0.8,  1,    1,    1   ], // 9
  [0.9,  0.9,  0.9,  0.9,  0.9,  0.9,  0,    0.25, 0.5,  0.8,  1,    1,    1   ], // 8
  [1,    1,    1,    1,    1,    1,    0.9,  0.5,  0.25, 0.5,  0.8,  1,    1   ], // 7
  [1,    1,    1,    1,    1,    1,    1,    0.9,  0.5,  0.25, 0.5,  0.8,  1   ], // 6
  [1,    1,    0.75, 0.5,  0.35, 0.35, 0.35, 1,    0.9,  0.5,  0.25, 1,    1   ], // 5
  [1,    0.9,  0.5,  0,    0,    0,    0,    0.5,  0.9,  0.5,  0.5,  0.8,  0.8 ], // 4
  [1,    0.9,  0.5,  0,    0,    0,    0,    0,    0,    0.25, 0.25, 0,    0.8 ], // 3
  [1,    0.9,  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.8 ], // 2
];

// 4. BTN 4-Bet: 5.1% — narrow value-heavy 4-bet range with a few suited bluffs.
final btn4bet = <List<double>>[
  // A    K     Q     J     T     9     8     7     6     5     4     3     2
  [0,    0,    0.5,  0.1,  0.1,  0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], // A
  [1,    1,    0.02, 0.02, 0.02, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.04, 0.04], // K
  [0.15, 0.1,  1,    0.02, 0.02, 0.07, 0.07, 0.07, 0.07, 0.04, 0.04, 0.04, 0.04], // Q
  [0.12, 0.05, 0.05, 1,    0.02, 0.07, 0.07, 0.07, 0.07, 0.04, 0.04, 0.04, 0.04], // J
  [0.12, 0.04, 0.05, 0.05, 1,    0.07, 0.07, 0.07, 0.07, 0.04, 0.04, 0.04, 0.04], // T
  [0.1,  0.04, 0.04, 0.04, 0.04, 0.2,  0.04, 0.07, 0.07, 0.07, 0.04, 0.04, 0.04], // 9
  [0.1,  0.04, 0.04, 0.04, 0.04, 0.04, 0.1,  0.1,  0.07, 0.07, 0.07, 0.04, 0.04], // 8
  [0.1,  0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0,    0.1,  0.07, 0.07, 0.07, 0.04], // 7
  [0.1,  0.04, 0.04, 0,    0,    0,    0,    0.04, 0,    0.1,  0.07, 0.07, 0   ], // 6
  [0.2,  0,    0,    0,    0,    0,    0,    0,    0,    0,    0.1,  0.07, 0.04], // 5
  [0.2,  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.1,  0   ], // 4
  [0.15, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.04], // 3
  [0.15, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0   ], // 2
];

// 5. BTN Call vs 3-Bet: 37.5% — wide flatting range, mostly mid-low cards.
final btnCall = <List<double>>[
  // A    K     Q     J     T     9     8     7     6     5     4     3     2
  [0,    0.5,  0.96, 0.96, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98], // A
  [0,    0,    0.98, 0.98, 0.98, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96], // K
  [0.85, 0.9,  0,    0.98, 0.98, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.5,  0.5 ], // Q
  [0.88, 0.95, 0.95, 0,    0.98, 0.96, 0.96, 0.96, 0.96, 0.96, 0.5,  0.5,  0   ], // J
  [0.88, 0.95, 0.95, 0.95, 0,    0.96, 0.96, 0.96, 0.96, 0.96, 0.5,  0,    0   ], // T
  [0.75, 0.6,  0.6,  0.6,  0.6,  0.8,  0.96, 0.96, 0.96, 0.96, 0.5,  0,    0   ], // 9
  [0.75, 0.1,  0.1,  0.1,  0.1,  0.1,  0.9,  0.93, 0.96, 0.96, 0.96, 0,    0   ], // 8
  [0.2,  0,    0,    0,    0.1,  0,    0,    1,    0.93, 0.96, 0.96, 0.5,  0   ], // 7
  [0.2,  0,    0,    0,    0,    0,    0,    0,    1,    0.96, 0.96, 0.96, 0.5 ], // 6
  [0.6,  0,    0,    0,    0,    0,    0,    0,    0.1,  1,    0.93, 0.96, 0.5 ], // 5
  [0.6,  0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0.93, 0.5 ], // 4
  [0.2,  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0.5 ], // 3
  [0.2,  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1   ], // 2
];

// 6. BB 5-Bet: 3.7% — extreme polar all-in range, just AA/KK/AK + some bluffs.
final bb5bet = <List<double>>[
  // A    K     Q     J     T     9     8     7     6     5     4     3     2
  [0.65, 1,    0,    0,    0,    0,    0.2,  0.2,  0,    0,    0,    0.2,  0.2 ], // A
  [1,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0   ], // K
  [0,    0.05, 1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0   ], // Q
  [0.1,  0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0   ], // J
  [0.05, 0,    0,    0,    0.5,  0,    0,    0,    0,    0,    0,    0,    0   ], // T
  [0,    0,    0,    0,    0,    0,    0.2,  0,    0,    0,    0,    0,    0   ], // 9
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0   ], // 8
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0   ], // 7
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0   ], // 6
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0   ], // 5
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0   ], // 4
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0   ], // 3
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0   ], // 2
];

// 7. BB Call vs 4-Bet: 13.42% — flatting range vs BTN 4-bet.
final bbCallVs4Bet = <List<double>>[
  // A    K     Q     J     T     9     8     7     6     5     4     3     2
  [0.35, 0,    1,    1,    1,    1,    0.5,  0.25, 0.25, 1,    1,    0.5,  0.5 ], // A
  [0,    0,    1,    1,    1,    0.75, 0.5,  0,    0,    0,    0,    0,    0   ], // K
  [1,    0.5,  0,    1,    1,    1,    0.75, 0,    0,    0,    0,    0,    0   ], // Q
  [0.5,  0.1,  0.1,  0,    1,    1,    1,    0.75, 0,    0,    0,    0,    0   ], // J
  [0.1,  0,    0,    0,    0,    1,    1,    0.75, 0,    0,    0,    0,    0   ], // T
  [0,    0,    0,    0,    0,    0,    0.8,  1,    0.75, 0,    0,    0,    0   ], // 9
  [0,    0,    0,    0,    0,    0,    0,    1,    1,    0,    0,    0,    0   ], // 8
  [0,    0,    0,    0,    0,    0,    0.5,  0,    1,    1,    0,    0,    0   ], // 7
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    1,    0,    0   ], // 6
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    1,    0   ], // 5
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.5,  0.5,  0   ], // 4
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.2,  0   ], // 3
  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.2 ], // 2
];

/// Combos in each hand class — pair=6, suited=4, offsuit=12. Used for the
/// percent-of-1326-combos sanity check below.
int _combosInClass(String cls) {
  if (cls.length == 2) return 6;
  return cls.endsWith('s') ? 4 : 12;
}

double _percentFromMatrix(List<List<double>> m) {
  var combos = 0.0;
  for (int r = 0; r < 13; r++) {
    for (int c = 0; c < 13; c++) {
      final cls = cellLabel(r, c);
      combos += _combosInClass(cls) * m[r][c];
    }
  }
  return 100.0 * combos / 1326.0;
}

double _percentFromString(String s) {
  final parsed = parseRangeString(s);
  var combos = 0.0;
  parsed.forEach((cls, w) {
    combos += _combosInClass(cls) * w;
  });
  return 100.0 * combos / 1326.0;
}

void main() {
  final charts = <String, List<List<double>>>{
    'BTN RFI 81%':              btnRfi,
    'BB 3-Bet 20.4%':           bb3bet,
    'BB Call 55.22%':           bbCall,
    'BTN 4-Bet 5.1%':           btn4bet,
    'BTN Call vs 3-Bet 37.5%':  btnCall,
    'BB 5-Bet 3.7%':            bb5bet,
    'BB Call vs 4-Bet 13.42%':  bbCallVs4Bet,
  };
  charts.forEach((name, m) {
    final s = matrixToRange(m);
    final pctMatrix = _percentFromMatrix(m);
    final pctRoundTrip = _percentFromString(s);
    final delta = (pctMatrix - pctRoundTrip).abs();
    print('// $name');
    print('//   matrix=${pctMatrix.toStringAsFixed(2)}%  '
        'roundtrip=${pctRoundTrip.toStringAsFixed(2)}%  '
        'Δ=${delta.toStringAsFixed(3)}');
    print('range: \'$s\',');
    print('');
  });
}
