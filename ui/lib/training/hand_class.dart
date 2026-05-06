/// Maps individual combos ("AhKs") to the 169 hand classes used by the
/// 13×13 strategy chart, and aggregates per-combo strategies into per-class
/// weighted means.
///
/// Chart layout (PokerStove convention):
///   * Row 0 .. 12 → high-to-low ranks (A, K, Q, ..., 2)
///   * Col 0 .. 12 → high-to-low ranks (A, K, Q, ..., 2)
///   * Diagonal (row == col)        → pair
///   * Above diagonal (row < col)   → suited (e.g. AKs at [0,1])
///   * Below diagonal (row > col)   → offsuit (e.g. AKo at [1,0])
library;

import '../ffi/actions.dart';

const String _rankCh = '23456789TJQKA';

/// Convert engine rank index (2=0 .. A=12) to chart row/col index (A=0 .. 2=12).
int _rankToChart(int engineRank) => 12 - engineRank;

class HandClass {
  final int row;
  final int col;
  final String label;
  const HandClass({required this.row, required this.col, required this.label});

  bool get isPair => row == col;
  bool get isSuited => row < col;
  bool get isOffsuit => row > col;
}

/// Parse a 4-char combo string ("AhKs") into its hand class. Unknown formats
/// throw; we want loud failures here because a silent miscategorization would
/// quietly poison the whole grid.
HandClass classifyCombo(String combo) {
  if (combo.length != 4) {
    throw FormatException('combo must be 4 chars: "$combo"');
  }
  final c0 = EngineCard.parse(combo.substring(0, 2));
  final c1 = EngineCard.parse(combo.substring(2, 4));
  final r0 = EngineCard.rankOf(c0);
  final r1 = EngineCard.rankOf(c1);
  final s0 = EngineCard.suitOf(c0);
  final s1 = EngineCard.suitOf(c1);
  final hi = r0 > r1 ? r0 : r1;
  final lo = r0 > r1 ? r1 : r0;

  if (hi == lo) {
    final i = _rankToChart(hi);
    return HandClass(row: i, col: i, label: '${_rankCh[hi]}${_rankCh[hi]}');
  }
  final hiCh = _rankCh[hi];
  final loCh = _rankCh[lo];
  if (s0 == s1) {
    return HandClass(
      row: _rankToChart(hi),
      col: _rankToChart(lo),
      label: '$hiCh${loCh}s',
    );
  }
  return HandClass(
    row: _rankToChart(lo),
    col: _rankToChart(hi),
    label: '$hiCh${loCh}o',
  );
}

/// One cell of the aggregated strategy chart.
class ChartCell {
  final int row;
  final int col;
  final String label;
  /// Total reach weight summed across all combos in this cell. 0 = no live
  /// combos here (off-range or board-blocked); render greyed-out.
  final double totalWeight;
  /// Per-action mixed-strategy frequencies. Sums to ~1 when totalWeight > 0.
  /// Length == number of actions at the node.
  final List<double> actionFreqs;
  /// Per-action weighted-mean EV in chips. Length == number of actions.
  final List<double> actionEvs;

  const ChartCell({
    required this.row,
    required this.col,
    required this.label,
    required this.totalWeight,
    required this.actionFreqs,
    required this.actionEvs,
  });
}

/// Aggregate per-combo strategies into a 169-cell grid. Caller passes the
/// node's strategy/EV/weight arrays plus the player's combo list (so we can
/// classify each row).
///
/// Returns a 13×13 row-major matrix with `null` in cells that have no combos.
/// `null` (rather than a zero-weight cell) lets the grid render an
/// unambiguous "no hands here" appearance — important for the user to tell
/// "the solver folds these" from "these never reach this node".
List<List<ChartCell?>> buildChart({
  required List<String> combos,
  required List<List<double>> strategy,
  required List<List<double>> ev,
  required List<double> weights,
  required int numActions,
}) {
  // Per-cell accumulators.
  final wSum = List.filled(169, 0.0);
  final actionWStrat = List.generate(169, (_) => List<double>.filled(numActions, 0.0));
  final actionWEv = List.generate(169, (_) => List<double>.filled(numActions, 0.0));

  for (int h = 0; h < combos.length; h++) {
    final w = weights.isNotEmpty ? weights[h] : 1.0;
    if (w <= 0.0) continue;
    final cls = classifyCombo(combos[h]);
    final idx = cls.row * 13 + cls.col;
    wSum[idx] += w;
    final s = strategy[h];
    final e = ev[h];
    for (int a = 0; a < numActions; a++) {
      actionWStrat[idx][a] += w * s[a];
      actionWEv[idx][a] += w * e[a];
    }
  }

  final grid = List<List<ChartCell?>>.generate(
    13,
    (_) => List<ChartCell?>.filled(13, null, growable: false),
    growable: false,
  );

  for (int r = 0; r < 13; r++) {
    for (int c = 0; c < 13; c++) {
      final idx = r * 13 + c;
      final tw = wSum[idx];
      if (tw <= 0.0) continue;
      final freqs = List<double>.generate(numActions, (a) => actionWStrat[idx][a] / tw);
      final evs = List<double>.generate(numActions, (a) => actionWEv[idx][a] / tw);
      grid[r][c] = ChartCell(
        row: r,
        col: c,
        label: _cellLabel(r, c),
        totalWeight: tw,
        actionFreqs: freqs,
        actionEvs: evs,
      );
    }
  }
  return grid;
}

String _cellLabel(int row, int col) {
  // Inverse of _rankToChart.
  final rRow = _rankCh[12 - row];
  final rCol = _rankCh[12 - col];
  if (row == col) return '$rRow$rRow';
  if (row < col) {
    // suited: high rank corresponds to row, low rank to col.
    return '$rRow${rCol}s';
  }
  // offsuit: high rank corresponds to col, low rank to row.
  return '$rCol${rRow}o';
}

/// Headers for axis labels.
const List<String> chartRanks = [
  'A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2',
];
