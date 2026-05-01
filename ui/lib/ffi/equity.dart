/// Heads-up hand equity (win / lose / tie %) when both holes are known.
///
/// - Postflop streets: enumerate all remaining-deck completions exhaustively
///   (≤ 990 combos at flop, ≤ 44 at turn, 1 at river — fast).
/// - Preflop: 1.7M completions is too slow for live UI, so we Monte Carlo
///   sample. Default 4000 samples → ~0.8% standard error, runs in ~50 ms.
///
/// Uses `pt_eval_7` for the per-hand rank lookup (lower = better).
library;

import 'dart:ffi' as ffi;
import 'dart:math';

import 'package:ffi/ffi.dart';

import 'actions.dart';
import 'engine.dart';

class EquityResult {
  /// Probability the player wins outright.
  final double sbWin;
  final double bbWin;
  /// Probability of a chopped pot (split equally).
  final double tie;
  /// True if computed exhaustively, false for MC sampling.
  final bool exhaustive;
  /// Number of (sampled or enumerated) board completions evaluated.
  final int trials;

  const EquityResult({
    required this.sbWin,
    required this.bbWin,
    required this.tie,
    required this.exhaustive,
    required this.trials,
  });

  /// Each player's effective equity = win + 0.5 × tie (tournament/cash game
  /// chip-EV convention). Sums to 1.0.
  double sbEquity() => sbWin + 0.5 * tie;
  double bbEquity() => bbWin + 0.5 * tie;
}

/// Compute equity given known hole pairs and the visible portion of the
/// board. `visibleBoard` should be the cards currently revealed by street
/// (length 0/3/4/5); any tail entries beyond `street.nVisibleBoard` are
/// ignored.
///
/// `mcSamples` only applies preflop. Throws if any hole card is NO_CARD or
/// if there's a duplicate among the inputs.
EquityResult computeEquity({
  required PokerEngine eng,
  required List<int> sbHole,
  required List<int> bbHole,
  required List<int> visibleBoard,
  required Street street,
  int mcSamples = 4000,
  int? rngSeed,
}) {
  if (sbHole.length != 2 || bbHole.length != 2) {
    throw ArgumentError('Each hole must be 2 cards');
  }
  for (final c in [...sbHole, ...bbHole]) {
    if (c == EngineCard.noCard) {
      throw ArgumentError('Equity needs both hole pairs known');
    }
  }

  // Slice visible board to street.
  final nVis = street.nVisibleBoard.clamp(0, visibleBoard.length);
  final vis  = visibleBoard.sublist(0, nVis)
      .where((c) => c != EngineCard.noCard)
      .toList(growable: false);

  // Build the deck of remaining cards.
  final used = <int>{...sbHole, ...bbHole, ...vis};
  if (used.length != 4 + vis.length) {
    throw ArgumentError('Duplicate cards in equity inputs');
  }
  final deck = <int>[];
  for (int c = 0; c < EngineCard.numCards; c++) {
    if (!used.contains(c)) deck.add(c);
  }
  final needed = 5 - vis.length;

  // Tally containers shared across both code paths.
  int sbWins = 0, bbWins = 0, ties = 0, total = 0;

  // Native scratch buffers (FFI hot path — alloc once).
  final sbBuf = malloc<ffi.Uint8>(7);
  final bbBuf = malloc<ffi.Uint8>(7);
  try {
    sbBuf[0] = sbHole[0]; sbBuf[1] = sbHole[1];
    bbBuf[0] = bbHole[0]; bbBuf[1] = bbHole[1];
    for (int i = 0; i < vis.length; i++) {
      sbBuf[2 + i] = vis[i];
      bbBuf[2 + i] = vis[i];
    }

    void evalOne() {
      final sbRank = eng.native.ptEval7(sbBuf);
      final bbRank = eng.native.ptEval7(bbBuf);
      // Lower = better in pt's hand evaluator.
      if (sbRank < bbRank) {
        sbWins++;
      } else if (bbRank < sbRank) {
        bbWins++;
      } else {
        ties++;
      }
      total++;
    }

    if (needed == 0) {
      // River: no enumeration, just one comparison.
      evalOne();
      return _packResult(sbWins, bbWins, ties, total, exhaustive: true);
    }

    // Decide exhaustive vs MC. nCk for needed in {1,2}: trivially small.
    final exhaustiveCount = _nCk(deck.length, needed);
    final exhaustive = exhaustiveCount <= 20000;   // turn (≤44), flop (≤990) included

    if (exhaustive) {
      _enumerate(deck, needed, (combo) {
        for (int i = 0; i < needed; i++) {
          sbBuf[2 + vis.length + i] = combo[i];
          bbBuf[2 + vis.length + i] = combo[i];
        }
        evalOne();
      });
    } else {
      final rng = Random(rngSeed ?? DateTime.now().microsecondsSinceEpoch);
      final pick = List<int>.filled(needed, 0);
      // Deck array we'll partial-Fisher-Yates each sample.
      final scratch = List<int>.from(deck);
      for (int s = 0; s < mcSamples; s++) {
        // Partial Fisher-Yates: pick `needed` distinct cards from deck.
        for (int i = 0; i < needed; i++) {
          final j = i + rng.nextInt(scratch.length - i);
          final tmp = scratch[i];
          scratch[i] = scratch[j];
          scratch[j] = tmp;
          pick[i] = scratch[i];
        }
        for (int i = 0; i < needed; i++) {
          sbBuf[2 + vis.length + i] = pick[i];
          bbBuf[2 + vis.length + i] = pick[i];
        }
        evalOne();
      }
    }

    return _packResult(sbWins, bbWins, ties, total,
        exhaustive: exhaustive);
  } finally {
    malloc.free(sbBuf);
    malloc.free(bbBuf);
  }
}

EquityResult _packResult(int sbWins, int bbWins, int ties, int total,
    {required bool exhaustive}) {
  if (total == 0) {
    return const EquityResult(
        sbWin: 0, bbWin: 0, tie: 1, exhaustive: true, trials: 0);
  }
  return EquityResult(
    sbWin: sbWins / total,
    bbWin: bbWins / total,
    tie:   ties   / total,
    exhaustive: exhaustive,
    trials: total,
  );
}

/// Enumerate all k-combinations of items, calling `consumer(combo)` for each.
/// Reuses a single scratch list; consumer must not retain a reference past
/// its return.
void _enumerate(List<int> items, int k, void Function(List<int>) consumer) {
  if (k == 0) { consumer(const []); return; }
  if (k > items.length) return;
  final combo = List<int>.filled(k, 0);
  void rec(int start, int depth) {
    if (depth == k) { consumer(combo); return; }
    final last = items.length - (k - depth) + 1;
    for (int i = start; i < last; i++) {
      combo[depth] = items[i];
      rec(i + 1, depth + 1);
    }
  }
  rec(0, 0);
}

int _nCk(int n, int k) {
  if (k < 0 || k > n) return 0;
  if (k == 0 || k == n) return 1;
  k = k > n - k ? n - k : k;
  int r = 1;
  for (int i = 0; i < k; i++) {
    r = r * (n - i) ~/ (i + 1);
  }
  return r;
}

/// Read the engine's category label for a rank ("Straight Flush", etc.).
String rankCategory(PokerEngine eng, int rank) {
  const cap = 32;
  final buf = malloc<ffi.Char>(cap);
  try {
    final n = eng.native.ptEvalRankCategory(rank, buf, cap);
    if (n < 0) return '?';
    final bytes = buf.cast<ffi.Uint8>().asTypedList(n.clamp(0, cap - 1));
    return String.fromCharCodes(bytes);
  } finally {
    malloc.free(buf);
  }
}
