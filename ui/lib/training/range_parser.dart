/// Parse a class-level PokerStove-style range string into a map of
/// `hand class → weight`. Used to round-trip the visual range editor:
/// load text → grid, edit, serialize back to text.
///
/// Handles the common class-level syntaxes:
///   * `AA`, `KK` — single pair
///   * `22+`      — pairs at-or-above (22, 33, ..., AA)
///   * `AA-22`    — pair range (also `22-AA`)
///   * `AKs`, `AKo` — single suited / offsuit class
///   * `A2s+`     — `A2s, A3s, ..., AKs`
///   * `A5s-A2s`  — explicit class range
///   * `:weight`  — optional weight suffix per token (e.g. `AKs:0.5`)
///   * comma-separated tokens
///
/// Per-combo inputs (`AhKh`, `AdKd:0.7`) are normalized to their hand class
/// (`AKs`) — the editor is class-level, so individual-suit weights are
/// averaged into the class. This is a deliberate simplification: per-combo
/// editing is out of scope for v1.
///
/// Unrecognized tokens are silently skipped — the parser is best-effort,
/// not a strict validator. The editor opens with whatever it could parse;
/// the rest is lost on Apply (warned in the UI).
library;

const String _ranks = '23456789TJQKA';

/// Returns rank index (2 → 0, ..., A → 12), or -1 if not a rank char.
int _rankIdx(String c) => _ranks.indexOf(c);

/// Hand-class label for a (rank1, rank2, suited?) tuple. Always normalized
/// so the higher rank comes first (so AKs not KAs).
String _classLabel(int hi, int lo, bool suited) {
  final h = _ranks[hi];
  final l = _ranks[lo];
  if (hi == lo) return '$h$l'; // pair
  return '$h$l${suited ? 's' : 'o'}';
}

/// Parse a comma-separated range string into class weights. Returns an
/// empty map if the input is empty or unparseable.
Map<String, double> parseRangeString(String input) {
  final out = <String, double>{};
  // Per-class accumulators for combo-level inputs (we average over combos).
  final comboHits = <String, _Acc>{};

  for (final raw in input.split(',')) {
    final tok = raw.trim();
    if (tok.isEmpty) continue;
    // Strip optional :weight suffix.
    var body = tok;
    var weight = 1.0;
    final colon = tok.indexOf(':');
    if (colon >= 0) {
      body = tok.substring(0, colon).trim();
      weight = double.tryParse(tok.substring(colon + 1).trim()) ?? 1.0;
    }
    if (body.isEmpty) continue;
    _parseToken(body, weight, out, comboHits);
  }
  // Fold combo accumulators into out (mean weight per class). This handles
  // inputs like "AhKh:0.7,AdKd:0.5" that would otherwise be ignored.
  comboHits.forEach((cls, acc) {
    if (acc.count == 0) return;
    final mean = acc.sum / acc.count;
    // If a class-level entry already exists, prefer the explicit class
    // weight rather than the combo average.
    out.putIfAbsent(cls, () => mean);
  });
  return out;
}

void _parseToken(
  String body,
  double weight,
  Map<String, double> out,
  Map<String, _Acc> comboHits,
) {
  // Per-combo input ("AhKh") — 4 chars, both suit chars from "cdhs".
  if (body.length == 4 && _isCombo(body)) {
    final r1 = _rankIdx(body[0]);
    final r2 = _rankIdx(body[2]);
    if (r1 < 0 || r2 < 0) return;
    final s1 = body[1];
    final s2 = body[3];
    final hi = r1 >= r2 ? r1 : r2;
    final lo = r1 >= r2 ? r2 : r1;
    final suited = s1 == s2;
    final cls = _classLabel(hi, lo, suited);
    final acc = comboHits.putIfAbsent(cls, () => _Acc());
    acc.sum += weight;
    acc.count += 1;
    return;
  }

  // Class-level patterns.
  if (body.length == 2) {
    // Pair like "AA".
    if (body[0] == body[1]) {
      final i = _rankIdx(body[0]);
      if (i >= 0) out['${body[0]}${body[1]}'] = weight;
    }
    return;
  }
  if (body.length == 3) {
    if (body[2] == '+' && body[0] == body[1]) {
      // "XX+" pairs at or above.
      final start = _rankIdx(body[0]);
      if (start < 0) return;
      for (var i = start; i < 13; i++) {
        out['${_ranks[i]}${_ranks[i]}'] = weight;
      }
      return;
    }
    if (body[2] == 's' || body[2] == 'o') {
      // Single class "XYs" or "XYo".
      final r1 = _rankIdx(body[0]);
      final r2 = _rankIdx(body[1]);
      if (r1 < 0 || r2 < 0 || r1 == r2) return;
      final hi = r1 >= r2 ? r1 : r2;
      final lo = r1 >= r2 ? r2 : r1;
      out[_classLabel(hi, lo, body[2] == 's')] = weight;
      return;
    }
    return;
  }
  if (body.length == 4 && body[3] == '+' && (body[2] == 's' || body[2] == 'o')) {
    // "XYs+" or "XYo+": expand from this class up to X-just-below-X.
    final r1 = _rankIdx(body[0]); // high
    final r2 = _rankIdx(body[1]); // low (start)
    if (r1 < 0 || r2 < 0 || r1 <= r2) return;
    final suited = body[2] == 's';
    for (var i = r2; i < r1; i++) {
      out[_classLabel(r1, i, suited)] = weight;
    }
    return;
  }
  if (body.length == 5 && body[2] == '-') {
    // "AA-22": pair range. Either order.
    final a = _rankIdx(body[0]);
    final b = _rankIdx(body[3]);
    if (a < 0 || b < 0) return;
    if (body[0] != body[1] || body[3] != body[4]) return;
    final lo = a < b ? a : b;
    final hi = a < b ? b : a;
    for (var i = lo; i <= hi; i++) {
      out['${_ranks[i]}${_ranks[i]}'] = weight;
    }
    return;
  }
  if (body.length == 7 && body[3] == '-') {
    // "A5s-A2s" — explicit class range. Expects same high rank, same suited.
    final hi1 = _rankIdx(body[0]);
    final loA = _rankIdx(body[1]);
    final s1 = body[2];
    final hi2 = _rankIdx(body[4]);
    final loB = _rankIdx(body[5]);
    final s2 = body[6];
    if (hi1 != hi2 || s1 != s2 || (s1 != 's' && s1 != 'o')) return;
    final loLo = loA < loB ? loA : loB;
    final loHi = loA < loB ? loB : loA;
    final suited = s1 == 's';
    for (var i = loLo; i <= loHi; i++) {
      out[_classLabel(hi1, i, suited)] = weight;
    }
    return;
  }
}

bool _isCombo(String s) {
  // Looks like "AhKs": rank-suit-rank-suit.
  if (s.length != 4) return false;
  if (_rankIdx(s[0]) < 0 || _rankIdx(s[2]) < 0) return false;
  if (!'cdhs'.contains(s[1]) || !'cdhs'.contains(s[3])) return false;
  return true;
}

class _Acc {
  double sum = 0.0;
  int count = 0;
}

/// Inverse of [parseRangeString]: serialize a weight map back into a clean
/// PokerStove-style string. Compresses contiguous suited/offsuit ranges
/// when possible (e.g. `A2s, A3s, A4s, A5s` ⇒ `A5s-A2s`) so the output is
/// readable.
String serializeClassWeights(Map<String, double> weights) {
  // Filter out zero-weighted classes.
  final cleaned = <String, double>{};
  weights.forEach((k, v) {
    if (v > 1e-4) cleaned[k] = v;
  });
  if (cleaned.isEmpty) return '';

  final emitted = <String>[];

  // Pairs first.
  final pairs = <int, double>{};
  cleaned.forEach((cls, w) {
    if (cls.length == 2 && cls[0] == cls[1]) {
      pairs[_rankIdx(cls[0])] = w;
    }
  });
  // Group consecutive pairs with the same weight.
  final pairKeys = pairs.keys.toList()..sort();
  var i = 0;
  while (i < pairKeys.length) {
    final start = pairKeys[i];
    final w = pairs[start]!;
    var j = i;
    while (j + 1 < pairKeys.length &&
        pairKeys[j + 1] == pairKeys[j] + 1 &&
        pairs[pairKeys[j + 1]] == w) {
      j++;
    }
    final end = pairKeys[j];
    final hiCh = _ranks[end];
    final loCh = _ranks[start];
    String token;
    if (start == end) {
      token = '$hiCh$hiCh';
    } else if (end == 12) {
      // Open-ended pair range up to AA — use "XX+" form.
      token = '$loCh$loCh+';
    } else {
      token = '$hiCh$hiCh-$loCh$loCh';
    }
    emitted.add(_withWeight(token, w));
    i = j + 1;
  }

  // Suited/offsuit, grouped by high rank + suited-ness.
  for (final suited in [true, false]) {
    for (var hi = 12; hi >= 1; hi--) {
      // Collect (lo → weight) for this high rank.
      final byLo = <int, double>{};
      cleaned.forEach((cls, w) {
        if (cls.length != 3) return;
        if ((cls[2] == 's') != suited) return;
        final h = _rankIdx(cls[0]);
        final l = _rankIdx(cls[1]);
        if (h == hi) byLo[l] = w;
      });
      if (byLo.isEmpty) continue;
      final loKeys = byLo.keys.toList()..sort();
      var k = 0;
      while (k < loKeys.length) {
        final start = loKeys[k];
        final w = byLo[start]!;
        var j2 = k;
        while (j2 + 1 < loKeys.length &&
            loKeys[j2 + 1] == loKeys[j2] + 1 &&
            byLo[loKeys[j2 + 1]] == w) {
          j2++;
        }
        final end = loKeys[j2];
        final suff = suited ? 's' : 'o';
        final hiCh = _ranks[hi];
        final endCh = _ranks[end];
        final startCh = _ranks[start];
        String token;
        if (start == end) {
          token = '$hiCh$endCh$suff';
        } else if (end == hi - 1) {
          // Open-ended up to highest possible: use "XYs+" form.
          token = '$hiCh$startCh$suff+';
        } else {
          // explicit "X-end-suff -- X-start-suff" range
          token = '$hiCh$endCh$suff-$hiCh$startCh$suff';
        }
        emitted.add(_withWeight(token, w));
        k = j2 + 1;
      }
    }
  }

  return emitted.join(',');
}

String _withWeight(String token, double w) {
  if (w >= 0.9999) return token;
  // Trim trailing zeros from the weight for cleaner output.
  var s = w.toStringAsFixed(3);
  while (s.endsWith('0')) {
    s = s.substring(0, s.length - 1);
  }
  if (s.endsWith('.')) s = s.substring(0, s.length - 1);
  return '$token:$s';
}
