/// Convert a list of combos with reach weights into a PokerStove range
/// string accepted by `pt-solver`.
///
/// Output format: `AhKh:0.85,AdKd:0.50,...`. Combos with weight ≈ 1 are
/// emitted without the `:weight` suffix; combos with weight ≤ 0 are dropped
/// entirely. Optionally a card byte can be excluded to remove combos that
/// share a card with the next dealt board card (since those reach weight 0
/// after the deal anyway).
///
/// This is the bridge that lets us re-solve a subgame from a deep node:
/// the parent action node's `weights` arrays *are* the conditional ranges
/// at that node, so re-emitting them as a range string and feeding back
/// into pt-solver gives a fresh solve rooted at the same range distribution.
library;

import '../ffi/actions.dart';

/// Serialize per-combo weights to a PokerStove range string.
///
/// * [combos] — 4-char combo strings ("AhKh", "AsKs", ...).
/// * [weights] — same length as [combos]; entries ≤ 0 are skipped.
/// * [excludeCardByte] — if non-null, combos containing this card are also
///   skipped (we use this when re-solving with a new board card pinned).
String serializeRange({
  required List<String> combos,
  required List<double> weights,
  int? excludeCardByte,
}) {
  String? excluded;
  if (excludeCardByte != null) {
    excluded = EngineCard.toReadable(excludeCardByte);
  }
  final out = StringBuffer();
  var first = true;
  for (var i = 0; i < combos.length; i++) {
    final w = weights[i];
    if (w <= 1e-4) continue;
    final combo = combos[i];
    if (excluded != null) {
      // combo is "AhKh" — two two-char sub-cards.
      if (combo.substring(0, 2) == excluded || combo.substring(2, 4) == excluded) {
        continue;
      }
    }
    if (!first) out.write(',');
    first = false;
    out.write(combo);
    if (w < 0.9999) {
      out.write(':');
      out.write(w.toStringAsFixed(3));
    }
  }
  return out.toString();
}
