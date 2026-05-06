/// Maps solver action labels ("Check", "Bet(30)", "Raise(75)", "AllIn(970)")
/// to display colors and short labels.
///
/// Bet/raise sizing buckets are computed against the pot at the node so the
/// same color scheme works across street boundaries — a 30-chip bet into a
/// 60-pot is "small", but a 30-chip bet into a 600-pot would be a min-bet
/// (still small). We deliberately keep this in the UI layer rather than in
/// the schema; bucket boundaries are taste, and may change.
library;

import 'package:flutter/material.dart';

/// Color buckets, GTO-Wizard-ish:
///   * Fold        — cool blue (the only "passive" cool color)
///   * Check       — light teal (passive but active in the hand)
///   * Call        — dark green
///   * Small bet/raise (≤ 35% pot)   — yellow
///   * Medium bet/raise (35-100%)    — orange
///   * Large bet/raise (>100%) / All — red
class ActionStyle {
  final Color color;
  /// 1–5 char display label, e.g. "F", "X", "C", "30", "75", "AI".
  final String shortLabel;
  /// Long human-readable label for tooltips/legends.
  final String longLabel;

  const ActionStyle({
    required this.color,
    required this.shortLabel,
    required this.longLabel,
  });
}

/// Parse an action label and pot context into a styled bucket.
ActionStyle styleFor(String action, int pot) {
  if (action == 'Fold') {
    return const ActionStyle(
      color: Color(0xFF3D6FA8),
      shortLabel: 'F',
      longLabel: 'Fold',
    );
  }
  if (action == 'Check') {
    return const ActionStyle(
      color: Color(0xFF5FB37A),
      shortLabel: 'X',
      longLabel: 'Check',
    );
  }
  if (action == 'Call') {
    return const ActionStyle(
      color: Color(0xFF3A7A4A),
      shortLabel: 'C',
      longLabel: 'Call',
    );
  }

  // Bet(N) / Raise(N) / AllIn(N): parse N, compare to pot.
  final amount = _parseAmount(action);
  final ratio = pot > 0 ? amount / pot : 1.0;

  final isAllIn = action.startsWith('AllIn');
  final isRaise = action.startsWith('Raise');
  final prefix = isAllIn ? 'AI' : (isRaise ? 'R' : 'B');
  final shortLabel = isAllIn ? 'AI' : '$prefix$amount';

  final Color color;
  if (isAllIn || ratio > 1.0) {
    color = const Color(0xFFC9522D); // red
  } else if (ratio > 0.35) {
    color = const Color(0xFFD47B3F); // orange
  } else {
    color = const Color(0xFFD4B43F); // yellow
  }
  return ActionStyle(
    color: color,
    shortLabel: shortLabel,
    longLabel: action,
  );
}

int _parseAmount(String action) {
  final l = action.indexOf('(');
  final r = action.indexOf(')');
  if (l < 0 || r < 0 || r <= l) return 0;
  return int.tryParse(action.substring(l + 1, r)) ?? 0;
}
