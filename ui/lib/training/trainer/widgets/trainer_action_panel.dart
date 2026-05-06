/// The bottom interactive area of the trainer. Renders different content
/// depending on the trainer's current phase:
///
///   awaitingUser       → action picker buttons
///   revealingFeedback  → GTO mix + EV table for the user's just-played
///                        action; "Continue" advances
///   chancePending      → card picker → triggers caller-supplied resolve
///   resolvingSubgame   → spinner + status
///   terminal           → "hand ended" placeholder; the real summary lives
///                        on a dedicated panel above this area
library;

import 'package:flutter/material.dart';

import '../../../ffi/actions.dart';
import '../../action_palette.dart';
import '../../solver_runner.dart';
import '../showdown.dart';
import '../trainer_session.dart';

class TrainerActionPanel extends StatelessWidget {
  final TrainerSession session;
  /// Async callback the screen wires up to spawn the subgame solver. The
  /// panel emits the picked card and the requested depth; the caller does
  /// the work.
  final Future<void> Function(int pickedCardByte, SolveDepth depth)
      onResolveSubgame;

  const TrainerActionPanel({
    super.key,
    required this.session,
    required this.onResolveSubgame,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF24282B),
        border: Border.all(color: const Color(0xFF4A4E52), width: 1),
        borderRadius: BorderRadius.circular(10),
      ),
      child: switch (session.phase) {
        TrainerPhase.awaitingUser => _ActionButtons(session: session),
        TrainerPhase.revealingFeedback => _Feedback(session: session),
        TrainerPhase.chancePending => _ChancePicker(
            session: session,
            onResolveSubgame: onResolveSubgame,
          ),
        TrainerPhase.resolvingSubgame => _ResolvingIndicator(session: session),
        TrainerPhase.terminal => _TerminalPlaceholder(session: session),
        TrainerPhase.abandoned => _AbandonedPlaceholder(session: session),
      },
    );
  }
}

// ── Phase: awaitingUser ─────────────────────────────────────────────────────

class _ActionButtons extends StatelessWidget {
  final TrainerSession session;
  const _ActionButtons({required this.session});

  @override
  Widget build(BuildContext context) {
    final n = session.currentNode;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          'YOUR TURN — pick an action',
          style: TextStyle(
            color: Color(0xFFEAE6D9),
            fontSize: 12,
            fontWeight: FontWeight.w800,
            letterSpacing: 1.4,
          ),
        ),
        const SizedBox(height: 10),
        for (var i = 0; i < n.actions.length; i++) ...[
          _ActionRow(
            label: n.actions[i],
            style: styleFor(n.actions[i], n.pot),
            onTap: () => session.chooseAction(i),
          ),
          if (i < n.actions.length - 1) const SizedBox(height: 6),
        ],
      ],
    );
  }
}

class _ActionRow extends StatelessWidget {
  final String label;
  final ActionStyle style;
  final VoidCallback onTap;

  const _ActionRow({
    required this.label,
    required this.style,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: style.color.withValues(alpha: 0.18),
          border: Border.all(color: style.color, width: 1.4),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Row(
          children: [
            Container(
              width: 12,
              height: 12,
              decoration: BoxDecoration(
                color: style.color,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                label,
                style: const TextStyle(
                  color: Color(0xFFEAE6D9),
                  fontFamily: 'monospace',
                  fontSize: 14,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
            const Icon(Icons.chevron_right, size: 18, color: Color(0xCCEAE6D9)),
          ],
        ),
      ),
    );
  }
}

// ── Phase: revealingFeedback ────────────────────────────────────────────────

class _Feedback extends StatelessWidget {
  final TrainerSession session;
  const _Feedback({required this.session});

  @override
  Widget build(BuildContext context) {
    final r = session.pendingFeedback!;
    final n = session.currentNode;
    final isBest = r.yourActionIdx == r.bestActionIdx;
    // EV gap colour-coded to set emotional tone before the user reads the
    // numbers. Thresholds in chips (the unit EVs are stored in by pt-solver).
    // Calibrated against typical 60-chip starting pots: <0.5 chips ≈ rounding;
    // 3+ chips is a real mistake.
    final gapColor = r.evGap < 0.5
        ? const Color(0xFF6FDC84)
        : r.evGap < 3.0
            ? const Color(0xFFD4B43F)
            : const Color(0xFFDC6F6F);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            Icon(
              isBest ? Icons.check_circle : Icons.info_outline,
              color: gapColor,
              size: 22,
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                isBest
                    ? 'Best play — ${r.yourAction}'
                    : 'Best was ${r.bestAction}; you played ${r.yourAction}',
                style: const TextStyle(
                  color: Color(0xFFEAE6D9),
                  fontSize: 14,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: gapColor.withValues(alpha: 0.22),
                border: Border.all(color: gapColor, width: 1),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                'EV gap ${r.evGap.toStringAsFixed(2)} chips',
                style: TextStyle(
                  color: gapColor,
                  fontFamily: 'monospace',
                  fontSize: 12,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        // Per-action table.
        for (var a = 0; a < n.actions.length; a++)
          _ActionStatRow(
            label: n.actions[a],
            style: styleFor(n.actions[a], n.pot),
            mix: r.mix[a],
            ev: r.evs[a],
            isUser: a == r.yourActionIdx,
            isBest: a == r.bestActionIdx,
          ),
        const SizedBox(height: 10),
        Align(
          alignment: Alignment.centerRight,
          child: FilledButton.icon(
            onPressed: session.continueAfterFeedback,
            icon: const Icon(Icons.arrow_forward, size: 16),
            label: const Text('Continue'),
          ),
        ),
      ],
    );
  }
}

class _ActionStatRow extends StatelessWidget {
  final String label;
  final ActionStyle style;
  final double mix;
  final double ev;
  final bool isUser;
  final bool isBest;

  const _ActionStatRow({
    required this.label,
    required this.style,
    required this.mix,
    required this.ev,
    required this.isUser,
    required this.isBest,
  });

  @override
  Widget build(BuildContext context) {
    final mixPct = (mix * 100).clamp(0.0, 100.0);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          // Marker tag — "you" or "best" flag.
          SizedBox(
            width: 36,
            child: Row(
              children: [
                if (isUser)
                  const Text(
                    '➤',
                    style: TextStyle(
                      color: Color(0xFFEAE6D9),
                      fontWeight: FontWeight.w800,
                      fontSize: 14,
                    ),
                  ),
                if (isBest)
                  const Padding(
                    padding: EdgeInsets.only(left: 2),
                    child: Icon(
                      Icons.star,
                      size: 13,
                      color: Color(0xFFD4B43F),
                    ),
                  ),
              ],
            ),
          ),
          Container(
            width: 12,
            height: 12,
            decoration: BoxDecoration(
              color: style.color,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(width: 6),
          Expanded(
            child: Text(
              label,
              style: const TextStyle(
                color: Color(0xCCEAE6D9),
                fontFamily: 'monospace',
                fontSize: 12,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          // Inline mix bar — 60px max, scales with frequency.
          SizedBox(
            width: 60,
            child: Stack(
              children: [
                Container(
                  height: 8,
                  decoration: BoxDecoration(
                    color: const Color(0xFF1B1E20),
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                FractionallySizedBox(
                  widthFactor: mix.clamp(0.0, 1.0),
                  child: Container(
                    height: 8,
                    decoration: BoxDecoration(
                      color: style.color,
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 44,
            child: Text(
              '${mixPct.toStringAsFixed(0)}%',
              textAlign: TextAlign.right,
              style: const TextStyle(
                color: Color(0xCCEAE6D9),
                fontFamily: 'monospace',
                fontSize: 12,
              ),
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 64,
            child: Text(
              ev.toStringAsFixed(2),
              textAlign: TextAlign.right,
              style: const TextStyle(
                color: Color(0xFFEAE6D9),
                fontFamily: 'monospace',
                fontSize: 12,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Phase: chancePending ────────────────────────────────────────────────────

class _ChancePicker extends StatefulWidget {
  final TrainerSession session;
  final Future<void> Function(int pickedCardByte, SolveDepth depth)
      onResolveSubgame;

  const _ChancePicker({
    required this.session,
    required this.onResolveSubgame,
  });

  @override
  State<_ChancePicker> createState() => _ChancePickerState();
}

class _ChancePickerState extends State<_ChancePicker> {
  int? _selectedCard;
  // Default to "Flop" (smallest dump, historical behaviour). The trainer
  // exposes the choice up-front so a user who knows they'll want a deeper
  // resolve doesn't have to chain multiple expansions.
  SolveDepth _depth = SolveDepth.flop;

  @override
  Widget build(BuildContext context) {
    final n = widget.session.currentNode;
    final pendingStreet = n.board.length == 3 ? 'TURN' : 'RIVER';

    // Cards already in play (board + both hole hands) — exclude from picker.
    final usedBytes = <int>{
      ...n.board.map(EngineCard.parse),
      ...widget.session.userHandStr.split('').isEmpty
          ? const <int>[]
          : _comboCards(widget.session.userHandStr),
      ..._comboCards(widget.session.oppHandStrRevealed),
    };

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: const Color(0x33D4B43F),
            border: Border.all(color: const Color(0xFFD4B43F), width: 1),
            borderRadius: BorderRadius.circular(6),
          ),
          child: Row(
            children: [
              const Icon(Icons.alt_route,
                  color: Color(0xFFD4B43F), size: 18),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Solve doesn\'t cover the $pendingStreet. '
                  'Pick a card to deal and we\'ll solve a fresh subgame.',
                  style: const TextStyle(
                    color: Color(0xFFEAE6D9),
                    fontSize: 12,
                    height: 1.4,
                  ),
                ),
              ),
            ],
          ),
        ),
        if (widget.session.statusMessage != null) ...[
          const SizedBox(height: 8),
          Text(
            widget.session.statusMessage!,
            style: const TextStyle(
              color: Color(0xFFDC6F6F),
              fontSize: 12,
            ),
          ),
        ],
        const SizedBox(height: 10),
        _CardGrid(
          excluded: usedBytes,
          selected: _selectedCard,
          onPick: (c) => setState(() => _selectedCard = c),
        ),
        const SizedBox(height: 10),
        _DepthPicker(
          depth: _depth,
          onChange: (d) => setState(() => _depth = d),
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            FilledButton.icon(
              onPressed: _selectedCard == null
                  ? null
                  : () => widget.onResolveSubgame(_selectedCard!, _depth),
              icon: const Icon(Icons.play_arrow, size: 16),
              label: Text(_selectedCard == null
                  ? 'Pick a card first'
                  : 'Deal ${EngineCard.toReadable(_selectedCard!)} '
                      '· ${_depth.human} & solve'),
            ),
            const SizedBox(width: 10),
            // Escape hatch for users who don't want to spend solver time on
            // a subgame they don't care about. Goes straight to summary.
            OutlinedButton.icon(
              onPressed: widget.session.abandonHand,
              icon: const Icon(Icons.stop_circle_outlined, size: 16),
              label: const Text('Stop here'),
            ),
          ],
        ),
      ],
    );
  }

  List<int> _comboCards(String combo) {
    if (combo.length != 4) return const [];
    try {
      return [
        EngineCard.parse(combo.substring(0, 2)),
        EngineCard.parse(combo.substring(2, 4)),
      ];
    } catch (_) {
      return const [];
    }
  }
}

/// Three-way segmented control for choosing how deep the subgame resolve
/// should walk: Flop = current street's actions only (smallest dump);
/// Turn = walk through every turn card; River = full both-streets walk.
/// Identical knob to the New-solve dialog; surfaced here so the trainer
/// user doesn't have to guess.
class _DepthPicker extends StatelessWidget {
  final SolveDepth depth;
  final ValueChanged<SolveDepth> onChange;

  const _DepthPicker({required this.depth, required this.onChange});

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        const Padding(
          padding: EdgeInsets.only(right: 8),
          child: Text(
            'DEPTH',
            style: TextStyle(
              color: Color(0x99EAE6D9),
              fontSize: 11,
              letterSpacing: 1.6,
              fontWeight: FontWeight.w800,
            ),
          ),
        ),
        Container(
          padding: const EdgeInsets.all(2),
          decoration: BoxDecoration(
            color: const Color(0xFF1B1E20),
            borderRadius: BorderRadius.circular(6),
            border: Border.all(color: const Color(0xFF45525A), width: 1),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              for (final d in SolveDepth.values)
                _pill(
                  label: d.human,
                  active: d == depth,
                  onTap: () => onChange(d),
                ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _pill({
    required String label,
    required bool active,
    required VoidCallback onTap,
  }) {
    final fg = active
        ? const Color(0xFFEAE6D9)
        : const Color(0xCCEAE6D9);
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: active ? const Color(0xFF1B2D40) : Colors.transparent,
          borderRadius: BorderRadius.circular(4),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: fg,
            fontFamily: 'monospace',
            fontSize: 11,
            fontWeight: active ? FontWeight.w800 : FontWeight.w700,
          ),
        ),
      ),
    );
  }
}

class _CardGrid extends StatelessWidget {
  final Set<int> excluded;
  final int? selected;
  final void Function(int) onPick;

  const _CardGrid({
    required this.excluded,
    required this.selected,
    required this.onPick,
  });

  @override
  Widget build(BuildContext context) {
    // 4 rows (suits) × 13 cols (ranks). Compact tiles, click to pick.
    return Column(
      children: [
        for (int s = 0; s < 4; s++)
          Row(
            children: [
              for (int r = 0; r < 13; r++) _tile(s, r),
            ],
          ),
      ],
    );
  }

  Widget _tile(int suit, int rank) {
    final card = EngineCard.make(rank, suit);
    final disabled = excluded.contains(card);
    final isSel = selected == card;
    final isRed = suit == 1 || suit == 2;
    return Expanded(
      child: Padding(
        padding: const EdgeInsets.all(1.5),
        child: InkWell(
          onTap: disabled ? null : () => onPick(card),
          child: Container(
            height: 26,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: disabled
                  ? const Color(0xFF24282B)
                  : isSel
                      ? const Color(0xFF1B2D40)
                      : const Color(0xFFF8F4E9),
              border: Border.all(
                color: isSel
                    ? const Color(0xFF6FB3DC)
                    : const Color(0xFF999999),
                width: isSel ? 1.6 : 1,
              ),
              borderRadius: BorderRadius.circular(3),
            ),
            child: disabled
                ? const Icon(Icons.block,
                    size: 12, color: Color(0x55EAE6D9))
                : Text(
                    EngineCard.toReadable(card),
                    style: TextStyle(
                      color: isSel
                          ? const Color(0xFFEAE6D9)
                          : (isRed
                              ? const Color(0xFFC72D2D)
                              : const Color(0xFF1A1A1A)),
                      fontFamily: 'monospace',
                      fontSize: 11,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
          ),
        ),
      ),
    );
  }
}

// ── Phase: resolvingSubgame ─────────────────────────────────────────────────

class _ResolvingIndicator extends StatelessWidget {
  final TrainerSession session;
  const _ResolvingIndicator({required this.session});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 22),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const SizedBox(
            width: 18,
            height: 18,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          const SizedBox(width: 14),
          Text(
            session.statusMessage ?? 'Solving subgame…',
            style: const TextStyle(
              color: Color(0xFFEAE6D9),
              fontSize: 13,
            ),
          ),
        ],
      ),
    );
  }
}

// ── Phase: terminal ─────────────────────────────────────────────────────────

class _TerminalPlaceholder extends StatelessWidget {
  final TrainerSession session;
  const _TerminalPlaceholder({required this.session});

  @override
  Widget build(BuildContext context) {
    final n = session.currentNode;
    // Find who acted last to decide between fold-terminal and showdown.
    // We walk backwards because chance / street markers are interleaved.
    final last = session.history.lastWhere(
      (e) => e.kind == 'user' || e.kind == 'opp',
      orElse: () => const TrainerHistoryEntry(kind: 'opp', label: '—'),
    );
    final isFold = last.label.toLowerCase().contains('fold');

    // Investments per side at the terminal — needed for the chip-delta math.
    // The scenario stores `effectiveStack` as the starting stack and current
    // `stacks[i]` as remaining; difference = invested.
    final userIdx = session.userSeat == 'oop' ? 0 : 1;
    final oppIdx = 1 - userIdx;
    final userInvested = session.scenario.effectiveStack - n.stacks[userIdx];
    final oppInvested = session.scenario.effectiveStack - n.stacks[oppIdx];

    if (isFold) {
      // Whoever played 'Fold' last loses what they put in; the other wins
      // the pot minus their own investment. Determining the folder: 'user'
      // kind means user folded; 'opp' means opp folded.
      final userFolded = last.kind == 'user';
      final yourDelta = userFolded
          ? -userInvested
          : (n.pot - userInvested);
      return _terminalCard(
        icon: Icons.flag,
        accent: const Color(0xFFD4B43F),
        title: userFolded ? 'You folded.' : 'Opponent folded.',
        chipDelta: yourDelta,
        pot: n.pot,
      );
    }

    // Showdown — try ptEval7 if the board is complete (it usually will be
    // at a non-fold terminal in a deeply-walked solve, but a partial-depth
    // solve can produce a check-call-down terminal short of the river).
    final isFullBoard = n.board.length == 5;
    if (!isFullBoard) {
      // Defensive: rare case where solver dump labels something terminal
      // before the river. Just show pot.
      return _terminalCard(
        icon: Icons.flag,
        accent: const Color(0xFFD4B43F),
        title: 'Hand ended (showdown short of river)',
        chipDelta: 0,
        pot: n.pot,
      );
    }

    final result = evaluateShowdown(
      userCards: _comboBytes(session.userHandStr),
      oppCards: _comboBytes(session.oppHandStrRevealed),
      board: n.board,
      pot: n.pot,
      userInvested: userInvested,
      oppInvested: oppInvested,
    );
    if (result == null) {
      return _terminalCard(
        icon: Icons.flag,
        accent: const Color(0xFFD4B43F),
        title: 'Showdown reached (engine eval unavailable)',
        chipDelta: 0,
        pot: n.pot,
      );
    }
    final title = switch (result.winner) {
      ShowdownWinner.you => 'You win at showdown!',
      ShowdownWinner.opp => 'Opponent wins at showdown.',
      ShowdownWinner.tie => 'Showdown — chopped pot.',
    };
    final accent = switch (result.winner) {
      ShowdownWinner.you => const Color(0xFF6FDC84),
      ShowdownWinner.opp => const Color(0xFFDC6F6F),
      ShowdownWinner.tie => const Color(0xFFD4B43F),
    };
    return _terminalCard(
      icon: result.winner == ShowdownWinner.you
          ? Icons.emoji_events
          : Icons.flag,
      accent: accent,
      title: title,
      chipDelta: result.yourChipDelta,
      pot: n.pot,
    );
  }

  /// Common renderer — keeps fold and showdown branches visually consistent.
  Widget _terminalCard({
    required IconData icon,
    required Color accent,
    required String title,
    required int chipDelta,
    required int pot,
  }) {
    final deltaSign = chipDelta > 0 ? '+' : '';
    final deltaColor = chipDelta > 0
        ? const Color(0xFF6FDC84)
        : chipDelta < 0
            ? const Color(0xFFDC6F6F)
            : const Color(0xCCEAE6D9);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 14),
      child: Column(
        children: [
          Icon(icon, size: 30, color: accent),
          const SizedBox(height: 8),
          Text(
            title,
            style: TextStyle(
              color: accent,
              fontSize: 14,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 6),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                'pot $pot',
                style: const TextStyle(
                  color: Color(0x99EAE6D9),
                  fontFamily: 'monospace',
                  fontSize: 12,
                ),
              ),
              const SizedBox(width: 14),
              Text(
                'chip Δ $deltaSign$chipDelta',
                style: TextStyle(
                  color: deltaColor,
                  fontFamily: 'monospace',
                  fontSize: 13,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  static List<int> _comboBytes(String combo) {
    if (combo.length != 4) return const [];
    try {
      return [
        EngineCard.parse(combo.substring(0, 2)),
        EngineCard.parse(combo.substring(2, 4)),
      ];
    } catch (_) {
      return const [];
    }
  }
}

// ── Phase: abandoned ────────────────────────────────────────────────────────

/// Rendered when the user clicked "Stop here" at a chance_pending. The hand
/// didn't reach a real terminal, so we don't have a fold/showdown outcome —
/// just a "you stopped" status. Decision summary still renders below this.
class _AbandonedPlaceholder extends StatelessWidget {
  final TrainerSession session;
  const _AbandonedPlaceholder({required this.session});

  @override
  Widget build(BuildContext context) {
    final n = session.currentNode;
    final pendingStreet = n.board.length == 3
        ? 'turn'
        : n.board.length == 4
            ? 'river'
            : 'next street';
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 14),
      child: Column(
        children: [
          const Icon(Icons.stop_circle_outlined,
              size: 28, color: Color(0xFFD4B43F)),
          const SizedBox(height: 8),
          Text(
            'Hand stopped before the $pendingStreet.',
            style: const TextStyle(
              color: Color(0xFFEAE6D9),
              fontSize: 13,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Subgame solve was skipped — outcome is undetermined. '
            'Recap below covers the decisions you completed.',
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: Color(0x99EAE6D9),
              fontSize: 12,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }
}
