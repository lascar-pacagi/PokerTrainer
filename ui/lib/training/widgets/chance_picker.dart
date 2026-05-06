/// Card-grid picker shown in place of the strategy chart when the current
/// node is a chance node. Two modes:
///
///   * **Walked chance** (`node.isChance`): the dump already has children
///     for each runout — clicking a card just navigates to the existing
///     subtree. Free, instant.
///   * **Pending chance** (`node.isChancePending`): no children — the dump
///     stopped at this boundary. Clicking a card spawns a subgame solve
///     with the runout pinned and the parent's weights re-emitted as the
///     new range. Takes seconds; new scenario auto-loads.
///
/// Uses a 4×13 layout (suits × ranks) — same shape every time so the user
/// builds a stable mental map of "where the deck lives."
library;

import 'package:flutter/material.dart';

import '../../ffi/actions.dart';
import '../scenario.dart';
import '../solver_runner.dart';
import '../training_state.dart';

class ChancePicker extends StatelessWidget {
  final TrainingState state;
  final ScenarioNode node;
  /// Required when the node is `chance_pending` — used to spawn the subgame
  /// solve. Unused when the node has children (walked chance).
  final SolverRunner? solver;
  /// Called with the resulting JSON path after a successful subgame solve.
  /// Same signature as the spot-config dialog's `onSolved`.
  final void Function(String path)? onSubgameSolved;

  const ChancePicker({
    super.key,
    required this.state,
    required this.node,
    this.solver,
    this.onSubgameSolved,
  });

  bool get _isPending => node.isChancePending;

  @override
  Widget build(BuildContext context) {
    final byByte = <int, ChildEdge>{
      for (final c in node.children) c.actionIdx: c,
    };
    final boardBytes = <int>{
      for (final cs in node.board) EngineCard.parse(cs),
    };
    final clickable = _isPending
        ? (52 - boardBytes.length)
        : node.children.length;

    final solver = this.solver;
    final pending = _isPending && solver != null;

    return AnimatedBuilder(
      animation: solver ?? const _NoListenable(),
      builder: (context, _) {
        final running = solver?.running ?? false;
        return Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 760),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _Header(isPending: _isPending),
                const SizedBox(height: 18),
                _CardGrid(
                  byByte: byByte,
                  boardBytes: boardBytes,
                  isPending: _isPending,
                  disabled: running,
                  onPickWalked: (edge) => state.descend(edge),
                  onPickPending: pending
                      ? (cardByte) async {
                          final scenario = state.scenario;
                          if (scenario == null) return;
                          final path = await solver.expandChancePending(
                            scenario: scenario,
                            chancePending: node,
                            pickedCardByte: cardByte,
                          );
                          if (path != null && onSubgameSolved != null) {
                            onSubgameSolved!(path);
                          }
                        }
                      : null,
                ),
                const SizedBox(height: 14),
                _Legend(
                  count: clickable,
                  isPending: _isPending,
                ),
                if (running) ...[
                  const SizedBox(height: 14),
                  _ProgressPanel(solver: solver!),
                ],
              ],
            ),
          ),
        );
      },
    );
  }
}

class _Header extends StatelessWidget {
  final bool isPending;
  const _Header({required this.isPending});

  @override
  Widget build(BuildContext context) {
    if (isPending) {
      return Column(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: const Color(0x33D4B43F),
              border: Border.all(color: const Color(0xFFD4B43F), width: 1),
              borderRadius: BorderRadius.circular(4),
            ),
            child: const Text(
              'CLICK A CARD → FRESH SUBGAME SOLVE',
              style: TextStyle(
                color: Color(0xFFD4B43F),
                fontSize: 12,
                fontWeight: FontWeight.w800,
                letterSpacing: 1.2,
              ),
            ),
          ),
          const SizedBox(height: 10),
          const Text(
            'Deal a card to expand',
            style: TextStyle(
              color: Color(0xFFEAE6D9),
              fontSize: 18,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.4,
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            'This chance branch was not pre-walked. Clicking a card runs a '
            'fresh subgame solve using the conditional ranges at this node '
            '— it does not navigate the existing root tree. Takes a few '
            'seconds; the result loads as a new (sub) scenario.',
            style: TextStyle(
              color: Color(0xCCEAE6D9),
              fontSize: 13,
              height: 1.4,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      );
    }
    return Column(
      children: const [
        Text(
          'Pick the next card',
          style: TextStyle(
            color: Color(0xFFEAE6D9),
            fontSize: 18,
            fontWeight: FontWeight.w800,
            letterSpacing: 0.4,
          ),
        ),
        SizedBox(height: 4),
        Text(
          'The solver computed strategy for every runout. Click any '
          'available card to descend into the existing subtree.',
          style: TextStyle(
            color: Color(0x99EAE6D9),
            fontSize: 13,
            height: 1.4,
          ),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }
}

class _CardGrid extends StatelessWidget {
  final Map<int, ChildEdge> byByte;
  final Set<int> boardBytes;
  final bool isPending;
  final bool disabled;
  final void Function(ChildEdge)? onPickWalked;
  final void Function(int cardByte)? onPickPending;

  const _CardGrid({
    required this.byByte,
    required this.boardBytes,
    required this.isPending,
    required this.disabled,
    required this.onPickWalked,
    required this.onPickPending,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        for (int suit = 0; suit < 4; suit++)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 2),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                for (int rank = 12; rank >= 0; rank--)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 2),
                    child: _cardButton(rank, suit),
                  ),
              ],
            ),
          ),
      ],
    );
  }

  Widget _cardButton(int rank, int suit) {
    final byte = EngineCard.make(rank, suit);
    final onBoard = boardBytes.contains(byte);
    final edge = byByte[byte];
    // Walked mode: clickable iff there's an existing child. Pending mode:
    // clickable iff the card isn't on the board (no child list to consult).
    final available = isPending ? !onBoard : edge != null;
    final label = '${EngineCard.rankChars[rank]}${EngineCard.suitChars[suit]}';
    final isRed = suit == 1 || suit == 2;
    final Color fg;
    final Color bg;
    final Color border;
    VoidCallback? tap;

    if (onBoard) {
      fg = const Color(0x55EAE6D9);
      bg = const Color(0xFF2D3134);
      border = const Color(0xFF4A4E52);
    } else if (available) {
      fg = isRed ? const Color(0xFFE07A7A) : const Color(0xFFEAE6D9);
      bg = disabled ? const Color(0xFF272A2D) : const Color(0xFF34383B);
      border = disabled ? const Color(0xFF4A4E52) : const Color(0xFF6FB3DC);
      if (!disabled) {
        if (isPending) {
          tap = onPickPending == null ? null : () => onPickPending!(byte);
        } else {
          tap = (edge == null || onPickWalked == null)
              ? null
              : () => onPickWalked!(edge);
        }
      }
    } else {
      fg = const Color(0x33EAE6D9);
      bg = const Color(0xFF272A2D);
      border = const Color(0xFF4A4E52);
    }

    return InkWell(
      onTap: tap,
      borderRadius: BorderRadius.circular(4),
      child: Container(
        width: 50,
        height: 60,
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(4),
          border: Border.all(color: border, width: 1.4),
        ),
        child: Stack(
          children: [
            Center(
              child: Text(
                label,
                style: TextStyle(
                  color: fg,
                  fontFamily: 'monospace',
                  fontSize: 16,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
            if (onBoard)
              const Positioned(
                top: 2,
                right: 2,
                child: Icon(
                  Icons.dashboard,
                  size: 11,
                  color: Color(0x66EAE6D9),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _ProgressPanel extends StatelessWidget {
  final SolverRunner solver;
  const _ProgressPanel({required this.solver});

  @override
  Widget build(BuildContext context) {
    final logs = solver.stderrLines;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF272A2D),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: const Color(0xFF6FB3DC), width: 1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: const [
              SizedBox(
                width: 14,
                height: 14,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
              SizedBox(width: 10),
              Text(
                'Solving subgame…',
                style: TextStyle(
                  color: Color(0xFFEAE6D9),
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          if (logs.isNotEmpty) ...[
            const SizedBox(height: 8),
            Container(
              constraints: const BoxConstraints(maxHeight: 80),
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: const Color(0xFF1B1F22),
                borderRadius: BorderRadius.circular(4),
              ),
              child: SingleChildScrollView(
                reverse: true,
                child: Text(
                  logs.join('\n'),
                  style: const TextStyle(
                    color: Color(0xAAEAE6D9),
                    fontFamily: 'monospace',
                    fontSize: 11,
                    height: 1.4,
                  ),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

/// Stand-in Listenable for the AnimatedBuilder when there's no solver. Never
/// notifies, so the AnimatedBuilder rebuilds only when the host setStates.
class _NoListenable extends Listenable {
  const _NoListenable();
  @override
  void addListener(VoidCallback listener) {}
  @override
  void removeListener(VoidCallback listener) {}
}

class _Legend extends StatelessWidget {
  final int count;
  final bool isPending;
  const _Legend({required this.count, required this.isPending});

  @override
  Widget build(BuildContext context) {
    return Wrap(
      alignment: WrapAlignment.center,
      spacing: 18,
      runSpacing: 6,
      children: [
        _legendItem(
          color: const Color(0xFF34383B),
          borderColor: const Color(0xFF6FB3DC),
          label: isPending
              ? '$count cards available — click to solve'
              : '$count clickable runouts',
        ),
        _legendItem(
          color: const Color(0xFF2D3134),
          borderColor: const Color(0xFF4A4E52),
          label: 'On board',
        ),
        if (!isPending)
          _legendItem(
            color: const Color(0xFF272A2D),
            borderColor: const Color(0xFF4A4E52),
            label: 'Dead / isomorphic',
            dim: true,
          ),
      ],
    );
  }

  Widget _legendItem({
    required Color color,
    required Color borderColor,
    required String label,
    bool dim = false,
  }) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 14,
          height: 18,
          decoration: BoxDecoration(
            color: color,
            border: Border.all(color: borderColor),
            borderRadius: BorderRadius.circular(2),
          ),
        ),
        const SizedBox(width: 6),
        Text(
          label,
          style: TextStyle(
            color: dim ? const Color(0x66EAE6D9) : const Color(0xCCEAE6D9),
            fontSize: 12,
          ),
        ),
      ],
    );
  }
}
