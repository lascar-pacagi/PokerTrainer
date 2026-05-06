/// Top-level layout for trainer mode. Lives inside the training tab and
/// replaces the centre+right area while in Train mode (left rail with the
/// scenario picker stays).
///
/// Wiring:
///   * Construction-time setup dialog asks the user which seat to play.
///     Once the user confirms, we build a TrainerSession and let the rest
///     of the UI render off it.
///   * Subgame resolution requires the SolverRunner — passed in from the
///     parent so we can share its in-flight state and cancel hooks.
///   * The screen owns nothing the parent screen needs to know about; on
///     "Quit", the parent flips its mode flag and the trainer state goes
///     away.
library;

import 'package:flutter/material.dart';

import '../scenario.dart';
import '../scenario_library.dart';
import '../solver_runner.dart';
import 'trainer_session.dart';
// `solver_runner.dart` exports `SolveDepth` — re-imported via the existing
// import above; the type appears in our resolve callback signature below.
import 'widgets/trainer_action_panel.dart';
import 'widgets/trainer_decision_summary.dart';
import 'widgets/trainer_history_strip.dart';
import 'widgets/trainer_range_viewer.dart';
import 'widgets/trainer_table.dart';

class TrainerScreen extends StatefulWidget {
  /// The scenario the user is currently training against. We hold the
  /// initial reference; if a subgame resolve swaps it, the new scenario is
  /// stored inside the TrainerSession.
  final Scenario initialScenario;
  /// Solver runner for chance_pending → subgame expansions.
  final SolverRunner solver;
  /// Library so successfully-resolved subgames get added to the left rail.
  final ScenarioLibrary library;
  /// "Quit trainer" callback — the parent flips back to Inspect mode.
  final VoidCallback onQuit;

  const TrainerScreen({
    super.key,
    required this.initialScenario,
    required this.solver,
    required this.library,
    required this.onQuit,
  });

  @override
  State<TrainerScreen> createState() => _TrainerScreenState();
}

class _TrainerScreenState extends State<TrainerScreen> {
  TrainerSession? _session;
  String _seat = 'oop'; // default; user picks at setup

  @override
  void initState() {
    super.initState();
    // Defer the setup dialog to post-build so we have a valid context.
    WidgetsBinding.instance.addPostFrameCallback((_) => _showSetupDialog());
  }

  @override
  void dispose() {
    _session?.dispose();
    super.dispose();
  }

  Future<void> _showSetupDialog() async {
    final seat = await showDialog<String>(
      context: context,
      barrierDismissible: false,
      builder: (_) => _SetupDialog(initialSeat: _seat),
    );
    if (!mounted) return;
    if (seat == null) {
      // User cancelled before picking — bail back to Inspect.
      widget.onQuit();
      return;
    }
    setState(() {
      _seat = seat;
      _session = TrainerSession.deal(
        scenario: widget.initialScenario,
        userSeat: seat,
      );
    });
    // Keep listening so phase changes redraw the panel.
    _session?.addListener(_onSessionChanged);
  }

  void _onSessionChanged() {
    if (!mounted) return;
    setState(() {});
  }

  /// Replay rewinds within the active scenario — could be the root or a
  /// resolved subgame. Keeps both hole cards and the scenario pointer where
  /// they are; just rewinds the action history. This matches the user
  /// expectation of "I want to try the same situation a different way."
  void _replayHand() {
    _session?.replayHand();
  }

  /// New hand always rewinds back to the ROOT scenario the trainer was
  /// started with. Without this reset, after a subgame swap "New hand"
  /// would deal from the conditional ranges of the subgame — surprising,
  /// since "new hand" suggests "fresh hand from preflop". A user who
  /// specifically wants subgame-conditional fresh hands should re-enter
  /// trainer mode after loading the subgame as the active scenario.
  void _newHand() {
    final s = _session;
    if (s == null) return;
    s.removeListener(_onSessionChanged);
    final fresh = TrainerSession.newHand(
      scenario: widget.initialScenario,
      userSeat: s.userSeat,
    );
    setState(() => _session = fresh);
    fresh?.addListener(_onSessionChanged);
  }

  Future<void> _onResolveSubgame(
      int pickedCardByte, SolveDepth depth) async {
    final s = _session;
    if (s == null) return;
    s.beginSubgameResolve();
    try {
      final outPath = await widget.solver.expandChancePending(
        scenario: s.scenario,
        chancePending: s.resolvingFrom!,
        pickedCardByte: pickedCardByte,
        depth: depth,
      );
      if (outPath == null) {
        s.failSubgameResolve(
            widget.solver.lastError ?? 'subgame solve failed');
        return;
      }
      // Load the new scenario and register it in the library so the user
      // can return to it later if they quit and re-open.
      final entry = widget.library.addEntry(outPath);
      final loaded = await Scenario.loadFromFile(entry.path,
          label: entry.label);
      // Read the dealt card label from the binding-provided readable form.
      final dealt = _cardLabel(pickedCardByte);
      final ok = s.completeSubgameSwap(loaded, dealt);
      if (!ok && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            backgroundColor: const Color(0xFF7A2A2A),
            content: Text(
              s.statusMessage ?? 'subgame swap failed',
            ),
          ),
        );
      }
    } catch (e) {
      s.failSubgameResolve('error: $e');
    }
  }

  String _cardLabel(int byte) {
    // Two-char "Tc" / "As" form; same as engine binding readable.
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    final r = (byte >> 2) & 0xF;
    final st = byte & 0x3;
    if (r >= ranks.length || st >= suits.length) return '??';
    return '${ranks[r]}${suits[st]}';
  }

  @override
  Widget build(BuildContext context) {
    final s = _session;
    if (s == null) {
      // Setup dialog is up; render an empty placeholder so the layout is
      // stable while waiting.
      return const SizedBox.expand();
    }
    final isShowdown = s.phase == TrainerPhase.terminal &&
        s.history.isNotEmpty &&
        !s.history.last.label.toLowerCase().contains('fold');

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _topBar(s),
        const SizedBox(height: 10),
        if (s.scenario.parent != null) _subgameBanner(s.scenario),
        if (s.scenario.parent != null) const SizedBox(height: 10),
        Expanded(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      TrainerTable(session: s, revealOpp: isShowdown),
                      const SizedBox(height: 12),
                      TrainerActionPanel(
                        session: s,
                        onResolveSubgame: _onResolveSubgame,
                      ),
                      if (s.phase == TrainerPhase.terminal ||
                          s.phase == TrainerPhase.abandoned) ...[
                        const SizedBox(height: 12),
                        TrainerDecisionSummary(
                          session: s,
                          onReplay: _replayHand,
                          onNewHand: _newHand,
                          onQuit: widget.onQuit,
                        ),
                      ],
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 12),
              SizedBox(
                width: 240,
                child: TrainerHistoryStrip(session: s),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _topBar(TrainerSession s) {
    final isUserTurn = s.phase == TrainerPhase.awaitingUser;
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: const Color(0xFF1B2D40),
            border: Border.all(color: const Color(0xFF6FB3DC), width: 1),
            borderRadius: BorderRadius.circular(6),
          ),
          child: Row(
            children: [
              const Icon(Icons.school,
                  size: 16, color: Color(0xFF6FB3DC)),
              const SizedBox(width: 6),
              Text(
                'TRAINER · ${s.userSeat.toUpperCase()} '
                '(${s.userSeat == 'oop' ? 'BB' : 'SB'})',
                style: const TextStyle(
                  color: Color(0xFFEAE6D9),
                  fontFamily: 'monospace',
                  fontSize: 12,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 10),
        if (isUserTurn)
          const Text(
            'your turn',
            style: TextStyle(
              color: Color(0xFF6FDC84),
              fontFamily: 'monospace',
              fontSize: 12,
              fontWeight: FontWeight.w800,
            ),
          ),
        const Spacer(),
        OutlinedButton.icon(
          onPressed: () => TrainerRangeViewer.show(context, s.scenario),
          icon: const Icon(Icons.grid_view, size: 16),
          label: const Text('Show ranges'),
        ),
        const SizedBox(width: 6),
        // One-step undo of the last decision. Disabled when there's nothing
        // to undo, or when the most recent decision was made in a different
        // (parent) scenario than the active one — see TrainerSession.canUndo.
        OutlinedButton.icon(
          onPressed: s.canUndo ? s.undoLastDecision : null,
          icon: const Icon(Icons.undo, size: 16),
          label: const Text('Back'),
        ),
        const SizedBox(width: 6),
        OutlinedButton.icon(
          onPressed: _replayHand,
          icon: const Icon(Icons.replay, size: 16),
          label: const Text('Replay'),
        ),
        const SizedBox(width: 6),
        OutlinedButton.icon(
          onPressed: _newHand,
          icon: const Icon(Icons.casino, size: 16),
          label: const Text('New hand'),
        ),
        const SizedBox(width: 6),
        OutlinedButton.icon(
          onPressed: widget.onQuit,
          icon: const Icon(Icons.close, size: 16),
          label: const Text('Quit'),
        ),
      ],
    );
  }

  Widget _subgameBanner(Scenario sc) {
    final p = sc.parent!;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0x33D4B43F),
        border: Border.all(color: const Color(0xFFD4B43F), width: 1),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        children: [
          const Icon(Icons.alt_route,
              size: 18, color: Color(0xFFD4B43F)),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              'NOT AT ROOT — playing the subgame from '
              '${p.parentLine.join(" › ")} → ${p.pickedCard}. '
              'The "Show ranges" button shows the conditional ranges.',
              style: const TextStyle(
                color: Color(0xFFEAE6D9),
                fontSize: 12,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _SetupDialog extends StatefulWidget {
  final String initialSeat;
  const _SetupDialog({required this.initialSeat});

  @override
  State<_SetupDialog> createState() => _SetupDialogState();
}

class _SetupDialogState extends State<_SetupDialog> {
  late String _seat;

  @override
  void initState() {
    super.initState();
    _seat = widget.initialSeat;
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: const Color(0xFF24282B),
      title: const Text(
        'Start trainer',
        style: TextStyle(color: Color(0xFFEAE6D9)),
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Pick the seat you want to play. Hole cards are dealt at '
            'random from that seat\'s range; you can re-deal anytime.',
            style: TextStyle(color: Color(0xCCEAE6D9), fontSize: 13),
          ),
          const SizedBox(height: 16),
          // Seat picker — segmented control feel.
          Row(
            children: [
              Expanded(
                child: _SeatTile(
                  label: 'BB / OOP',
                  hint: 'acts first postflop',
                  selected: _seat == 'oop',
                  onTap: () => setState(() => _seat = 'oop'),
                  accent: const Color(0xFF6FB3DC),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _SeatTile(
                  label: 'SB / IP',
                  hint: 'acts last postflop',
                  selected: _seat == 'ip',
                  onTap: () => setState(() => _seat = 'ip'),
                  accent: const Color(0xFFDCBE6F),
                ),
              ),
            ],
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).maybePop(),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: () => Navigator.of(context).pop(_seat),
          child: const Text('Deal & start'),
        ),
      ],
    );
  }
}

class _SeatTile extends StatelessWidget {
  final String label;
  final String hint;
  final bool selected;
  final VoidCallback onTap;
  final Color accent;

  const _SeatTile({
    required this.label,
    required this.hint,
    required this.selected,
    required this.onTap,
    required this.accent,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
        decoration: BoxDecoration(
          color: selected
              ? accent.withValues(alpha: 0.22)
              : const Color(0xFF1B1E20),
          border: Border.all(
            color: selected ? accent : const Color(0xFF45525A),
            width: selected ? 2 : 1,
          ),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: TextStyle(
                color: accent,
                fontWeight: FontWeight.w800,
                fontSize: 14,
                letterSpacing: 1.0,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              hint,
              style: const TextStyle(
                color: Color(0xCCEAE6D9),
                fontSize: 12,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
