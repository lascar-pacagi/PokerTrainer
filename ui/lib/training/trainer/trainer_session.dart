/// Stateful logic for the interactive trainer mode.
///
/// One TrainerSession plays out one hand against the solver:
///   1. Hole cards are sampled at construction from the root weights of the
///      active scenario (random weighted by combo presence).
///   2. The session walks the action tree. At your nodes it pauses for input;
///      at opponent nodes it samples an action from `strategy[h_opp][·]`.
///   3. After your click, it records `(your action, your EV, best action,
///      best EV)` and exposes that for the feedback panel.
///   4. At a chance_pending the session pauses and asks the caller to resolve
///      the subgame; on success it swaps `_scenario` to the subgame, re-
///      indexes both hands into the new combo lists, and continues.
///
/// Pure logic — no Flutter widgets. The screen layer wraps it with UI.
library;

import 'dart:math' as math;

import 'package:flutter/foundation.dart';

import '../scenario.dart';

/// Where we are in the per-decision dance.
enum TrainerPhase {
  /// Waiting for the user to click an action.
  awaitingUser,

  /// User clicked; the feedback panel is showing GTO mix + EVs. Caller must
  /// invoke [continueAfterFeedback] to advance.
  revealingFeedback,

  /// Hand has reached a terminal node (fold or showdown). Decision summary
  /// is the natural next view; replay / new-hand / quit are the available
  /// controls.
  terminal,

  /// Reached a chance_pending. The user must pick a card and trigger a
  /// subgame solve via [resolveSubgame] before the trainer can continue.
  chancePending,

  /// Subgame solve in flight (set by [resolveSubgame] while it awaits the
  /// solver). UI should render a "Solving…" indicator and disable input.
  resolvingSubgame,

  /// User chose to end the hand at a chance_pending without resolving the
  /// subgame. Decision summary is shown the same way as for [terminal];
  /// the difference is purely UI labelling ("Hand abandoned" vs. "Hand
  /// over").
  abandoned,
}

/// One row of the end-of-hand summary table — also driven into the running
/// log so the feedback panel can render the most recent entry.
class DecisionRecord {
  /// Scenario label and node id — useful when the summary spans a subgame.
  final String scenarioLabel;
  /// Source path of the scenario when this decision was recorded. Used by
  /// the back/undo button to refuse undoing across a subgame boundary
  /// (which would also need to revert the scenario swap, which is complex).
  final String scenarioSourcePath;
  final String nodeId;
  final List<String> line;

  /// Available actions at this node in display order.
  final List<String> actions;
  /// Index of the user's pick within [actions].
  final int yourActionIdx;
  /// EV of the user's chosen action, in chips.
  final double yourEv;
  /// Index of the EV-maximising action.
  final int bestActionIdx;
  /// EV of the best action.
  final double bestEv;
  /// Per-action GTO mix for the user's hand at this node (sums to 1).
  final List<double> mix;
  /// Per-action EV for the user's hand at this node.
  final List<double> evs;
  /// History entry-count at the moment this decision was recorded (BEFORE
  /// the user's action got appended). Lets us trim the history list back
  /// to this snapshot on undo so the right-rail strip stays consistent.
  final int historyLenAtDecision;

  const DecisionRecord({
    required this.scenarioLabel,
    required this.scenarioSourcePath,
    required this.nodeId,
    required this.line,
    required this.actions,
    required this.yourActionIdx,
    required this.yourEv,
    required this.bestActionIdx,
    required this.bestEv,
    required this.mix,
    required this.evs,
    required this.historyLenAtDecision,
  });

  String get yourAction => actions[yourActionIdx];
  String get bestAction => actions[bestActionIdx];
  double get evGap => bestEv - yourEv;   // ≥ 0 by construction
}

/// One action that happened, regardless of who took it. The history strip
/// renders these in order.
class TrainerHistoryEntry {
  /// 'user' | 'opp' | 'chance' | 'street'.
  final String kind;
  /// 'oop' or 'ip' for user/opp actions; null for chance/street markers.
  final String? player;
  /// Display label — action label, dealt card "Tc", or street banner "FLOP".
  final String label;

  const TrainerHistoryEntry({
    required this.kind,
    this.player,
    required this.label,
  });
}

class TrainerSession extends ChangeNotifier {
  Scenario _scenario;
  /// 'oop' or 'ip' — which seat the user is playing.
  final String userSeat;

  /// Index into `_scenario.combosFor(userSeat)`.
  int _userHandIdx;
  /// Combo string for the user's hand. Held separately so that on subgame
  /// swap we can re-find the index in the new scenario's combo list (the
  /// numeric index is not portable across scenarios).
  final String _userHandStr;

  int _oppHandIdx;
  final String _oppHandStr;

  String _currentNodeId = '';
  TrainerPhase _phase = TrainerPhase.awaitingUser;

  final math.Random _rng;

  /// Completed decisions (user-only). Drives the end-of-hand summary.
  final List<DecisionRecord> _log = [];

  /// All actions in chronological order — used by the history strip.
  final List<TrainerHistoryEntry> _history = [];

  /// Set when [phase] is [TrainerPhase.revealingFeedback]. Holds the
  /// just-recorded decision so the feedback panel can render it without a
  /// separate pointer.
  DecisionRecord? _pendingFeedback;

  /// Set when [phase] is [TrainerPhase.resolvingSubgame] — the chance_pending
  /// the user wants to resolve. After resolveSubgame() returns, we've moved
  /// into the new scenario at its root (currentNodeId == "").
  ScenarioNode? _resolvingFrom;

  /// Optional progress / error message shown by the UI while resolving.
  String? _statusMessage;

  TrainerSession._({
    required Scenario scenario,
    required this.userSeat,
    required int userHandIdx,
    required String userHandStr,
    required int oppHandIdx,
    required String oppHandStr,
    required math.Random rng,
  })  : _scenario = scenario,
        _userHandIdx = userHandIdx,
        _userHandStr = userHandStr,
        _oppHandIdx = oppHandIdx,
        _oppHandStr = oppHandStr,
        _rng = rng {
    _bootstrap();
  }

  /// Build a session by sampling both hole cards from the root weights.
  /// Returns null if either side's range has zero total weight (degenerate
  /// scenario; shouldn't happen with a well-formed solve).
  static TrainerSession? deal({
    required Scenario scenario,
    required String userSeat,
    int? rngSeed,
  }) {
    assert(userSeat == 'oop' || userSeat == 'ip', 'userSeat must be oop|ip');
    final oppSeat = userSeat == 'oop' ? 'ip' : 'oop';

    final root = scenario.root;
    final userWeights =
        root.player == userSeat ? root.weights : root.weightsOpp;
    final oppWeights =
        root.player == oppSeat ? root.weights : root.weightsOpp;
    final userCombos = scenario.combosFor(userSeat);
    final oppCombos = scenario.combosFor(oppSeat);

    if (userCombos.isEmpty || oppCombos.isEmpty) return null;
    if (userWeights.length != userCombos.length ||
        oppWeights.length != oppCombos.length) {
      // Schema mismatch — refuse rather than silently sample garbage.
      return null;
    }

    final rng = math.Random(rngSeed);
    final userIdx = _sampleIndex(userWeights, rng);
    if (userIdx < 0) return null;

    // Reject combos that share a card with the user's hand. That's the
    // correct conditional opp range given we've already "dealt" the user's
    // cards. Without this filter, opp could be dealt a hand sharing a card
    // with the user's, which is impossible.
    final userCardA = userCombos[userIdx].substring(0, 2);
    final userCardB = userCombos[userIdx].substring(2, 4);
    final oppFilteredWeights = List<double>.generate(
      oppWeights.length,
      (i) {
        final c = oppCombos[i];
        final shares = c.substring(0, 2) == userCardA ||
            c.substring(0, 2) == userCardB ||
            c.substring(2, 4) == userCardA ||
            c.substring(2, 4) == userCardB;
        return shares ? 0.0 : oppWeights[i];
      },
    );
    final oppIdx = _sampleIndex(oppFilteredWeights, rng);
    if (oppIdx < 0) return null;

    return TrainerSession._(
      scenario: scenario,
      userSeat: userSeat,
      userHandIdx: userIdx,
      userHandStr: userCombos[userIdx],
      oppHandIdx: oppIdx,
      oppHandStr: oppCombos[oppIdx],
      rng: rng,
    );
  }

  // ── Public read-only state ─────────────────────────────────────────────

  Scenario get scenario => _scenario;
  String get oppSeat => userSeat == 'oop' ? 'ip' : 'oop';
  int get userHandIdx => _userHandIdx;
  String get userHandStr => _userHandStr;
  /// Opponent's hand string. Only call this AFTER showdown — leaking it
  /// during play would compromise the trainer.
  String get oppHandStrRevealed => _oppHandStr;
  String get currentNodeId => _currentNodeId;
  ScenarioNode get currentNode => _scenario.nodeById(_currentNodeId)!;
  TrainerPhase get phase => _phase;
  List<DecisionRecord> get log => List.unmodifiable(_log);
  List<TrainerHistoryEntry> get history => List.unmodifiable(_history);
  DecisionRecord? get pendingFeedback => _pendingFeedback;
  ScenarioNode? get resolvingFrom => _resolvingFrom;
  String? get statusMessage => _statusMessage;

  // ── State transitions ──────────────────────────────────────────────────

  /// Initial transition: at construction, advance opp/chance steps until we
  /// land somewhere the UI cares about (user node / terminal / chance_pending).
  void _bootstrap() {
    _appendStreetBannerIfNew(_scenario.root);
    _settleAtBoundary();
  }

  /// User clicks an action at a user node. Records the decision into [log],
  /// transitions to [revealingFeedback].
  void chooseAction(int actionIdx) {
    if (_phase != TrainerPhase.awaitingUser) return;
    final n = currentNode;
    assert(n.player == userSeat,
        'chooseAction called when actor != userSeat');
    assert(actionIdx >= 0 && actionIdx < n.actions.length);

    final evs = _safeEvRow(n, _userHandIdx);
    final mix = _safeStratRow(n, _userHandIdx);
    final bestIdx = _argmax(evs);

    final record = DecisionRecord(
      scenarioLabel: _scenario.label,
      scenarioSourcePath: _scenario.sourcePath,
      nodeId: n.id,
      line: List.unmodifiable(n.line),
      actions: List.unmodifiable(n.actions),
      yourActionIdx: actionIdx,
      yourEv: evs[actionIdx],
      bestActionIdx: bestIdx,
      bestEv: evs[bestIdx],
      mix: List.unmodifiable(mix),
      evs: List.unmodifiable(evs),
      // Snapshot history length BEFORE we append anything for this decision.
      // continueAfterFeedback() appends the user's action label later.
      historyLenAtDecision: _history.length,
    );
    _log.add(record);
    _pendingFeedback = record;
    _phase = TrainerPhase.revealingFeedback;
    notifyListeners();
  }

  /// User dismisses the feedback panel. Descend into the chosen child and
  /// auto-resolve any opponent / chance steps until we reach the next user
  /// boundary.
  void continueAfterFeedback() {
    if (_phase != TrainerPhase.revealingFeedback || _pendingFeedback == null) {
      return;
    }
    final r = _pendingFeedback!;
    final n = currentNode;
    final edge = n.children[r.yourActionIdx];
    _history.add(TrainerHistoryEntry(
      kind: 'user',
      player: userSeat,
      label: edge.action,
    ));
    _currentNodeId = edge.nodeId;
    _pendingFeedback = null;
    _appendStreetBannerIfNew(currentNode);
    _settleAtBoundary();
  }

  /// Reset to the root of the current scenario, keeping both hole cards.
  /// Useful for "replay this hand and pick a different action."
  void replayHand() {
    _log.clear();
    _history.clear();
    _pendingFeedback = null;
    _resolvingFrom = null;
    _statusMessage = null;
    _currentNodeId = '';
    _appendStreetBannerIfNew(_scenario.root);
    _phase = TrainerPhase.awaitingUser;
    _settleAtBoundary();
  }

  /// True iff [undoLastDecision] would do something. The "Back" button uses
  /// this to enable/disable. Two preconditions: there's a decision to undo,
  /// AND that decision was made in the currently active scenario (we don't
  /// rewind across subgame swaps because the parent scenario isn't held in
  /// memory by the session anymore).
  bool get canUndo {
    if (_log.isEmpty) return false;
    final last = _log.last;
    if (last.scenarioSourcePath != _scenario.sourcePath) return false;
    // Don't allow undo while a subgame solve is in flight.
    if (_phase == TrainerPhase.resolvingSubgame) return false;
    return true;
  }

  /// One-step back: pop the most recent decision, trim history back to the
  /// snapshot taken at decision time, restore [currentNodeId] to the node
  /// where the decision was made, return to [awaitingUser].
  ///
  /// Refuses across subgame boundaries (see [canUndo]). The previous
  /// scenario isn't held by the session — only the active one is — so
  /// undoing a pre-swap decision would leak the swap. If you want to go
  /// back to before the swap, use Replay (which restores the hand to the
  /// active scenario root) or New hand (which goes all the way to the
  /// trainer's initial scenario).
  void undoLastDecision() {
    if (!canUndo) return;
    final last = _log.removeLast();
    // Trim history back to where it was right before the decision was made.
    if (last.historyLenAtDecision < _history.length) {
      _history.removeRange(last.historyLenAtDecision, _history.length);
    }
    _currentNodeId = last.nodeId;
    _pendingFeedback = null;
    _resolvingFrom = null;
    _statusMessage = null;
    _phase = TrainerPhase.awaitingUser;
    notifyListeners();
  }

  /// User wants to end the hand at a chance_pending without paying for a
  /// subgame solve. Goes to [abandoned] phase; the rest of the UI uses
  /// the same machinery as [terminal] (decision summary, replay/new hand).
  void abandonHand() {
    if (_phase != TrainerPhase.chancePending) return;
    _phase = TrainerPhase.abandoned;
    _statusMessage = null;
    _resolvingFrom = null;
    notifyListeners();
  }

  /// Re-deal both hands at random and restart. Returns null on degenerate
  /// scenario (same condition as deal()).
  static TrainerSession? newHand({
    required Scenario scenario,
    required String userSeat,
    int? rngSeed,
  }) =>
      deal(scenario: scenario, userSeat: userSeat, rngSeed: rngSeed);

  /// Caller-driven subgame resolution. Phase must be [chancePending].
  ///
  /// The two-step shape (mark-as-resolving → caller does async work →
  /// completeSubgameSwap) keeps async I/O out of this class. The screen
  /// holds the SolverRunner, runs the solve, then hands us the loaded
  /// subgame Scenario via [completeSubgameSwap]. We can also be told the
  /// solve failed via [failSubgameResolve].
  void beginSubgameResolve() {
    if (_phase != TrainerPhase.chancePending) return;
    _resolvingFrom = currentNode;
    _phase = TrainerPhase.resolvingSubgame;
    _statusMessage = 'Solving subgame…';
    notifyListeners();
  }

  /// Successful subgame solve. We now switch the active scenario, re-find
  /// both hole-card indices in the new combo lists, and resume from the
  /// subgame root. Returns false (and stays in error state) if the user's
  /// or opponent's hand isn't in the new scenario's combos (which can only
  /// happen if the picked card overlaps a hole card — caller should have
  /// filtered that already).
  bool completeSubgameSwap(Scenario subgame, String dealtCardLabel) {
    final newUserCombos = subgame.combosFor(userSeat);
    final newOppCombos = subgame.combosFor(oppSeat);
    final newUserIdx = newUserCombos.indexOf(_userHandStr);
    final newOppIdx = newOppCombos.indexOf(_oppHandStr);
    if (newUserIdx < 0 || newOppIdx < 0) {
      _statusMessage = 'Subgame is missing the dealt hand — '
          'the picked card overlapped one of the hole cards.';
      _phase = TrainerPhase.chancePending;
      _resolvingFrom = null;
      notifyListeners();
      return false;
    }
    _scenario = subgame;
    _userHandIdx = newUserIdx;
    _oppHandIdx = newOppIdx;
    _currentNodeId = '';
    _phase = TrainerPhase.awaitingUser;
    _resolvingFrom = null;
    _statusMessage = null;
    _history.add(TrainerHistoryEntry(
      kind: 'chance',
      label: '— $dealtCardLabel dealt; subgame solved —',
    ));
    _appendStreetBannerIfNew(_scenario.root);
    _settleAtBoundary();
    return true;
  }

  /// Subgame solve failed (or user cancelled). Return to the chance_pending
  /// node so the user can try again or quit.
  void failSubgameResolve(String message) {
    _phase = TrainerPhase.chancePending;
    _resolvingFrom = null;
    _statusMessage = message;
    notifyListeners();
  }

  // ── Internal: advance until the UI needs to do something ───────────────

  /// Loop: while the current node is opp/chance/terminal-needing-no-pause,
  /// resolve it and advance. Stop at user nodes, terminals, chance_pendings.
  void _settleAtBoundary() {
    while (true) {
      final n = currentNode;
      if (n.isTerminal) {
        _phase = TrainerPhase.terminal;
        notifyListeners();
        return;
      }
      if (n.isChancePending) {
        _phase = TrainerPhase.chancePending;
        notifyListeners();
        return;
      }
      if (n.isAction) {
        if (n.player == userSeat) {
          _phase = TrainerPhase.awaitingUser;
          notifyListeners();
          return;
        }
        // Opponent acts. Sample from the opp's strategy for their hand.
        _resolveOpponentAction(n);
        continue;
      }
      if (n.isChance) {
        // Walked-through chance node (e.g., turn was sampled in the solve).
        // Sample one of the dealt cards (uniform — these are the cards the
        // solver actually evaluated; we don't have per-card weights here).
        _resolveChance(n);
        continue;
      }
      // Unknown kind — bail to avoid infinite loop.
      _phase = TrainerPhase.terminal;
      notifyListeners();
      return;
    }
  }

  void _resolveOpponentAction(ScenarioNode n) {
    final mix = _safeStratRow(n, _oppHandIdx);
    final actionIdx = _sampleIndex(mix, _rng);
    if (actionIdx < 0 || actionIdx >= n.children.length) {
      // Degenerate — fall back to first child to avoid crash. Shouldn't
      // happen for hands inside the opp's range.
      _currentNodeId = n.children.isEmpty ? n.id : n.children.first.nodeId;
      return;
    }
    final edge = n.children[actionIdx];
    _history.add(TrainerHistoryEntry(
      kind: 'opp',
      player: oppSeat,
      label: edge.action,
    ));
    _currentNodeId = edge.nodeId;
    _appendStreetBannerIfNew(currentNode);
  }

  void _resolveChance(ScenarioNode n) {
    if (n.children.isEmpty) {
      // Defensive: chance with no children → treat as terminal-ish.
      return;
    }
    final pick = n.children[_rng.nextInt(n.children.length)];
    _history.add(TrainerHistoryEntry(
      kind: 'chance',
      label: pick.action, // action label here is the dealt card
    ));
    _currentNodeId = pick.nodeId;
    _appendStreetBannerIfNew(currentNode);
  }

  /// Insert a banner ("FLOP" / "TURN" / "RIVER") when we transition to a
  /// node whose board length differs from the previous entry's. Cheap
  /// rendering hint — the history strip uses it to draw section dividers.
  void _appendStreetBannerIfNew(ScenarioNode n) {
    final boardLen = n.board.length;
    String? banner;
    if (boardLen == 3 && _history.where((h) => h.label == 'FLOP').isEmpty) {
      banner = 'FLOP';
    } else if (boardLen == 4 &&
        _history.where((h) => h.label == 'TURN').isEmpty) {
      banner = 'TURN';
    } else if (boardLen == 5 &&
        _history.where((h) => h.label == 'RIVER').isEmpty) {
      banner = 'RIVER';
    }
    if (banner != null) {
      _history.add(TrainerHistoryEntry(kind: 'street', label: banner));
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────────

  /// Argmax — first index attaining the maximum value. Defensive against
  /// empty input (returns 0).
  static int _argmax(List<double> v) {
    if (v.isEmpty) return 0;
    var best = 0;
    var bestVal = v[0];
    for (var i = 1; i < v.length; i++) {
      if (v[i] > bestVal) {
        bestVal = v[i];
        best = i;
      }
    }
    return best;
  }

  /// Sample a non-negative-weighted index. Returns -1 if total weight is 0.
  static int _sampleIndex(List<double> weights, math.Random rng) {
    var total = 0.0;
    for (final w in weights) {
      if (w > 0) total += w;
    }
    if (total <= 0) return -1;
    var pick = rng.nextDouble() * total;
    for (var i = 0; i < weights.length; i++) {
      final w = weights[i];
      if (w <= 0) continue;
      pick -= w;
      if (pick <= 0) return i;
    }
    // Floating-point drift: fall through to the last positive index.
    for (var i = weights.length - 1; i >= 0; i--) {
      if (weights[i] > 0) return i;
    }
    return -1;
  }

  /// Strategy row for hand `h` at node `n`, with the same length as actions.
  /// Returns a uniform-over-actions row when the stored row is missing or
  /// malformed (defensive for malformed dumps).
  static List<double> _safeStratRow(ScenarioNode n, int h) {
    if (h < 0 || h >= n.strategy.length) {
      return List<double>.filled(n.actions.length, 1.0 / n.actions.length);
    }
    final row = n.strategy[h];
    if (row.length != n.actions.length) {
      return List<double>.filled(n.actions.length, 1.0 / n.actions.length);
    }
    return row;
  }

  static List<double> _safeEvRow(ScenarioNode n, int h) {
    if (h < 0 || h >= n.ev.length) {
      return List<double>.filled(n.actions.length, 0.0);
    }
    final row = n.ev[h];
    if (row.length != n.actions.length) {
      return List<double>.filled(n.actions.length, 0.0);
    }
    return row;
  }
}
