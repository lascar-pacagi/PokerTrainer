/// Stateful controller wrapping one EnvHandle.
///
/// Owns the running hand: deals new ones, advances state on user actions,
/// emits ChangeNotifier ticks so widgets rebuild. Per-seat agent assignment
/// (Human / Model) lives here too — when control passes to a Model seat
/// after a step, GameSession auto-steps until the next Human turn (or the
/// hand ends).
library;

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';

import '../ffi/actions.dart';
import '../ffi/engine.dart';
import '../ffi/equity.dart';

/// Rich end-of-hand summary surfaced to the UI.
class TerminalSummary {
  final Terminal reason;
  /// null on a tied/chopped hand.
  final Player? winner;
  /// Magnitude of the winner's pot in BB. Zero on a chop.
  final double winAmountBb;
  final double sbDeltaBb;
  final double bbDeltaBb;
  /// Hand-category labels for showdown ("Two Pair", "Flush", ...). Null
  /// when the hand ended on fold or the categories aren't reachable.
  final String? sbHandCategory;
  final String? bbHandCategory;

  const TerminalSummary({
    required this.reason,
    required this.winner,
    required this.winAmountBb,
    required this.sbDeltaBb,
    required this.bbDeltaBb,
    required this.sbHandCategory,
    required this.bbHandCategory,
  });
}

/// Who controls a seat.
sealed class SeatAgent {
  const SeatAgent();
}

class HumanAgent extends SeatAgent {
  const HumanAgent();
}

class ModelAgent extends SeatAgent {
  final ModelHandle model;
  final String label;
  const ModelAgent(this.model, {this.label = 'Model'});
}

/// What the model thinks of the current decision. `qValues[i]` aligns with
/// `Observation.legal[i]`. Null when no model is loaded for the seat to act.
class StrategyView {
  final Float32List qValues;
  final int argmaxLegalIdx;
  const StrategyView({required this.qValues, required this.argmaxLegalIdx});
}

class GameSession extends ChangeNotifier {
  final PokerEngine engine;
  final EnvHandle env;

  /// Per-seat agent. Defaults to Human / Human.
  final Map<Player, SeatAgent> _agents = {
    Player.sb: const HumanAgent(),
    Player.bb: const HumanAgent(),
  };

  /// Reproducibility seed for the current hand (last `resetSeed` value, or
  /// null if dealt via internal RNG).
  int? _currentSeed;

  /// Latest engine state — refreshed after every mutation.
  late TableState _tableState;
  Observation?   _observation;          // null iff terminal
  List<HistoryEntry> _history = const [];

  /// Cached strategy for the seat-to-act, if a model is loaded for them.
  StrategyView? _strategy;

  /// Cached hand equity for the current (street, board, holes). Recomputed
  /// when those change. Null when one or both holes are NO_CARD or the
  /// hand is terminal.
  EquityResult? _equity;
  String? _equityKey;

  /// Counter the UI can show ("Hand #N"). Bumped on every fresh deal.
  int _handCount = 1;

  GameSession(this.engine) : env = EnvHandle.create(engine) {
    _refresh();
  }

  // ─── public read API ────────────────────────────────────────────────────

  TableState get tableState => _tableState;
  Observation? get observation => _observation;
  List<HistoryEntry> get history => _history;
  StrategyView? get strategy => _strategy;
  EquityResult? get equity => _equity;
  int get handCount => _handCount;
  int? get currentSeed => _currentSeed;
  SeatAgent agentFor(Player p) => _agents[p]!;
  bool get isTerminal => _tableState.isTerminal;

  // ─── deal / reset ───────────────────────────────────────────────────────

  /// Deal a fresh hand using the Env's internal RNG. The seed is unknown —
  /// the UI shows "—" instead of a hex value.
  void dealRandom() {
    env.reset();
    _currentSeed = null;
    _handCount += 1;
    _equityKey = null;
    _refresh();
    _autoStepIfModel();
  }

  /// Deal a reproducible hand from a specific seed.
  void dealWithSeed(int seed) {
    env.resetSeed(seed);
    _currentSeed = seed;
    _handCount += 1;
    _equityKey = null;
    _refresh();
    _autoStepIfModel();
  }

  /// Apply the i-th legal action of the current observation. Throws if
  /// terminal or if `idx` is out of range.
  void applyLegal(int idx) {
    if (_observation == null) {
      throw StateError('applyLegal on terminal hand');
    }
    if (idx < 0 || idx >= _observation!.legal.length) {
      throw RangeError.index(idx, _observation!.legal);
    }
    env.applyLegal(idx);
    _refresh();
    _autoStepIfModel();
  }

  // ─── agent assignment ───────────────────────────────────────────────────

  /// Reassign a seat. If the new seat-to-act is a model, fires off
  /// auto-stepping. Strategy cache is invalidated.
  void setAgent(Player p, SeatAgent agent) {
    _agents[p] = agent;
    _strategy = null;
    notifyListeners();
    _autoStepIfModel();
  }

  // ─── internals ──────────────────────────────────────────────────────────

  void _refresh() {
    _tableState = env.snapshot();
    _history    = env.readHistory();
    _observation = _tableState.isTerminal ? null : env.observe();
    _strategy = _computeStrategyIfPossible();
    _equity   = _computeEquityIfBothHolesKnown();
    notifyListeners();
  }

  /// Rich end-of-hand summary. Null while the hand is still in progress.
  /// For showdown, runs the 7-card evaluator on both seats so the UI can
  /// say "BB wins 6.5 bb at showdown — Two Pair".
  TerminalSummary? terminalSummary() {
    if (!_tableState.isTerminal) return null;
    final sbDelta = _tableState.payoffChips[0] / TableState.chipsPerBb;
    final bbDelta = _tableState.payoffChips[1] / TableState.chipsPerBb;
    final winner = sbDelta > 0
        ? Player.sb
        : (bbDelta > 0 ? Player.bb : null);

    if (_tableState.terminalReason == Terminal.fold) {
      return TerminalSummary(
        reason: Terminal.fold,
        winner: winner,
        winAmountBb: winner == null ? 0 : (winner == Player.sb ? sbDelta : bbDelta),
        sbDeltaBb: sbDelta,
        bbDeltaBb: bbDelta,
        sbHandCategory: null,
        bbHandCategory: null,
      );
    }

    // Showdown: read the full board + both holes and run hand eval.
    final sb    = env.getHole(Player.sb);
    final bb    = env.getHole(Player.bb);
    final board = env.getBoard();
    String? sbCat, bbCat;
    if (sb.every((c) => c != EngineCard.noCard) &&
        bb.every((c) => c != EngineCard.noCard) &&
        board.every((c) => c != EngineCard.noCard)) {
      final ranks = _evalShowdown(sb, bb, board);
      sbCat = rankCategory(engine, ranks.sb);
      bbCat = rankCategory(engine, ranks.bb);
    }

    return TerminalSummary(
      reason: Terminal.showdown,
      winner: winner,
      winAmountBb: winner == null
          ? 0
          : (winner == Player.sb ? sbDelta : bbDelta),
      sbDeltaBb: sbDelta,
      bbDeltaBb: bbDelta,
      sbHandCategory: sbCat,
      bbHandCategory: bbCat,
    );
  }

  ({int sb, int bb}) _evalShowdown(
      List<int> sbHole, List<int> bbHole, List<int> board) {
    // Tiny scratch for two evaluate7 calls.
    final sbCards = <int>[...sbHole, ...board];
    final bbCards = <int>[...bbHole, ...board];
    int evalOne(List<int> cards) {
      final p = malloc<ffi.Uint8>(7);
      try {
        for (int i = 0; i < 7; i++) {
          p[i] = cards[i];
        }
        return engine.native.ptEval7(p);
      } finally {
        malloc.free(p);
      }
    }
    return (sb: evalOne(sbCards), bb: evalOne(bbCards));
  }

  EquityResult? _computeEquityIfBothHolesKnown() {
    // No equity for a finished hand — the result IS the equity.
    if (_tableState.isTerminal) {
      _equityKey = null;
      return null;
    }
    final sb = env.getHole(Player.sb);
    final bb = env.getHole(Player.bb);
    if (sb[0] == EngineCard.noCard || sb[1] == EngineCard.noCard) return null;
    if (bb[0] == EngineCard.noCard || bb[1] == EngineCard.noCard) return null;
    final board = env.getBoard();
    final nVis = _tableState.street.nVisibleBoard;

    // Cache key: equity is a pure function of (sb, bb, visible-board).
    // Within a street with the same dealt cards, no need to recompute.
    final key = '${sb[0]}.${sb[1]}|${bb[0]}.${bb[1]}|${board.take(nVis).join(",")}';
    if (key == _equityKey && _equity != null) return _equity;
    _equityKey = key;

    return computeEquity(
      eng: engine,
      sbHole: sb,
      bbHole: bb,
      visibleBoard: board,
      street: _tableState.street,
    );
  }

  StrategyView? _computeStrategyIfPossible() {
    final obs = _observation;
    if (obs == null || _tableState.toAct == null) return null;
    final agent = _agents[_tableState.toAct!];
    if (agent is! ModelAgent) return null;
    final r = agent.model.query(env);
    return StrategyView(qValues: r.qValues, argmaxLegalIdx: r.argmax);
  }

  /// If the seat-to-act is a Model, take its argmax action. Loops until
  /// either the hand ends or control returns to a Human. Cap iterations as
  /// a safety net against runaway loops.
  void _autoStepIfModel() {
    int safety = 64;
    while (!_tableState.isTerminal && safety-- > 0) {
      final p = _tableState.toAct;
      if (p == null) break;
      final agent = _agents[p];
      if (agent is! ModelAgent) break;
      final r = agent.model.query(env);
      env.applyLegal(r.argmax);
      _refresh();
    }
  }

  @override
  void dispose() {
    env.dispose();
    super.dispose();
  }
}
