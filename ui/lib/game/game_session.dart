/// Stateful controller wrapping one EnvHandle.
///
/// Owns the running hand: deals new ones, advances state on user actions,
/// emits ChangeNotifier ticks so widgets rebuild. Per-seat agent assignment
/// (Human / Model) lives here too — when control passes to a Model seat
/// after a step, GameSession auto-steps until the next Human turn (or the
/// hand ends).
library;

import 'package:flutter/foundation.dart';

import '../ffi/actions.dart';
import '../ffi/engine.dart';

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
    _refresh();
    _autoStepIfModel();
  }

  /// Deal a reproducible hand from a specific seed.
  void dealWithSeed(int seed) {
    env.resetSeed(seed);
    _currentSeed = seed;
    _handCount += 1;
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
    notifyListeners();
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
