/// End-to-end FFI smoke for the Dart side. Exercises every typed wrapper
/// in `lib/ffi/engine.dart` against the live libpokertrainer.so.
///
/// Run:
///   cd ui && dart run tool/ffi_smoke.dart
///
/// Or with an explicit .so path:
///   POKERTRAINER_SO=/path/to/libpokertrainer.so dart run tool/ffi_smoke.dart
library;

// CLI tool: print() is the entire UX.
// ignore_for_file: avoid_print

import 'package:pokertrainer_ui/ffi/actions.dart';
import 'package:pokertrainer_ui/ffi/engine.dart';

void main() {
  final eng = PokerEngine.init();
  print('engine: x=${eng.xDim} a=${eng.aDim} '
      'hist=${eng.histMax}x${eng.histFeat} '
      'streetSlots=${eng.streetSlots} '
      'modelAvailable=${eng.modelAvailable}');

  // Constants must match the live engine (v0.4 dims).
  assert(eng.xDim == 812, 'xDim drift: ${eng.xDim}');
  assert(eng.aDim == 11);
  assert(eng.histMax == 34);
  assert(eng.histFeat == 20);
  assert(eng.streetSlots.length == 4);
  assert(eng.streetSlots[0] == 10);

  final env = EnvHandle.create(eng, seed: 0x2026_04_30);
  try {
    // ─── 1. Fresh deal + snapshot ────────────────────────────────────────
    var snap = env.snapshot();
    assert(snap.potChips == 150, 'pot: ${snap.potChips}');
    assert(snap.toAct == Player.sb);
    assert(!snap.isTerminal);
    assert(snap.street == Street.preflop);
    print('deal: pot=${snap.potBb}bb toAct=${snap.toAct} '
        'sbStack=${snap.stackBb(Player.sb)}bb '
        'bbStack=${snap.stackBb(Player.bb)}bb');

    // ─── 2. Observe + legal sizings ──────────────────────────────────────
    var obs = env.observe();
    assert(obs.x.length == eng.xDim);
    assert(obs.legal.isNotEmpty);
    print('legal preflop: ${obs.legal.map((a) => a.shortLabel).join(", ")}');
    print('preflop sizings (chips): ${obs.sizingsByActionType}');
    // CHECK_CALL on SB facing BB = call to 100 chips.
    assert(obs.sizingsByActionType[ActionType.checkCall.index] == 100);
    // ALL_IN = total invest after = 10000 chips (full stack).
    assert(obs.sizingsByActionType[ActionType.allIn.index] == 10000);

    // ─── 3. Hand-pick cards ──────────────────────────────────────────────
    env.clearCards();
    env.setHole(Player.sb, EngineCard.parse('As'), EngineCard.parse('Kh'));
    env.setHole(Player.bb, EngineCard.parse('Qd'), EngineCard.parse('Qc'));
    env.setBoard([
      EngineCard.parse('2s'), EngineCard.parse('7d'), EngineCard.parse('Tc'),
      EngineCard.noCard, EngineCard.noCard,
    ]);
    final sbHole = env.getHole(Player.sb);
    final bbHole = env.getHole(Player.bb);
    final board  = env.getBoard();
    print('after override: SB=${sbHole.map(EngineCard.toReadable).join(" ")} '
        'BB=${bbHole.map(EngineCard.toReadable).join(" ")} '
        'board=${board.map(EngineCard.toReadable).join(" ")}');
    assert(sbHole[0] == EngineCard.parse('As'));

    // Uniqueness rejection.
    bool threw = false;
    try {
      env.setHole(Player.bb, EngineCard.parse('As'), EngineCard.parse('5h'));
    } catch (_) { threw = true; }
    assert(threw, 'expected duplicate-card rejection');

    // ─── 4. Step preflop → check-call → check-call → flop ────────────────
    obs = env.observe();
    final raise25Idx = obs.legal.indexOf(ActionType.raise25);
    assert(raise25Idx >= 0, 'RAISE_25 should be legal preflop on SB');
    var done = env.applyLegal(raise25Idx);
    assert(!done);

    obs = env.observe();
    final callIdx = obs.legal.indexOf(ActionType.checkCall);
    assert(callIdx >= 0);
    done = env.applyLegal(callIdx);   // BB calls
    assert(!done);

    snap = env.snapshot();
    assert(snap.street == Street.flop, 'expected flop, got ${snap.street}');
    assert(snap.toAct == Player.bb, 'BB acts first postflop');
    print('post-call: street=${snap.street} pot=${snap.potBb}bb toAct=${snap.toAct}');

    // History readout — should have 2 preflop entries.
    final hist = env.readHistory();
    assert(hist.length == 2);
    assert(hist[0].actor == Player.sb);
    assert(hist[0].type == ActionType.raise25);
    assert(hist[1].actor == Player.bb);
    assert(hist[1].type == ActionType.checkCall);
    print('history: ${hist.map((h) => "${h.actor.shortLabel} ${h.type.shortLabel} ${h.betToBb}bb").join(" / ")}');

    // ─── 5. Replay-by-action (BB checks flop, SB checks → turn) ──────────
    done = env.applyAction(ActionType.checkCall);
    assert(!done);
    done = env.applyAction(ActionType.checkCall);
    assert(!done);
    snap = env.snapshot();
    assert(snap.street == Street.turn);

    // ─── 6. Terminate via fold ───────────────────────────────────────────
    obs = env.observe();
    final foldIdx = obs.legal.indexOf(ActionType.fold);
    final raiseIdx = obs.legal.indexWhere((a) => a.isRaise);
    if (raiseIdx >= 0) {
      env.applyLegal(raiseIdx);
      obs = env.observe();
      final fIdx = obs.legal.indexOf(ActionType.fold);
      assert(fIdx >= 0);
      done = env.applyLegal(fIdx);
    } else {
      assert(foldIdx >= 0);
      done = env.applyLegal(foldIdx);
    }
    assert(done);
    snap = env.snapshot();
    assert(snap.isTerminal);
    assert(snap.terminalReason == Terminal.fold);
    assert(snap.payoffChips[0] + snap.payoffChips[1] == 0);
    print('terminal: payoffSB=${snap.payoffChips[0]} payoffBB=${snap.payoffChips[1]}');

    print('OK — Dart FFI smoke passed.');
  } finally {
    env.dispose();
  }
}
