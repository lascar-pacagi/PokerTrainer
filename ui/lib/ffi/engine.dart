/// Typed wrapper around the raw FFI bindings.
///
/// Owns malloc/free discipline so screens can hand off `EnvHandle` /
/// `ModelHandle` and never see a `Pointer<Void>`. Surfaces engine state via
/// plain Dart records / classes — Flutter widgets read these and never poke
/// the C structs directly.
library;

import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'actions.dart';
import 'bindings.dart';

/// Process-wide singleton — wraps the loaded .so + the cached engine
/// constants. Loads the hand-eval tables on first call so callers don't
/// need to remember.
class PokerEngine {
  final PokerNative _native;

  /// Cached after the first `pt_*` constant lookup so widgets can read
  /// without an FFI hop.
  late final int xDim;
  late final int aDim;
  late final int numActions;
  late final int histMax;
  late final int histFeat;
  late final List<int> streetOffsets;   // [preflop, flop, turn, river]
  late final List<int> streetSlots;
  late final bool modelAvailable;

  PokerEngine._(this._native) {
    xDim          = _native.ptXDim();
    aDim          = _native.ptADim();
    numActions    = _native.ptNumActions();
    histMax       = _native.ptHistMax();
    histFeat      = _native.ptHistFeat();
    streetOffsets = [
      for (int s = 0; s < 4; s++) _native.ptXOffStreet(s),
    ];
    streetSlots = [
      for (int s = 0; s < 4; s++) _native.ptStreetSlots(s),
    ];
    modelAvailable = _native.ptModelAvailable() == 1;
  }

  static PokerEngine? _instance;

  /// Loads the .so + initializes the hand-eval tables. Call once at app
  /// startup. `tablesPath` is passed straight to `pt_eval_init`; an empty
  /// string means "load-or-generate at the engine's default location."
  factory PokerEngine.init({String? libraryPath, String tablesPath = ''}) {
    if (_instance != null) return _instance!;
    final native = PokerNative.load(libraryPath: libraryPath);
    final tablesC = tablesPath.toNativeUtf8().cast<ffi.Char>();
    try {
      final rc = native.ptEvalInit(tablesC);
      if (rc != 0) {
        throw StateError('pt_eval_init failed (rc=$rc, path="$tablesPath")');
      }
    } finally {
      malloc.free(tablesC);
    }
    return _instance = PokerEngine._(native);
  }

  /// Build a card byte from rank (0..12, 2..A) and suit (0..3, cdhs).
  int makeCard(int rank, int suit) => _native.ptMakeCard(rank, suit);

  PokerNative get native => _native;
}

/// Plain-data view of the table state at one point in time. Painted from
/// `PtStateSnapshot` so widgets can `==` / cache it; the underlying FFI
/// struct is never held past the call that filled it.
class TableState {
  final int  potChips;
  final List<int> stacksChips;            // [SB, BB]
  final List<int> investedThisStreet;     // [SB, BB]
  final List<int> investedPriorStreets;   // [SB, BB]
  final int  toCallChips;
  final int  minRaiseToChips;
  final List<int> payoffChips;            // [SB, BB] — zero mid-hand
  final Street   street;
  final Player?  toAct;                   // null iff terminal
  final Terminal terminalReason;
  final bool isTerminal;
  final List<bool> allIn;                 // [SB, BB]
  final int  historySize;

  const TableState({
    required this.potChips,
    required this.stacksChips,
    required this.investedThisStreet,
    required this.investedPriorStreets,
    required this.toCallChips,
    required this.minRaiseToChips,
    required this.payoffChips,
    required this.street,
    required this.toAct,
    required this.terminalReason,
    required this.isTerminal,
    required this.allIn,
    required this.historySize,
  });

  factory TableState._fromSnap(ffi.Pointer<PtStateSnapshot> ptr) {
    final s = ptr.ref;
    return TableState(
      potChips: s.potChips,
      stacksChips: [s.stacksChips[0], s.stacksChips[1]],
      investedThisStreet: [s.investedThisStreet[0], s.investedThisStreet[1]],
      investedPriorStreets: [s.investedPriorStreets[0], s.investedPriorStreets[1]],
      toCallChips: s.toCallChips,
      minRaiseToChips: s.minRaiseToChips,
      payoffChips: [s.payoffChips[0], s.payoffChips[1]],
      street: Street.fromInt(s.street),
      toAct: s.toAct < 0 ? null : Player.fromInt(s.toAct),
      terminalReason: Terminal.fromInt(s.terminal),
      isTerminal: s.isTerminal == 1,
      allIn: [s.allIn[0] == 1, s.allIn[1] == 1],
      historySize: s.historySize,
    );
  }

  static const int chipsPerBb = 100;

  double get potBb => potChips / chipsPerBb;
  double stackBb(Player p) => stacksChips[p.index] / chipsPerBb;
  double investedThisStreetBb(Player p) =>
      investedThisStreet[p.index] / chipsPerBb;
  double get toCallBb => toCallChips / chipsPerBb;

  /// Effective stack in chips (the smaller player's total committed+behind).
  int get effectiveStackChips {
    final sb = stacksChips[0] + investedThisStreet[0];
    final bb = stacksChips[1] + investedThisStreet[1];
    return sb < bb ? sb : bb;
  }
}

/// One past action in the hand history.
class HistoryEntry {
  final Player actor;
  final Street street;
  final ActionType type;
  final int  betToChips;
  final int  potAfterChips;
  final int  stackAfterChips;
  final bool wasAllIn;

  const HistoryEntry({
    required this.actor,
    required this.street,
    required this.type,
    required this.betToChips,
    required this.potAfterChips,
    required this.stackAfterChips,
    required this.wasAllIn,
  });

  factory HistoryEntry._fromEntry(ffi.Pointer<PtHistoryEntry> ptr) {
    final e = ptr.ref;
    return HistoryEntry(
      actor: Player.fromInt(e.actor),
      street: Street.fromInt(e.street),
      type: ActionType.fromInt(e.type),
      betToChips: e.betToChips,
      potAfterChips: e.potAfterChips,
      stackAfterChips: e.stackAfterChips,
      wasAllIn: e.wasAllIn == 1,
    );
  }

  double get betToBb        => betToChips / TableState.chipsPerBb;
  double get potAfterBb     => potAfterChips / TableState.chipsPerBb;
  double get stackAfterBb   => stackAfterChips / TableState.chipsPerBb;
}

/// What the network sees on this turn — the encoded `x` plus the legal
/// action set. The strategy panel asks the model for Q-values keyed off the
/// same legal-actions vector this carries, so order is load-bearing.
class Observation {
  final Float32List x;        // (xDim,)
  final Float32List aFlat;    // (nLegal * aDim,)
  final List<ActionType> legal;
  /// Absolute bet-to chip target per ActionType slot. -1 for illegal slots,
  /// 0 for FOLD. Index by `ActionType.index`, NOT by `legal[i]`. Mirrors
  /// `pt_env_legal_sizings` exactly.
  final List<int> sizingsByActionType;

  const Observation({
    required this.x,
    required this.aFlat,
    required this.legal,
    required this.sizingsByActionType,
  });

  /// Bet-to chip target for the i-th legal action (so `legal[i]` and
  /// `sizingsForLegal(i)` are aligned).
  int sizingForLegal(int i) => sizingsByActionType[legal[i].index];
}

/// One handle = one Env. Owns the C-side `Pointer<Void>`. Always destroy.
class EnvHandle {
  final PokerEngine engine;
  ffi.Pointer<ffi.Void> _ptr;
  bool _disposed = false;

  EnvHandle._(this.engine, this._ptr);

  /// Allocate a fresh Env. `seed` drives the internal RNG for subsequent
  /// random deals; `startingStackChips` defaults to 100 BB at 100 chips/BB.
  factory EnvHandle.create(
    PokerEngine engine, {
    int seed = 0xC0FFEE,
    int startingStackChips = 100 * TableState.chipsPerBb,
  }) {
    final ptr = engine.native.ptEnvCreate(seed, startingStackChips);
    if (ptr == ffi.nullptr) {
      throw StateError('pt_env_create returned nullptr');
    }
    return EnvHandle._(engine, ptr);
  }

  void dispose() {
    if (_disposed) return;
    engine.native.ptEnvDestroy(_ptr);
    _ptr = ffi.nullptr;
    _disposed = true;
  }

  void _checkAlive() {
    if (_disposed) throw StateError('EnvHandle used after dispose');
  }

  /// Re-deal a fresh hand using the Env's internal RNG.
  void reset() {
    _checkAlive();
    final rc = engine.native.ptEnvReset(_ptr);
    if (rc != 0) throw StateError('pt_env_reset failed rc=$rc');
  }

  /// Re-deal a reproducible hand from `handSeed`.
  void resetSeed(int handSeed) {
    _checkAlive();
    final rc = engine.native.ptEnvResetSeed(_ptr, handSeed);
    if (rc != 0) throw StateError('pt_env_reset_seed failed rc=$rc');
  }

  bool get isTerminal {
    _checkAlive();
    return engine.native.ptEnvIsTerminal(_ptr) == 1;
  }

  Player? get toAct {
    _checkAlive();
    final v = engine.native.ptEnvToAct(_ptr);
    return v < 0 ? null : Player.fromInt(v);
  }

  /// One-shot table state read.
  TableState snapshot() {
    _checkAlive();
    final p = malloc<PtStateSnapshot>();
    try {
      final rc = engine.native.ptEnvStateSnapshot(_ptr, p);
      if (rc != 0) throw StateError('pt_env_state_snapshot failed rc=$rc');
      return TableState._fromSnap(p);
    } finally {
      malloc.free(p);
    }
  }

  /// Reads the entire history vector.
  List<HistoryEntry> readHistory() {
    _checkAlive();
    final s = snapshot();
    final n = s.historySize;
    final entries = <HistoryEntry>[];
    if (n == 0) return entries;
    final p = malloc<PtHistoryEntry>();
    try {
      for (int i = 0; i < n; i++) {
        final rc = engine.native.ptEnvHistoryAt(_ptr, i, p);
        if (rc != 0) throw StateError('pt_env_history_at($i) failed rc=$rc');
        entries.add(HistoryEntry._fromEntry(p));
      }
    } finally {
      malloc.free(p);
    }
    return entries;
  }

  /// Encoded observation + legal action menu for the current decision point.
  /// Throws if the env is terminal.
  Observation observe() {
    _checkAlive();
    final n = engine.native.ptEnvNLegal(_ptr);
    if (n < 0) throw StateError('pt_env_n_legal failed rc=$n');
    if (n == 0) {
      throw StateError('observe() on terminal env');
    }

    final xPtr     = malloc<ffi.Float>(engine.xDim);
    final aPtr     = malloc<ffi.Float>(n * engine.aDim);
    final legalPtr = malloc<ffi.Int32>(n);
    final szPtr    = malloc<ffi.Int64>(engine.numActions);
    try {
      final rc = engine.native.ptEnvObserve(_ptr, xPtr, aPtr, legalPtr, n);
      if (rc < 0) throw StateError('pt_env_observe failed rc=$rc');
      final szRc = engine.native.ptEnvLegalSizings(_ptr, szPtr);
      if (szRc < 0) throw StateError('pt_env_legal_sizings failed rc=$szRc');

      // Copy out so we can free the native buffers.
      final x = Float32List.fromList(
        xPtr.asTypedList(engine.xDim));
      final a = Float32List.fromList(
        aPtr.asTypedList(n * engine.aDim));
      final legalArr = legalPtr.asTypedList(n);
      final legal = List<ActionType>.generate(
        n, (i) => ActionType.fromInt(legalArr[i]),
        growable: false);
      final szArr = szPtr.asTypedList(engine.numActions);
      final sizings = List<int>.from(szArr);

      return Observation(
        x: x, aFlat: a, legal: legal, sizingsByActionType: sizings);
    } finally {
      malloc.free(xPtr);
      malloc.free(aPtr);
      malloc.free(legalPtr);
      malloc.free(szPtr);
    }
  }

  /// Apply by index into the most-recent observe()'s `legal` vector.
  /// Returns true iff the hand is terminal after this step.
  bool applyLegal(int legalIdx) {
    _checkAlive();
    final rewardPtr = malloc<ffi.Double>();
    try {
      final rc = engine.native.ptEnvStep(_ptr, legalIdx, rewardPtr);
      if (rc < 0) throw StateError('pt_env_step failed rc=$rc');
      return rc == 1;
    } finally {
      malloc.free(rewardPtr);
    }
  }

  /// Apply by ActionType (used for replay from a recorded history).
  bool applyAction(ActionType type) {
    _checkAlive();
    final rewardPtr = malloc<ffi.Double>();
    try {
      final rc = engine.native.ptEnvStepAction(_ptr, type.index, rewardPtr);
      if (rc < 0) throw StateError('pt_env_step_action failed rc=$rc');
      return rc == 1;
    } finally {
      malloc.free(rewardPtr);
    }
  }

  // ─── Inspector card setters ────────────────────────────────────────────

  /// Wipe all 4 hole + 5 board slots to NO_CARD.
  void clearCards() {
    _checkAlive();
    final rc = engine.native.ptEnvClearCards(_ptr);
    if (rc != 0) throw StateError('pt_env_clear_cards failed rc=$rc');
  }

  /// Overwrite a player's hole pair. Throws on uniqueness violation.
  void setHole(Player player, int c0, int c1) {
    _checkAlive();
    final rc = engine.native.ptEnvSetHole(_ptr, player.index, c0, c1);
    if (rc != 0) {
      throw StateError(
          'pt_env_set_hole rejected (${EngineCard.toReadable(c0)} ${EngineCard.toReadable(c1)}) — '
          'duplicate against existing dealt cards?');
    }
  }

  /// Overwrite the board (5 slots, NO_CARD = unset).
  void setBoard(List<int> cards5) {
    if (cards5.length != 5) {
      throw ArgumentError('setBoard: need exactly 5 cards (NO_CARD for unset)');
    }
    _checkAlive();
    final p = malloc<ffi.Uint8>(5);
    try {
      for (int i = 0; i < 5; i++) {
        p[i] = cards5[i];
      }
      final rc = engine.native.ptEnvSetBoard(_ptr, p);
      if (rc != 0) {
        throw StateError('pt_env_set_board rejected — duplicate cards?');
      }
    } finally {
      malloc.free(p);
    }
  }

  /// Read SB or BB hole. Returns `[c0, c1]`.
  List<int> getHole(Player player) {
    _checkAlive();
    final c0 = malloc<ffi.Uint8>();
    final c1 = malloc<ffi.Uint8>();
    try {
      final rc = engine.native.ptEnvGetHole(_ptr, player.index, c0, c1);
      if (rc != 0) throw StateError('pt_env_get_hole failed rc=$rc');
      return [c0.value, c1.value];
    } finally {
      malloc.free(c0);
      malloc.free(c1);
    }
  }

  /// Read the 5-slot board.
  List<int> getBoard() {
    _checkAlive();
    final p = malloc<ffi.Uint8>(5);
    try {
      final rc = engine.native.ptEnvGetBoard(_ptr, p);
      if (rc != 0) throw StateError('pt_env_get_board failed rc=$rc');
      return [for (int i = 0; i < 5; i++) p[i]];
    } finally {
      malloc.free(p);
    }
  }
}

/// One handle = one TorchScript model loaded into LibTorch. Always destroy.
class ModelHandle {
  final PokerEngine engine;
  ffi.Pointer<ffi.Void> _ptr;
  bool _disposed = false;

  ModelHandle._(this.engine, this._ptr);

  /// Loads a TorchScript .pt. Returns null if the inference surface isn't
  /// linked into this build of the .so.
  static ModelHandle? load(
    PokerEngine engine,
    String path, {
    String device = 'cpu',
  }) {
    if (!engine.modelAvailable) return null;
    final p = path.toNativeUtf8().cast<ffi.Char>();
    final d = device.toNativeUtf8().cast<ffi.Char>();
    try {
      final ptr = engine.native.ptModelLoad(p, d);
      if (ptr == ffi.nullptr) return null;
      return ModelHandle._(engine, ptr);
    } finally {
      malloc.free(p);
      malloc.free(d);
    }
  }

  void dispose() {
    if (_disposed) return;
    engine.native.ptModelDestroy(_ptr);
    _ptr = ffi.nullptr;
    _disposed = true;
  }

  /// Score the env's current observation. Returns the argmax legal-action
  /// index plus the per-legal-action Q-values (one value per row of the
  /// observation's `legal`, in the same order). Throws on terminal env or
  /// other failure.
  ({int argmax, Float32List qValues}) query(EnvHandle env) {
    if (_disposed) throw StateError('ModelHandle used after dispose');
    final n = engine.native.ptEnvNLegal(env._ptr);
    if (n < 0) throw StateError('pt_env_n_legal failed rc=$n');
    if (n == 0) throw StateError('query on terminal env');

    final qPtr = malloc<ffi.Float>(n);
    try {
      final argmax = engine.native.ptModelQuery(_ptr, env._ptr, qPtr);
      if (argmax < 0) throw StateError('pt_model_query failed rc=$argmax');
      final q = Float32List.fromList(qPtr.asTypedList(n));
      return (argmax: argmax, qValues: q);
    } finally {
      malloc.free(qPtr);
    }
  }
}
