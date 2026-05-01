/// Raw dart:ffi bindings for libpokertrainer.so.
///
/// One Dart top-level function per C symbol; no semantics. The typed wrapper
/// lives in `engine.dart`. Field offsets in the struct mirrors are pinned by
/// matching `static_assert(sizeof(...))` checks on the C side — see
/// `engine/bindings/ffi.cpp`.
library;

import 'dart:ffi' as ffi;
import 'dart:io';

// ─── struct mirrors (must match engine/bindings/ffi.cpp byte-for-byte) ────

/// One-shot table-state read. Total size = 120 bytes; layout pinned by
/// `static_assert(sizeof(PtStateSnapshot) == 120, ...)` on the C side.
final class PtStateSnapshot extends ffi.Struct {
  @ffi.Int64() external int potChips;

  @ffi.Array(2) external ffi.Array<ffi.Int64> stacksChips;            // [SB, BB]
  @ffi.Array(2) external ffi.Array<ffi.Int64> investedThisStreet;     // [SB, BB]
  @ffi.Array(2) external ffi.Array<ffi.Int64> investedPriorStreets;   // [SB, BB]

  @ffi.Int64() external int toCallChips;
  @ffi.Int64() external int minRaiseToChips;

  @ffi.Array(2) external ffi.Array<ffi.Int64> payoffChips;            // [SB, BB]

  @ffi.Int32() external int street;        // 0=PRE, 1=FLOP, 2=TURN, 3=RIVER, 4=SHOWDOWN
  @ffi.Int32() external int toAct;         // 0=SB, 1=BB, -1 if terminal
  @ffi.Int32() external int terminal;      // 0=NONE, 1=FOLD, 2=SHOWDOWN
  @ffi.Int32() external int isTerminal;    // 0/1

  @ffi.Array(2) external ffi.Array<ffi.Int32> allIn;                  // [SB, BB]

  @ffi.Int32() external int historySize;
  @ffi.Int32() external int padding;       // explicit, do not read
}

/// One historical action. Total size = 40 bytes.
final class PtHistoryEntry extends ffi.Struct {
  @ffi.Int64() external int betToChips;
  @ffi.Int64() external int potAfterChips;
  @ffi.Int64() external int stackAfterChips;
  @ffi.Int32() external int actor;       // 0=SB, 1=BB
  @ffi.Int32() external int street;
  @ffi.Int32() external int type;        // ActionType ordinal
  @ffi.Int32() external int wasAllIn;    // 0/1
}

// ─── typedefs ─────────────────────────────────────────────────────────────

typedef _IntCNative   = ffi.Int32 Function();
typedef _IntDart      = int Function();

typedef _StreetIntC   = ffi.Int32 Function(ffi.Int32);
typedef _StreetIntD   = int Function(int);

typedef _EvalInitC    = ffi.Int32 Function(ffi.Pointer<ffi.Char>);
typedef _EvalInitD    = int Function(ffi.Pointer<ffi.Char>);

typedef _EnvCreateC   = ffi.Pointer<ffi.Void> Function(ffi.Uint64, ffi.Int64);
typedef _EnvCreateD   = ffi.Pointer<ffi.Void> Function(int, int);

typedef _VoidPC       = ffi.Void Function(ffi.Pointer<ffi.Void>);
typedef _VoidPD       = void Function(ffi.Pointer<ffi.Void>);

typedef _EnvIntC      = ffi.Int32 Function(ffi.Pointer<ffi.Void>);
typedef _EnvIntD      = int Function(ffi.Pointer<ffi.Void>);

typedef _EnvResetSeedC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Uint64);
typedef _EnvResetSeedD = int Function(ffi.Pointer<ffi.Void>, int);

typedef _EnvObserveC  = ffi.Int32 Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>, ffi.Pointer<ffi.Int32>, ffi.Int32);
typedef _EnvObserveD  = int Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>, ffi.Pointer<ffi.Int32>, int);

typedef _EnvStepC     = ffi.Int32 Function(
    ffi.Pointer<ffi.Void>, ffi.Int32, ffi.Pointer<ffi.Double>);
typedef _EnvStepD     = int Function(
    ffi.Pointer<ffi.Void>, int, ffi.Pointer<ffi.Double>);

typedef _EnvPayoffsC  = ffi.Void Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Double>, ffi.Pointer<ffi.Double>);
typedef _EnvPayoffsD  = void Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Double>, ffi.Pointer<ffi.Double>);

typedef _EnvSnapshotC = ffi.Int32 Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<PtStateSnapshot>);
typedef _EnvSnapshotD = int Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<PtStateSnapshot>);

typedef _EnvHistoryAtC = ffi.Int32 Function(
    ffi.Pointer<ffi.Void>, ffi.Int32, ffi.Pointer<PtHistoryEntry>);
typedef _EnvHistoryAtD = int Function(
    ffi.Pointer<ffi.Void>, int, ffi.Pointer<PtHistoryEntry>);

typedef _EnvLegalSizingsC = ffi.Int32 Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Int64>);
typedef _EnvLegalSizingsD = int Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Int64>);

typedef _EnvSetHoleC  = ffi.Int32 Function(
    ffi.Pointer<ffi.Void>, ffi.Int32, ffi.Uint8, ffi.Uint8);
typedef _EnvSetHoleD  = int Function(
    ffi.Pointer<ffi.Void>, int, int, int);

typedef _EnvSetBoardC = ffi.Int32 Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>);
typedef _EnvSetBoardD = int Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>);

typedef _EnvGetHoleC  = ffi.Int32 Function(
    ffi.Pointer<ffi.Void>, ffi.Int32,
    ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _EnvGetHoleD  = int Function(
    ffi.Pointer<ffi.Void>, int,
    ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);

typedef _EnvGetBoardC = ffi.Int32 Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>);
typedef _EnvGetBoardD = int Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>);

typedef _MakeCardC    = ffi.Uint8 Function(ffi.Int32, ffi.Int32);
typedef _MakeCardD    = int Function(int, int);

typedef _ModelLoadC   = ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Char>, ffi.Pointer<ffi.Char>);
typedef _ModelLoadD   = ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Char>, ffi.Pointer<ffi.Char>);

typedef _ModelQueryC  = ffi.Int32 Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>);
typedef _ModelQueryD  = int Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>);

// ─── library loader ────────────────────────────────────────────────────────

/// Resolves the path to libpokertrainer.so. In dev we read from the in-tree
/// build directory; the prod-bundled location will be `assets/<platform>/`
/// once the build pipeline is wired up.
String _resolveLibraryPath() {
  // Override hook for tests + tooling.
  final env = Platform.environment['POKERTRAINER_SO'];
  if (env != null && env.isNotEmpty) return env;

  // Walk up from the script directory looking for engine/build/<libname>.
  final libname = Platform.isLinux
      ? 'libpokertrainer.so'
      : Platform.isMacOS
          ? 'libpokertrainer.dylib'
          : 'pokertrainer.dll';

  // Common dev path: invoked from anywhere inside PokerTrainer/.
  final candidates = <String>[
    '/home/elucterio/Poker/PokerTrainer/engine/build/$libname',
    'engine/build/$libname',
    '../engine/build/$libname',
  ];
  for (final p in candidates) {
    if (File(p).existsSync()) return p;
  }
  throw StateError(
      'Could not find $libname. Set POKERTRAINER_SO or build the engine '
      '(cmake --build engine/build).');
}

/// Native bindings — one instance per process. Hand off to the typed wrapper
/// in `engine.dart` for everything else.
final class PokerNative {
  final ffi.DynamicLibrary lib;

  // Constants
  final int Function() ptXDim;
  final int Function() ptADim;
  final int Function() ptNumActions;
  final int Function() ptHistMax;
  final int Function() ptHistFeat;
  final int Function(int) ptXOffStreet;
  final int Function(int) ptStreetSlots;
  final int Function() ptModelAvailable;

  // Eval init
  final int Function(ffi.Pointer<ffi.Char>) ptEvalInit;

  // Card helpers
  final int Function(int, int) ptMakeCard;

  // Env lifecycle
  final ffi.Pointer<ffi.Void> Function(int, int) ptEnvCreate;
  final void Function(ffi.Pointer<ffi.Void>) ptEnvDestroy;
  final int Function(ffi.Pointer<ffi.Void>) ptEnvReset;
  final int Function(ffi.Pointer<ffi.Void>, int) ptEnvResetSeed;

  // Env queries
  final int Function(ffi.Pointer<ffi.Void>) ptEnvIsTerminal;
  final int Function(ffi.Pointer<ffi.Void>) ptEnvToAct;
  final int Function(ffi.Pointer<ffi.Void>) ptEnvNLegal;
  final int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>,
      ffi.Pointer<ffi.Float>, ffi.Pointer<ffi.Int32>, int) ptEnvObserve;
  final int Function(ffi.Pointer<ffi.Void>, int,
      ffi.Pointer<ffi.Double>) ptEnvStep;
  final int Function(ffi.Pointer<ffi.Void>, int,
      ffi.Pointer<ffi.Double>) ptEnvStepAction;
  final void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Double>,
      ffi.Pointer<ffi.Double>) ptEnvPayoffsBb;

  // Inspector surface
  final int Function(ffi.Pointer<ffi.Void>,
      ffi.Pointer<PtStateSnapshot>) ptEnvStateSnapshot;
  final int Function(ffi.Pointer<ffi.Void>, int,
      ffi.Pointer<PtHistoryEntry>) ptEnvHistoryAt;
  final int Function(ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Int64>) ptEnvLegalSizings;
  final int Function(ffi.Pointer<ffi.Void>) ptEnvClearCards;
  final int Function(ffi.Pointer<ffi.Void>, int, int, int) ptEnvSetHole;
  final int Function(ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Uint8>) ptEnvSetBoard;
  final int Function(ffi.Pointer<ffi.Void>, int,
      ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>) ptEnvGetHole;
  final int Function(ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Uint8>) ptEnvGetBoard;

  // Model surface (gated runtime via ptModelAvailable)
  final ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Char>, ffi.Pointer<ffi.Char>) ptModelLoad;
  final void Function(ffi.Pointer<ffi.Void>) ptModelDestroy;
  final int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Float>) ptModelQuery;

  PokerNative._(this.lib)
      : ptXDim         = lib.lookupFunction<_IntCNative, _IntDart>('pt_x_dim'),
        ptADim         = lib.lookupFunction<_IntCNative, _IntDart>('pt_a_dim'),
        ptNumActions   = lib.lookupFunction<_IntCNative, _IntDart>('pt_num_actions'),
        ptHistMax      = lib.lookupFunction<_IntCNative, _IntDart>('pt_hist_max'),
        ptHistFeat     = lib.lookupFunction<_IntCNative, _IntDart>('pt_hist_feat'),
        ptXOffStreet   = lib.lookupFunction<_StreetIntC, _StreetIntD>('pt_x_off_street'),
        ptStreetSlots  = lib.lookupFunction<_StreetIntC, _StreetIntD>('pt_street_slots'),
        ptModelAvailable = lib.lookupFunction<_IntCNative, _IntDart>('pt_model_available'),
        ptEvalInit     = lib.lookupFunction<_EvalInitC, _EvalInitD>('pt_eval_init'),
        ptMakeCard     = lib.lookupFunction<_MakeCardC, _MakeCardD>('pt_make_card'),
        ptEnvCreate    = lib.lookupFunction<_EnvCreateC, _EnvCreateD>('pt_env_create'),
        ptEnvDestroy   = lib.lookupFunction<_VoidPC, _VoidPD>('pt_env_destroy'),
        ptEnvReset     = lib.lookupFunction<_EnvIntC, _EnvIntD>('pt_env_reset'),
        ptEnvResetSeed = lib.lookupFunction<_EnvResetSeedC, _EnvResetSeedD>('pt_env_reset_seed'),
        ptEnvIsTerminal = lib.lookupFunction<_EnvIntC, _EnvIntD>('pt_env_is_terminal'),
        ptEnvToAct     = lib.lookupFunction<_EnvIntC, _EnvIntD>('pt_env_to_act'),
        ptEnvNLegal    = lib.lookupFunction<_EnvIntC, _EnvIntD>('pt_env_n_legal'),
        ptEnvObserve   = lib.lookupFunction<_EnvObserveC, _EnvObserveD>('pt_env_observe'),
        ptEnvStep      = lib.lookupFunction<_EnvStepC, _EnvStepD>('pt_env_step'),
        ptEnvStepAction = lib.lookupFunction<_EnvStepC, _EnvStepD>('pt_env_step_action'),
        ptEnvPayoffsBb = lib.lookupFunction<_EnvPayoffsC, _EnvPayoffsD>('pt_env_payoffs_bb'),
        ptEnvStateSnapshot = lib.lookupFunction<_EnvSnapshotC, _EnvSnapshotD>('pt_env_state_snapshot'),
        ptEnvHistoryAt = lib.lookupFunction<_EnvHistoryAtC, _EnvHistoryAtD>('pt_env_history_at'),
        ptEnvLegalSizings = lib.lookupFunction<_EnvLegalSizingsC, _EnvLegalSizingsD>('pt_env_legal_sizings'),
        ptEnvClearCards = lib.lookupFunction<_EnvIntC, _EnvIntD>('pt_env_clear_cards'),
        ptEnvSetHole   = lib.lookupFunction<_EnvSetHoleC, _EnvSetHoleD>('pt_env_set_hole'),
        ptEnvSetBoard  = lib.lookupFunction<_EnvSetBoardC, _EnvSetBoardD>('pt_env_set_board'),
        ptEnvGetHole   = lib.lookupFunction<_EnvGetHoleC, _EnvGetHoleD>('pt_env_get_hole'),
        ptEnvGetBoard  = lib.lookupFunction<_EnvGetBoardC, _EnvGetBoardD>('pt_env_get_board'),
        ptModelLoad    = lib.lookupFunction<_ModelLoadC, _ModelLoadD>('pt_model_load'),
        ptModelDestroy = lib.lookupFunction<_VoidPC, _VoidPD>('pt_model_destroy'),
        ptModelQuery   = lib.lookupFunction<_ModelQueryC, _ModelQueryD>('pt_model_query');

  static PokerNative? _instance;

  /// Loads the .so and binds all symbols. Idempotent within a process.
  factory PokerNative.load({String? libraryPath}) {
    if (_instance != null) return _instance!;
    final path = libraryPath ?? _resolveLibraryPath();
    final lib = ffi.DynamicLibrary.open(path);
    return _instance = PokerNative._(lib);
  }
}
