/// Spawns `pt-solver` as a subprocess and streams progress back to listeners.
///
/// The runner is stateful: it remembers the last estimate (for the dry-run
/// → solve pattern) and the in-flight subprocess so a "cancel" button can
/// kill it. Output is always written to a temp file under
/// `validation_runs/scenarios/`; auto-named via a hash of the input so
/// re-solving the same spot overwrites rather than littering.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';

import '../ffi/actions.dart';
import 'range_serializer.dart';
import 'scenario.dart';

/// How deep through chance nodes the tree dump descends. Maps directly to
/// pt-solver's `--depth` flag.
enum SolveDepth {
  /// Stop at the first chance node (turn deal). Smallest file (~150 KB).
  flop,
  /// Walk through every turn card; stop at river chance. ~5–50 MB.
  turn,
  /// Walk through both turn and river chance. Hundreds of MB; only practical
  /// for narrow ranges or shallow stacks.
  river;

  String get cliFlag => name; // values: 'flop', 'turn', 'river' — match Rust ValueEnum.

  String get human {
    switch (this) {
      case SolveDepth.flop:
        return 'Flop only';
      case SolveDepth.turn:
        return 'Flop + Turn';
      case SolveDepth.river:
        return 'Flop + Turn + River';
    }
  }
}

/// Mirrors `SpotInput` in `validation/src/main.rs`. The names match the JSON
/// field names so we can `jsonEncode` the struct directly.
class SpotConfig {
  String oopRange;
  String ipRange;
  String flop; // e.g. "Td9d6h"
  String? turn;
  String? river;
  int startingPot;
  int effectiveStack;
  String flopBet;
  String flopRaise;
  String turnBet;
  String turnRaise;
  String riverBet;
  String riverRaise;
  int maxIterations;
  double targetExploitabilityPctPot;
  /// Tree-walk depth — passed to pt-solver as `--depth` (not part of the
  /// SpotInput JSON). Folded into `fingerprint` so different depths yield
  /// different output filenames.
  SolveDepth depth;

  SpotConfig({
    required this.oopRange,
    required this.ipRange,
    required this.flop,
    this.turn,
    this.river,
    this.startingPot = 60,
    this.effectiveStack = 970,
    this.flopBet = '50%',
    this.flopRaise = '2.5x',
    this.turnBet = '50%',
    this.turnRaise = '2.5x',
    this.riverBet = '50%,a',
    this.riverRaise = '2.5x',
    // 200 iterations gets the SRP fixture from ~5% exploitability down to
    // ~0.5–1% — close enough for class-level study (action-EV gaps shrink
    // to ~0.05 chips). 50 iterations was visibly under-converged; users
    // saw "EV(check) > EV(bet)" alongside "bet 75%", which violates the
    // indifference principle and signalled a half-baked solve.
    this.maxIterations = 200,
    this.targetExploitabilityPctPot = 1.0,
    this.depth = SolveDepth.flop,
  });

  Map<String, dynamic> toJson() => {
        'oop_range': oopRange,
        'ip_range': ipRange,
        'flop': flop,
        'turn': turn,
        'river': river,
        'starting_pot': startingPot,
        'effective_stack': effectiveStack,
        'flop_bet_sizes': {'bet': flopBet, 'raise': flopRaise},
        'turn_bet_sizes': {'bet': turnBet, 'raise': turnRaise},
        'river_bet_sizes': {'bet': riverBet, 'raise': riverRaise},
        'max_iterations': maxIterations,
        'target_exploitability_pct_pot': targetExploitabilityPctPot,
      };

  factory SpotConfig.preset() => SpotConfig(
        oopRange: '66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs',
        ipRange: 'QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+',
        flop: 'Td9d6h',
        startingPot: 60,
        effectiveStack: 970,
      );

  /// 8-char content hash for auto-naming output files. Includes depth so
  /// the same spot at different depths produces distinct filenames.
  String get fingerprint {
    final payload = {
      ...toJson(),
      'depth': depth.cliFlag,
    };
    final bytes = utf8.encode(jsonEncode(payload));
    return sha1.convert(bytes).toString().substring(0, 8);
  }

  /// Default file name for solver output: `<flop>_<depth>_<hash>.json`.
  String suggestedFilename() => '${flop}_${depth.cliFlag}_$fingerprint.json';
}

/// Result of a successful dry-run.
class DryRunEstimate {
  final int memoryBytes;
  final int memoryBytesCompressed;
  final bool tooLarge;
  final int memoryCapBytes;

  const DryRunEstimate({
    required this.memoryBytes,
    required this.memoryBytesCompressed,
    required this.tooLarge,
    required this.memoryCapBytes,
  });

  factory DryRunEstimate.fromJson(Map<String, dynamic> j) => DryRunEstimate(
        memoryBytes: j['memory_bytes'] as int,
        memoryBytesCompressed: j['memory_bytes_compressed'] as int,
        tooLarge: j['too_large'] as bool,
        memoryCapBytes: j['memory_cap_bytes'] as int,
      );

  String get memoryHuman => _bytesHuman(memoryBytes);
  String get memoryCompressedHuman => _bytesHuman(memoryBytesCompressed);
  String get capHuman => _bytesHuman(memoryCapBytes);
}

String _bytesHuman(int b) {
  if (b < 1 << 20) return '${(b / 1024).toStringAsFixed(1)} KB';
  if (b < 1 << 30) return '${(b / (1 << 20)).toStringAsFixed(1)} MB';
  return '${(b / (1 << 30)).toStringAsFixed(2)} GB';
}

/// Runs `pt-solver` and exposes progress as a ChangeNotifier.
class SolverRunner extends ChangeNotifier {
  /// Path to the pt-solver binary. Defaulted via [_defaultBinary]; can be
  /// overridden for testing or non-standard layouts.
  final String binary;
  /// Directory where output JSONs land. Auto-created on first run.
  final String outputDir;

  Process? _proc;
  final List<String> _stderrLines = [];
  DryRunEstimate? _lastEstimate;
  String? _lastError;
  String? _lastOutputPath;
  bool _running = false;

  SolverRunner({String? binary, String? outputDir})
      : binary = binary ?? _defaultBinary(),
        outputDir = outputDir ?? _defaultOutputDir();

  static String _defaultBinary() {
    final candidates = [
      '/home/elucterio/Poker/PokerTrainer/validation/target/release/pt-solver',
      '${Platform.environment['HOME'] ?? ''}/Poker/PokerTrainer/validation/target/release/pt-solver',
      'validation/target/release/pt-solver',
    ];
    for (final c in candidates) {
      if (File(c).existsSync()) return c;
    }
    // Fallback — let the spawn fail with a clearer error if not found.
    return 'pt-solver';
  }

  static String _defaultOutputDir() {
    const candidates = [
      '/home/elucterio/Poker/PokerTrainer/validation_runs/scenarios',
    ];
    for (final c in candidates) {
      if (Directory(c).existsSync()) return c;
    }
    return candidates.first;
  }

  bool get running => _running;
  List<String> get stderrLines => List.unmodifiable(_stderrLines);
  DryRunEstimate? get lastEstimate => _lastEstimate;
  String? get lastError => _lastError;
  String? get lastOutputPath => _lastOutputPath;

  Future<void> _writeInput(SpotConfig cfg, File target) async {
    await target.parent.create(recursive: true);
    await target.writeAsString(jsonEncode(cfg.toJson()));
  }

  /// Run pt-solver with `--dry-run`. Updates [lastEstimate] on success or
  /// [lastError] on failure. Cheap (~100ms), so we just wait for it.
  Future<void> estimate(SpotConfig cfg) async {
    if (_running) return;
    _running = true;
    _lastError = null;
    _stderrLines.clear();
    notifyListeners();

    final tmpInput = File('${Directory.systemTemp.path}/pt_input_${cfg.fingerprint}.json');
    final tmpOutput = File('${Directory.systemTemp.path}/pt_estimate_${cfg.fingerprint}.json');
    try {
      await _writeInput(cfg, tmpInput);
      final result = await Process.run(
        binary,
        [
          '--input', tmpInput.path,
          '--output', tmpOutput.path,
          '--dry-run',
        ],
      );
      if (result.exitCode != 0) {
        _lastError = 'pt-solver exit ${result.exitCode}: ${result.stderr}';
        _lastEstimate = null;
      } else {
        for (final line in (result.stderr as String).split('\n')) {
          if (line.isNotEmpty) _stderrLines.add(line);
        }
        final raw = await tmpOutput.readAsString();
        _lastEstimate = DryRunEstimate.fromJson(jsonDecode(raw) as Map<String, dynamic>);
      }
    } catch (e) {
      _lastError = e.toString();
      _lastEstimate = null;
    } finally {
      _running = false;
      // Best-effort cleanup; ignore failures.
      try { await tmpInput.delete(); } catch (_) {}
      try { await tmpOutput.delete(); } catch (_) {}
      notifyListeners();
    }
  }

  /// Run the full solve. Streams stderr live into [stderrLines] so callers
  /// can render a progress log. Returns the path of the resulting JSON, or
  /// null on failure.
  Future<String?> solve(SpotConfig cfg) async {
    if (_running) return null;
    _running = true;
    _lastError = null;
    _lastOutputPath = null;
    _stderrLines.clear();
    notifyListeners();

    final outFile = File('$outputDir/${cfg.suggestedFilename()}');
    final tmpInput = File('${Directory.systemTemp.path}/pt_input_${cfg.fingerprint}.json');
    try {
      await _writeInput(cfg, tmpInput);
      await outFile.parent.create(recursive: true);
      _proc = await Process.start(
        binary,
        [
          '--input', tmpInput.path,
          '--output', outFile.path,
          '--mode', 'tree',
          '--depth', cfg.depth.cliFlag,
        ],
      );
      _proc!.stderr.transform(utf8.decoder).transform(const LineSplitter()).listen((line) {
        if (line.isEmpty) return;
        _stderrLines.add(line);
        notifyListeners();
      });
      // Drop stdout (we use --output, not stdout, for the JSON).
      _proc!.stdout.drain<void>();
      final code = await _proc!.exitCode;
      if (code != 0) {
        _lastError = 'pt-solver exit $code';
        return null;
      }
      _lastOutputPath = outFile.path;
      return outFile.path;
    } catch (e) {
      _lastError = e.toString();
      return null;
    } finally {
      _proc = null;
      _running = false;
      try { await tmpInput.delete(); } catch (_) {}
      notifyListeners();
    }
  }

  /// Solve a small subgame rooted at a `chance_pending` node — i.e. the
  /// user clicked "deal this card" on a turn or river chance. Builds a
  /// new SpotConfig where:
  ///
  ///   * Both ranges come from the parent action node's `weights`/`weightsOpp`,
  ///     re-serialized as PokerStove combo:weight pairs (combos that share a
  ///     card with [pickedCard] are dropped — they have weight 0 after the deal).
  ///   * The board is the parent's board plus the picked card.
  ///   * Pot/stack are taken from the parent (already reflect prior betting).
  ///   * Bet sizings, iterations, and rake are copied from the parent's
  ///     original [SpotInput].
  ///   * Depth is hard-coded to `flop` (smallest dump): walk only the action
  ///     subtree on the new street, stop at the next chance. The user can
  ///     then click again to drill deeper.
  ///
  /// On success returns the new file path; the caller loads it and switches
  /// the UI to the new scenario.
  Future<String?> expandChancePending({
    required Scenario scenario,
    required ScenarioNode chancePending,
    required int pickedCardByte,
    SolveDepth depth = SolveDepth.flop,
  }) async {
    if (_running) return null;
    _running = true;
    _lastError = null;
    _lastOutputPath = null;
    _stderrLines.clear();
    notifyListeners();

    try {
      final cfg = _buildSubgameConfig(
        scenario: scenario,
        chancePending: chancePending,
        pickedCardByte: pickedCardByte,
        depth: depth,
      );
      final pickedCardStr = EngineCard.toReadable(pickedCardByte);
      final parent = ScenarioParent(
        parentLabel: scenario.label,
        parentSourcePath: scenario.sourcePath,
        parentNodeId: chancePending.id,
        parentLine: chancePending.line,
        pickedCard: pickedCardStr,
      );
      // Output filename embeds parent fingerprint + picked card so re-clicking
      // overwrites idempotently and different runouts stay distinct.
      final outFile = File('$outputDir/${scenario.label}__$pickedCardStr'
          '_${cfg.fingerprint}.json');
      final tmpInput = File('${Directory.systemTemp.path}/'
          'pt_input_${cfg.fingerprint}.json');
      await _writeInput(cfg, tmpInput);
      await outFile.parent.create(recursive: true);
      _proc = await Process.start(
        binary,
        [
          '--input', tmpInput.path,
          '--output', outFile.path,
          '--mode', 'tree',
          '--depth', cfg.depth.cliFlag,
        ],
      );
      _proc!.stderr.transform(utf8.decoder).transform(const LineSplitter()).listen((line) {
        if (line.isEmpty) return;
        _stderrLines.add(line);
        notifyListeners();
      });
      _proc!.stdout.drain<void>();
      final code = await _proc!.exitCode;
      if (code != 0) {
        _lastError = 'pt-solver exit $code';
        try { await tmpInput.delete(); } catch (_) {}
        return null;
      }
      try { await tmpInput.delete(); } catch (_) {}
      // Stitch the parent metadata into the freshly written JSON so that
      // when it's loaded later, scenario.parent surfaces the back-link.
      try {
        final raw = await outFile.readAsString();
        final j = jsonDecode(raw) as Map<String, dynamic>;
        j['parent'] = parent.toJson();
        await outFile.writeAsString(jsonEncode(j));
      } catch (e) {
        // Non-fatal — solve still succeeded; just lose the back-link.
        _stderrLines.add('warn: could not annotate parent metadata: $e');
      }
      _lastOutputPath = outFile.path;
      return outFile.path;
    } catch (e) {
      _lastError = e.toString();
      return null;
    } finally {
      _proc = null;
      _running = false;
      notifyListeners();
    }
  }

  /// Build the SpotConfig for a subgame solve. Pulls weights from the
  /// chance_pending's parent action node (the last point with strategy
  /// data) and re-serializes them as PokerStove ranges.
  SpotConfig _buildSubgameConfig({
    required Scenario scenario,
    required ScenarioNode chancePending,
    required int pickedCardByte,
    SolveDepth depth = SolveDepth.flop,
  }) {
    // Find the action node immediately above the chance_pending. The
    // chance_pending's history minus its last entry gives the parent's id.
    String parentId;
    if (chancePending.id.isEmpty) {
      // chance is the root — unusual but handle gracefully.
      parentId = '';
    } else {
      final slash = chancePending.id.lastIndexOf('/');
      parentId = slash < 0 ? '' : chancePending.id.substring(0, slash);
    }
    final parent = scenario.nodeById(parentId) ?? chancePending;

    // Determine which weights belong to OOP vs IP. Parent is an action node
    // with `player` set; its `weights` are for that player, `weightsOpp` for
    // the other.
    final List<double> oopWeights;
    final List<double> ipWeights;
    if (parent.player == 'oop') {
      oopWeights = parent.weights;
      ipWeights = parent.weightsOpp;
    } else if (parent.player == 'ip') {
      oopWeights = parent.weightsOpp;
      ipWeights = parent.weights;
    } else {
      // Defensive fallback: equal weights (would be wrong but at least solves).
      oopWeights = List.filled(scenario.oopCombos.length, 1.0);
      ipWeights = List.filled(scenario.ipCombos.length, 1.0);
    }

    final oopRange = serializeRange(
      combos: scenario.oopCombos,
      weights: oopWeights,
      excludeCardByte: pickedCardByte,
    );
    final ipRange = serializeRange(
      combos: scenario.ipCombos,
      weights: ipWeights,
      excludeCardByte: pickedCardByte,
    );

    // Determine the new board configuration. The chance_pending's `board`
    // already reflects the prior streets (3 cards = turn pending, 4 = river
    // pending). The picked card extends it.
    final pickedStr = EngineCard.toReadable(pickedCardByte);
    final pendingBoardLen = chancePending.board.length;
    String? newTurn;
    String? newRiver;
    if (pendingBoardLen == 3) {
      // Pending was the turn deal.
      newTurn = pickedStr;
      newRiver = null;
    } else if (pendingBoardLen == 4) {
      // Pending was the river deal — the existing 4th card was the turn.
      newTurn = chancePending.board[3];
      newRiver = pickedStr;
    } else {
      // Unexpected — treat as turn pin and hope for the best.
      newTurn = pickedStr;
    }

    // Pot/stack at this chance (no betting on chance, so they match the
    // parent's). chance_pending stores them too — use them directly.
    final pot = chancePending.pot;
    final stacks = chancePending.stacks;
    final eff = stacks[0] < stacks[1] ? stacks[0] : stacks[1];

    // Lift the original spec's bet sizings, iterations, etc. from the
    // scenario's echoed input. Fall back to preset defaults if missing
    // (older JSONs without `input`).
    final raw = scenario.rawInput ?? const {};
    String pickStr(dynamic v, String fallback) =>
        v is String ? v : fallback;
    Map<String, dynamic> mapOr(dynamic v) =>
        v is Map<String, dynamic> ? v : const {};
    final flopBs = mapOr(raw['flop_bet_sizes']);
    final turnBs = mapOr(raw['turn_bet_sizes']);
    final riverBs = mapOr(raw['river_bet_sizes']);

    return SpotConfig(
      oopRange: oopRange,
      ipRange: ipRange,
      flop: '${chancePending.board[0]}${chancePending.board[1]}${chancePending.board[2]}',
      turn: newTurn,
      river: newRiver,
      startingPot: pot,
      effectiveStack: eff,
      flopBet: pickStr(flopBs['bet'], '50%'),
      flopRaise: pickStr(flopBs['raise'], '2.5x'),
      turnBet: pickStr(turnBs['bet'], '50%'),
      turnRaise: pickStr(turnBs['raise'], '2.5x'),
      riverBet: pickStr(riverBs['bet'], '50%,a'),
      riverRaise: pickStr(riverBs['raise'], '2.5x'),
      maxIterations: (raw['max_iterations'] as int?) ?? 50,
      targetExploitabilityPctPot:
          (raw['target_exploitability_pct_pot'] as num?)?.toDouble() ?? 1.0,
      // Caller chooses how deep to walk. Default (flop) is the smallest
      // dump and the historical behaviour; the trainer surfaces a picker
      // so users can ask for a deeper resolve up-front when they know
      // they'll want it.
      depth: depth,
    );
  }

  /// Cancel an in-flight solve. No-op if nothing is running.
  void cancel() {
    final p = _proc;
    if (p == null) return;
    p.kill(ProcessSignal.sigterm);
    _stderrLines.add('[cancelled]');
    notifyListeners();
  }
}
