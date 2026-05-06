/// Discovers scenario JSON files on disk and exposes them as a navigable list.
///
/// v1: scans a single hard-coded directory (validation_runs/scenarios/) for
/// *.json files. Each entry is loaded lazily on selection; listing is cheap
/// (just a directory listing). When the batch-pre-solve driver lands later,
/// it'll write a manifest.json that this loader can consume instead of a raw
/// glob — but the API on this class won't change.
library;

import 'dart:io';

import 'package:flutter/foundation.dart';

import 'scenario.dart';

class ScenarioEntry {
  /// Display name (filename without extension, for now).
  final String label;
  /// Full path on disk.
  final String path;

  const ScenarioEntry({required this.label, required this.path});

  /// True if this looks like an on-demand subgame expansion (named
  /// `<parent>__<card>_<hash>.json`). Doesn't open the file — the picker
  /// uses this to render a distinct icon without paying the parse cost.
  bool get isSubgame => label.contains('__');
}

class ScenarioLibrary extends ChangeNotifier {
  final List<String> _searchRoots;
  List<ScenarioEntry> _entries = const [];
  Scenario? _loaded;
  String? _loadError;
  bool _loading = false;

  ScenarioLibrary({List<String>? searchRoots})
      : _searchRoots = searchRoots ?? _defaultRoots();

  List<ScenarioEntry> get entries => _entries;
  Scenario? get loaded => _loaded;
  String? get loadError => _loadError;
  bool get loading => _loading;

  static List<String> _defaultRoots() {
    final home = Platform.environment['HOME'] ?? '';
    return [
      '/home/elucterio/Poker/PokerTrainer/validation_runs/scenarios',
      '$home/Poker/PokerTrainer/validation_runs/scenarios',
      // Relative to CWD too — convenient when launching from the project root.
      'validation_runs/scenarios',
    ];
  }

  /// Refresh the directory listing. Cheap; safe to call from a button.
  /// Not called automatically — the picker only shows scenarios that the
  /// user has explicitly opened or that the solver has written this session.
  /// Kept available for callers that want to re-populate from disk on demand.
  void rescan() {
    final out = <ScenarioEntry>[];
    final seen = <String>{};
    for (final root in _searchRoots) {
      final dir = Directory(root);
      if (!dir.existsSync()) continue;
      for (final ent in dir.listSync()) {
        if (ent is! File) continue;
        final p = ent.path;
        if (!p.endsWith('.json')) continue;
        if (!seen.add(p)) continue;
        final base = p.split(Platform.pathSeparator).last;
        out.add(ScenarioEntry(
          label: base.endsWith('.json') ? base.substring(0, base.length - 5) : base,
          path: p,
        ));
      }
    }
    out.sort((a, b) => a.label.compareTo(b.label));
    _entries = out;
    notifyListeners();
  }

  /// Add (or re-add) a single scenario entry by path. Idempotent — if the
  /// path is already in the list, returns the existing entry; otherwise
  /// derives a label from the filename and appends. Used by Open… and the
  /// solve / subgame-expand flows so the picker shows what the user has
  /// touched this session, not the whole scenarios/ directory.
  ScenarioEntry addEntry(String path) {
    for (final existing in _entries) {
      if (existing.path == path) return existing;
    }
    final base = path.split(Platform.pathSeparator).last;
    final label = base.endsWith('.json')
        ? base.substring(0, base.length - 5)
        : base;
    final entry = ScenarioEntry(label: label, path: path);
    _entries = [..._entries, entry]
      ..sort((a, b) => a.label.compareTo(b.label));
    notifyListeners();
    return entry;
  }

  Future<void> load(ScenarioEntry e) async {
    _loading = true;
    _loadError = null;
    notifyListeners();
    try {
      _loaded = await Scenario.loadFromFile(e.path, label: e.label);
      _loadError = null;
    } catch (err) {
      _loaded = null;
      _loadError = err.toString();
    } finally {
      _loading = false;
      notifyListeners();
    }
  }
}
