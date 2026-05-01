/// Per-app registry of loaded TorchScript models. Keyed by file path so the
/// same checkpoint file isn't loaded twice; labels default to the file's
/// basename (without `.pt`) so dropdowns stay short.
///
/// `ChangeNotifier` so the UI rebuilds the per-seat agent dropdown when a
/// new model lands.
library;

import 'dart:io';

import 'package:flutter/foundation.dart';

import '../ffi/engine.dart';

class LoadedModel {
  final String path;
  final String label;
  final ModelHandle handle;
  const LoadedModel({required this.path, required this.label, required this.handle});
}

class ModelRegistry extends ChangeNotifier {
  final PokerEngine engine;
  final Map<String, LoadedModel> _byPath = {};

  ModelRegistry(this.engine);

  List<LoadedModel> get models => List.unmodifiable(_byPath.values);

  LoadedModel? byLabel(String label) {
    for (final m in _byPath.values) {
      if (m.label == label) return m;
    }
    return null;
  }

  /// Load a model from disk. Returns the loaded entry on success, null if
  /// already loaded (idempotent), throws on file-not-found / load failure.
  LoadedModel load(String path) {
    if (!File(path).existsSync()) {
      throw StateError('Model file not found: $path');
    }
    if (_byPath.containsKey(path)) return _byPath[path]!;
    final handle = ModelHandle.load(engine, path);
    if (handle == null) {
      throw StateError(
          'pt_model_load returned null — wrong .pt format, or '
          'PT_FFI_INFER not compiled in?');
    }
    final entry = LoadedModel(
      path: path,
      label: _shortLabel(path, _byPath.values.map((e) => e.label).toSet()),
      handle: handle,
    );
    _byPath[path] = entry;
    notifyListeners();
    return entry;
  }

  /// Drop a model. Disposes the underlying handle. Caller is responsible
  /// for not having any seat agent still holding it (currently the
  /// inspector reassigns to Human before unloading).
  void unload(String path) {
    final entry = _byPath.remove(path);
    if (entry != null) {
      entry.handle.dispose();
      notifyListeners();
    }
  }

  @override
  void dispose() {
    for (final e in _byPath.values) {
      e.handle.dispose();
    }
    _byPath.clear();
    super.dispose();
  }

  /// "ckpt_180000" from "/foo/bar/runs/.../ckpt_180000.pt". If two checkpoints
  /// from different runs share a basename, disambiguate with the parent dir.
  String _shortLabel(String path, Set<String> taken) {
    final parts = path.split(Platform.pathSeparator);
    final base = parts.last.replaceAll(RegExp(r'\.pt$'), '');
    if (!taken.contains(base)) return base;
    if (parts.length >= 2) {
      final withParent = '${parts[parts.length - 2]}/$base';
      if (!taken.contains(withParent)) return withParent;
    }
    // Last resort: append a counter.
    int i = 2;
    while (taken.contains('$base#$i')) {
      i++;
    }
    return '$base#$i';
  }
}
