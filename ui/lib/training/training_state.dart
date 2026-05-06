/// Tracks which node the user is currently viewing inside a loaded scenario.
/// Exposes navigation primitives (descend by action, ascend, jump to root)
/// that the breadcrumb and action buttons hook into.
library;

import 'package:flutter/foundation.dart';

import 'scenario.dart';

class TrainingState extends ChangeNotifier {
  Scenario? _scenario;
  String _currentId = '';

  Scenario? get scenario => _scenario;
  String get currentId => _currentId;
  ScenarioNode? get currentNode => _scenario?.nodeById(_currentId);

  void setScenario(Scenario? s) {
    _scenario = s;
    _currentId = '';
    notifyListeners();
  }

  void descend(ChildEdge edge) {
    if (_scenario?.nodeById(edge.nodeId) == null) return;
    _currentId = edge.nodeId;
    notifyListeners();
  }

  /// Jump to a specific node id (e.g. clicking an item in the breadcrumb).
  void jumpTo(String id) {
    if (_scenario?.nodeById(id) == null) return;
    _currentId = id;
    notifyListeners();
  }

  void backToRoot() {
    _currentId = '';
    notifyListeners();
  }

  /// Walk back up to the parent node by trimming the last segment of the id.
  /// No-op at root.
  void up() {
    if (_currentId.isEmpty) return;
    final i = _currentId.lastIndexOf('/');
    final parent = i < 0 ? '' : _currentId.substring(0, i);
    if (_scenario?.nodeById(parent) == null) return;
    _currentId = parent;
    notifyListeners();
  }

  /// Breadcrumb entries from root to current node, inclusive. Each entry
  /// carries the id (for jumping) and the label that led to that node.
  /// The root has label "Root".
  List<BreadcrumbEntry> breadcrumb() {
    final n = currentNode;
    if (n == null) return const [];
    final out = <BreadcrumbEntry>[BreadcrumbEntry(id: '', label: 'Root')];
    final s = _scenario!;
    String pathId = '';
    for (var i = 0; i < n.history.length; i++) {
      pathId = pathId.isEmpty ? '${n.history[i]}' : '$pathId/${n.history[i]}';
      out.add(BreadcrumbEntry(
        id: pathId,
        label: i < n.line.length ? n.line[i] : '?',
        playerHint: _playerAtDepth(s, i),
      ));
    }
    return out;
  }

  /// Look up which player acted at depth `i` in the line. We do this by
  /// walking the parent's `player` field — at depth 0 that's the root's
  /// player, at depth 1 it's root.children[..].nodeId's player, etc.
  String? _playerAtDepth(Scenario s, int i) {
    String pid = '';
    for (var k = 0; k <= i; k++) {
      final node = s.nodeById(pid);
      if (node == null) return null;
      if (k == i) return node.player;
      // descend by the actual history index
      final histIdx = currentNode!.history[k];
      pid = pid.isEmpty ? '$histIdx' : '$pid/$histIdx';
    }
    return null;
  }
}

class BreadcrumbEntry {
  final String id;
  final String label;
  final String? playerHint;
  const BreadcrumbEntry({
    required this.id,
    required this.label,
    this.playerHint,
  });
}
