/// Data model + JSON parsing for one solved postflop scenario.
///
/// Mirrors the schema produced by `pt-solver --mode tree` (see
/// `validation/src/main.rs`). The JSON is a flat list of nodes, each with a
/// stable `id` derived from the action history; we re-shape that into an
/// id-keyed map at load time so the UI can walk the tree by clicking edges.
library;

import 'dart:convert';
import 'dart:io';

/// A handful of float lists per node — kept as `List<double>` rather than
/// `Float32List` because we re-aggregate them into the 13×13 grid lazily and
/// want the simple List-of-List interface JSON gives us.
class ScenarioNode {
  /// Path id: "" for root, "0", "0/1", "0/1/2".
  final String id;
  /// Raw action indices from root to this node (length == depth).
  final List<int> history;
  /// "action" | "chance_pending" | "terminal".
  final String kind;
  /// Cards on the board (3, 4, or 5 entries — flop + optional turn/river).
  final List<String> board;
  /// Pot at this decision (chips).
  final int pot;
  /// Remaining stacks [oop, ip].
  final List<int> stacks;
  /// Human-readable line from root, e.g. ["Check", "Bet(30)", "Call"].
  final List<String> line;

  // Action-only fields. Empty/null at chance/terminal.
  /// "oop" or "ip"; null at chance/terminal.
  final String? player;
  /// Available action labels at this node.
  final List<String> actions;
  /// Edges to child nodes.
  final List<ChildEdge> children;
  /// Per-hand × per-action strategy (rows sum to 1). Hand index is into
  /// `Scenario.privateCards[player]`.
  final List<List<double>> strategy;
  /// Per-hand × per-action expected values, in chips.
  final List<List<double>> ev;
  /// Per-hand reach probabilities for the player to act. Combos that overlap
  /// the board have weight 0; multiply strategy/ev by these before any
  /// hand-class aggregation.
  final List<double> weights;
  /// Per-hand reach probabilities for the *other* player at this node.
  final List<double> weightsOpp;

  const ScenarioNode({
    required this.id,
    required this.history,
    required this.kind,
    required this.board,
    required this.pot,
    required this.stacks,
    required this.line,
    required this.player,
    required this.actions,
    required this.children,
    required this.strategy,
    required this.ev,
    required this.weights,
    required this.weightsOpp,
  });

  bool get isAction => kind == 'action';
  bool get isTerminal => kind == 'terminal';
  bool get isChancePending => kind == 'chance_pending';
  /// A chance node that the solver dump walked through — has child entries
  /// for each possible turn/river card. Distinct from `chance_pending`,
  /// where the walker stopped.
  bool get isChance => kind == 'chance';

  factory ScenarioNode.fromJson(Map<String, dynamic> j) {
    return ScenarioNode(
      id: j['id'] as String,
      history: (j['history'] as List).cast<int>(),
      kind: j['kind'] as String,
      board: (j['board'] as List).cast<String>(),
      pot: j['pot'] as int,
      stacks: (j['stacks'] as List).cast<int>(),
      line: (j['line'] as List).cast<String>(),
      player: j['player'] as String?,
      actions: (j['actions'] as List?)?.cast<String>() ?? const [],
      children: ((j['children'] as List?) ?? const [])
          .map((e) => ChildEdge.fromJson(e as Map<String, dynamic>))
          .toList(growable: false),
      strategy: _matrix(j['strategy']),
      ev: _matrix(j['ev']),
      weights: _vec(j['weights']),
      weightsOpp: _vec(j['weights_opp']),
    );
  }

  static List<List<double>> _matrix(dynamic raw) {
    if (raw == null) return const [];
    return (raw as List)
        .map((row) => (row as List).map((x) => (x as num).toDouble()).toList(growable: false))
        .toList(growable: false);
  }

  static List<double> _vec(dynamic raw) {
    if (raw == null) return const [];
    return (raw as List).map((x) => (x as num).toDouble()).toList(growable: false);
  }
}

class ChildEdge {
  final int actionIdx;
  final String action;
  final String nodeId;

  const ChildEdge({
    required this.actionIdx,
    required this.action,
    required this.nodeId,
  });

  factory ChildEdge.fromJson(Map<String, dynamic> j) => ChildEdge(
        actionIdx: j['action_idx'] as int,
        action: j['action'] as String,
        nodeId: j['node_id'] as String,
      );
}

/// One solved spot. The whole tree is in memory at once; for the SRP fixture
/// it's ~160 KB, comfortable to hold.
class Scenario {
  /// Display label shown in the scenario picker.
  final String label;
  /// Source path on disk — handy for debugging and "reload" affordances.
  final String sourcePath;
  /// Final solver exploitability, in chips.
  final double exploitability;
  /// Echoed from the input spec.
  final int startingPot;
  final int effectiveStack;
  final List<String> flop;
  final String? turn;
  final String? river;
  /// Combo strings, one list per player. Index into these for `strategy[h]`.
  final List<String> oopCombos;
  final List<String> ipCombos;
  /// id → node, ready for tree navigation. Always contains a root with id "".
  final Map<String, ScenarioNode> nodes;
  /// Convenience: the root node ("").
  final ScenarioNode root;

  /// Sum of OOP weights at the root — i.e., the original range mass before
  /// any actions. Used as the denominator for "% of range surviving" displays.
  /// Note: not literally num_combos; suited combos contribute 1 unit, offsuit
  /// 1 unit, etc. (the solver treats each unblocked combo equally).
  final double oopRootMass;
  final double ipRootMass;
  /// Original `SpotInput` echoed by pt-solver. Kept verbatim so we can copy
  /// bet sizings, iteration counts, and rake when spawning a subgame solve
  /// from a chance_pending node. Null only if the JSON pre-dated the field.
  final Map<String, dynamic>? rawInput;
  /// Set on scenarios produced by on-demand subgame expansion. Tells the UI
  /// "this scenario continues from another scenario" so we can render a
  /// breadcrumb-style back-link. Null for top-level scenarios.
  final ScenarioParent? parent;

  const Scenario({
    required this.label,
    required this.sourcePath,
    required this.exploitability,
    required this.startingPot,
    required this.effectiveStack,
    required this.flop,
    required this.turn,
    required this.river,
    required this.oopCombos,
    required this.ipCombos,
    required this.nodes,
    required this.root,
    required this.oopRootMass,
    required this.ipRootMass,
    this.rawInput,
    this.parent,
  });

  ScenarioNode? nodeById(String id) => nodes[id];

  /// Combo list for a player label ("oop"/"ip"). Empty for unknown.
  List<String> combosFor(String player) =>
      player == 'oop' ? oopCombos : (player == 'ip' ? ipCombos : const []);

  /// Sum of `weights` (or `weightsOpp`) at this node, projected onto OOP.
  /// Returns null at chance/terminal nodes (no weights stored).
  double? oopMassAt(ScenarioNode n) => _massForPlayer(n, 'oop');
  double? ipMassAt(ScenarioNode n) => _massForPlayer(n, 'ip');

  double? _massForPlayer(ScenarioNode n, String player) {
    if (!n.isAction) return null;
    final list = n.player == player ? n.weights : n.weightsOpp;
    if (list.isEmpty) return null;
    var s = 0.0;
    for (final v in list) {
      s += v;
    }
    return s;
  }

  static Future<Scenario> loadFromFile(String path, {String? label}) async {
    final raw = await File(path).readAsString();
    final j = jsonDecode(raw) as Map<String, dynamic>;
    if (j['version'] != 1) {
      throw FormatException(
        'unsupported scenario schema version: ${j['version']} (expected 1)',
      );
    }
    final nodesList = (j['nodes'] as List)
        .map((e) => ScenarioNode.fromJson(e as Map<String, dynamic>))
        .toList(growable: false);
    final byId = {for (final n in nodesList) n.id: n};
    final root = byId[''];
    if (root == null) {
      throw FormatException('scenario has no root node (id "")');
    }
    final priv = j['private_cards'] as Map<String, dynamic>;
    final rawInput = j['input'] is Map<String, dynamic>
        ? j['input'] as Map<String, dynamic>
        : null;
    final parent = j['parent'] is Map<String, dynamic>
        ? ScenarioParent.fromJson(j['parent'] as Map<String, dynamic>)
        : null;
    final scenario = Scenario(
      label: label ?? _labelFromPath(path),
      sourcePath: path,
      exploitability: (j['exploitability'] as num).toDouble(),
      startingPot: j['starting_pot'] as int,
      effectiveStack: j['effective_stack'] as int,
      flop: (j['flop'] as List).cast<String>(),
      turn: j['turn'] as String?,
      river: j['river'] as String?,
      oopCombos: (priv['oop'] as List).cast<String>(),
      ipCombos: (priv['ip'] as List).cast<String>(),
      nodes: byId,
      root: root,
      oopRootMass: 0.0,
      ipRootMass: 0.0,
      rawInput: rawInput,
      parent: parent,
    );
    return Scenario(
      label: scenario.label,
      sourcePath: scenario.sourcePath,
      exploitability: scenario.exploitability,
      startingPot: scenario.startingPot,
      effectiveStack: scenario.effectiveStack,
      flop: scenario.flop,
      turn: scenario.turn,
      river: scenario.river,
      oopCombos: scenario.oopCombos,
      ipCombos: scenario.ipCombos,
      nodes: scenario.nodes,
      root: scenario.root,
      oopRootMass: scenario.oopMassAt(root) ?? 0.0,
      ipRootMass: scenario.ipMassAt(root) ?? 0.0,
      rawInput: scenario.rawInput,
      parent: scenario.parent,
    );
  }

  static String _labelFromPath(String path) {
    final base = path.split('/').last;
    return base.endsWith('.json') ? base.substring(0, base.length - 5) : base;
  }
}

/// Optional "where this scenario came from" metadata, set when a scenario
/// was generated by on-demand expansion of a chance_pending node.
class ScenarioParent {
  /// Filename / label of the parent scenario.
  final String parentLabel;
  /// Absolute filesystem path to the parent scenario's JSON.
  final String parentSourcePath;
  /// Node id within the parent that the user expanded from (the
  /// chance_pending). Useful for rendering "← Back to {parent} at {line}".
  final String parentNodeId;
  /// Human-readable line up to the chance_pending in the parent.
  final List<String> parentLine;
  /// The card the user picked at the chance to spawn this subgame.
  final String pickedCard;

  const ScenarioParent({
    required this.parentLabel,
    required this.parentSourcePath,
    required this.parentNodeId,
    required this.parentLine,
    required this.pickedCard,
  });

  Map<String, dynamic> toJson() => {
        'parent_label': parentLabel,
        'parent_source_path': parentSourcePath,
        'parent_node_id': parentNodeId,
        'parent_line': parentLine,
        'picked_card': pickedCard,
      };

  factory ScenarioParent.fromJson(Map<String, dynamic> j) => ScenarioParent(
        parentLabel: j['parent_label'] as String,
        parentSourcePath: j['parent_source_path'] as String,
        parentNodeId: j['parent_node_id'] as String,
        parentLine: (j['parent_line'] as List).cast<String>(),
        pickedCard: j['picked_card'] as String,
      );
}
