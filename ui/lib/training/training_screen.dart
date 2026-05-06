/// Training tab: scenario library on the left, 13×13 hand-class strategy
/// chart in the middle, board + action tree + summary on the right.
library;

import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';

import 'scenario.dart';
import 'scenario_library.dart';
import 'solver_runner.dart';
import 'trainer/trainer_screen.dart';
import 'training_state.dart';
import 'widgets/chance_picker.dart';
import 'widgets/combo_detail.dart';
import 'widgets/hand_grid.dart';
import 'widgets/node_panel.dart';
import 'widgets/scenario_picker.dart';
import 'widgets/spot_config_dialog.dart';

const XTypeGroup _scenarioTypeGroup = XTypeGroup(
  label: 'Solved scenarios',
  extensions: <String>['json'],
);

/// Default starting directory for the file picker. Falls back gracefully if
/// the canonical scenarios dir doesn't exist (e.g. on a fresh checkout).
String? _defaultPickerDir() {
  for (final candidate in [
    '/home/elucterio/Poker/PokerTrainer/validation_runs/scenarios',
    '${Platform.environment['HOME'] ?? ''}/Poker/PokerTrainer/validation_runs/scenarios',
  ]) {
    if (Directory(candidate).existsSync()) return candidate;
  }
  return null;
}

class TrainingScreen extends StatefulWidget {
  const TrainingScreen({super.key});

  @override
  State<TrainingScreen> createState() => _TrainingScreenState();
}

/// Inspect: passive walk through the action tree. Train: interactive play
/// against the solver. Mode is a per-tab toggle; switching modes does not
/// reload the scenario.
enum _Mode { inspect, train }

class _TrainingScreenState extends State<TrainingScreen> {
  final ScenarioLibrary _library = ScenarioLibrary();
  final TrainingState _state = TrainingState();
  final SolverRunner _solver = SolverRunner();
  /// Last spot configuration used in the New-solve dialog. Persists across
  /// dialog opens so the user doesn't retype after a successful solve.
  final SpotConfig _lastSpotConfig = SpotConfig.preset();
  String? _selectedPath;
  _Mode _mode = _Mode.inspect;

  @override
  void initState() {
    super.initState();
    // No auto-scan: the left rail starts empty. Users add scenarios via
    // Open… or by running a new solve. Keeping the rail to "what the user
    // has touched this session" avoids drowning them in stale runs from
    // validation_runs/scenarios/.
    _library.addListener(_onLibChanged);
  }

  @override
  void dispose() {
    _library.removeListener(_onLibChanged);
    _library.dispose();
    _state.dispose();
    _solver.dispose();
    super.dispose();
  }

  Future<void> _openNewSolveDialog() async {
    await showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => SpotConfigDialog(
        runner: _solver,
        initial: _lastSpotConfig,
        onSolved: (path) async {
          // Add the freshly-solved file to the picker (no full rescan), then
          // auto-select it so the user sees the result immediately.
          final entry = _library.addEntry(path);
          _selectAndLoad(entry);
        },
      ),
    );
  }

  /// Open... — let the user pick a saved scenario JSON anywhere on disk and
  /// add it to the left rail, then load it.
  Future<void> _openFile() async {
    final XFile? picked = await openFile(
      acceptedTypeGroups: const [_scenarioTypeGroup],
      initialDirectory: _defaultPickerDir(),
    );
    if (picked == null) return;
    final entry = _library.addEntry(picked.path);
    _selectAndLoad(entry);
  }

  /// Save as... — copy the currently-loaded scenario's source file to a
  /// user-chosen path. The active scenario keeps pointing at its original
  /// path; the saved copy is independent and can be reopened later.
  Future<void> _saveAs() async {
    final s = _state.scenario;
    if (s == null) return;
    final src = File(s.sourcePath);
    if (!src.existsSync()) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: const Color(0xFF7A2A2A),
          content: Text('Source file not found: ${s.sourcePath}'),
        ),
      );
      return;
    }
    final FileSaveLocation? loc = await getSaveLocation(
      acceptedTypeGroups: const [_scenarioTypeGroup],
      initialDirectory: _defaultPickerDir(),
      suggestedName: '${s.label}.json',
    );
    if (loc == null) return;
    try {
      // Ensure .json extension — the picker on Linux/GTK doesn't enforce it.
      final dest = loc.path.endsWith('.json') ? loc.path : '${loc.path}.json';
      await src.copy(dest);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: const Color(0xFF1F4429),
          content: Text('Saved to $dest'),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: const Color(0xFF7A2A2A),
          content: Text('Save failed: $e'),
        ),
      );
    }
  }

  void _onLibChanged() {
    if (_library.loaded != null && _library.loaded != _state.scenario) {
      _state.setScenario(_library.loaded);
    } else if (_library.loaded == null && _library.loadError != null) {
      // Surface error in a snack — do it post-build to avoid context issues.
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            backgroundColor: const Color(0xFF7A2A2A),
            content: Text('Load failed: ${_library.loadError}'),
            duration: const Duration(seconds: 6),
          ),
        );
      });
    }
    setState(() {});
  }

  void _selectAndLoad(ScenarioEntry e) {
    setState(() {
      _selectedPath = e.path;
      // Always return to Inspect when changing scenario — Train mode keeps a
      // session pointer that's tied to a specific scenario instance, so
      // switching scenarios mid-train would silently invalidate it.
      _mode = _Mode.inspect;
    });
    _library.load(e);
  }

  void _setMode(_Mode m) {
    if (_mode == m) return;
    setState(() => _mode = m);
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _state,
      builder: (context, _) => Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _toolbar(),
            const SizedBox(height: 12),
            Expanded(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  SizedBox(
                    width: 220,
                    child: ScenarioPicker(
                      library: _library,
                      selectedPath: _selectedPath,
                      onSelect: _selectAndLoad,
                    ),
                  ),
                  const SizedBox(width: 14),
                  Expanded(child: _modeBody()),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Centre+right area. Inspect mode = grid + NodePanel. Train mode =
  /// the trainer's full layout.
  Widget _modeBody() {
    if (_mode == _Mode.train) {
      final s = _state.scenario;
      if (s == null) {
        return const Center(
          child: Text(
            'Load a scenario first to start training.',
            style: TextStyle(
              color: Color(0x99EAE6D9),
              fontStyle: FontStyle.italic,
              fontSize: 13,
            ),
          ),
        );
      }
      // The trainer reads from `_state.scenario` at the moment of mount.
      // If a subgame is resolved internally the trainer tracks the active
      // scenario itself; the inspector view doesn't need to follow.
      return TrainerScreen(
        key: ValueKey('trainer-${s.sourcePath}'),
        initialScenario: s,
        solver: _solver,
        library: _library,
        onQuit: () => _setMode(_Mode.inspect),
      );
    }
    return Row(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Expanded(child: _gridArea()),
        const SizedBox(width: 14),
        SizedBox(
          width: 340,
          child: NodePanel(state: _state),
        ),
      ],
    );
  }

  Widget _toolbar() {
    final s = _state.scenario;
    final parent = s?.parent;
    final canSave = s != null;
    final canTrain = s != null;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            FilledButton.icon(
              icon: const Icon(Icons.add, size: 18),
              label: const Text('New solve…'),
              onPressed: _openNewSolveDialog,
            ),
            const SizedBox(width: 8),
            OutlinedButton.icon(
              icon: const Icon(Icons.folder_open, size: 18),
              label: const Text('Open…'),
              onPressed: _openFile,
            ),
            const SizedBox(width: 8),
            OutlinedButton.icon(
              icon: const Icon(Icons.save_alt, size: 18),
              label: const Text('Save as…'),
              onPressed: canSave ? _saveAs : null,
            ),
            const SizedBox(width: 12),
            // Mode switch — disabled until a scenario is loaded. The two
            // pills act as a segmented control; current mode is filled.
            _ModeSwitch(
              mode: _mode,
              enabled: canTrain,
              onChange: _setMode,
            ),
            const SizedBox(width: 12),
            if (s != null && parent == null)
              Expanded(
                child: Row(
                  children: [
                    const Icon(
                      Icons.account_tree_outlined,
                      size: 16,
                      color: Color(0xFF6FB3DC),
                    ),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        'Root solve · ${s.label}',
                        style: const TextStyle(
                          color: Color(0xCCEAE6D9),
                          fontSize: 13,
                          fontFamily: 'monospace',
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
        if (parent != null) ...[
          const SizedBox(height: 8),
          _SubgameBanner(
            parent: parent,
            onBack: () {
              // Add-or-reuse the parent file in the library, then load it.
              // addEntry() is idempotent so this also works if the user had
              // already opened the parent earlier in the session.
              final entry = _library.addEntry(parent.parentSourcePath);
              _selectAndLoad(entry);
            },
          ),
        ],
      ],
    );
  }

  Widget _gridArea() {
    final s = _state.scenario;
    final n = _state.currentNode;
    if (_library.loading) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 12),
            Text(
              'Loading scenario…',
              style: TextStyle(color: Color(0x99EAE6D9), fontSize: 13),
            ),
          ],
        ),
      );
    }
    if (s == null || n == null) {
      return const Center(
        child: Text(
          'No scenario loaded.',
          style: TextStyle(
            color: Color(0x99EAE6D9),
            fontStyle: FontStyle.italic,
            fontSize: 14,
          ),
        ),
      );
    }
    final combos = n.player == null ? const <String>[] : s.combosFor(n.player!);
    final Widget body;
    if (n.isChance || n.isChancePending) {
      body = ChancePicker(
        state: _state,
        node: n,
        solver: n.isChancePending ? _solver : null,
        onSubgameSolved: (path) {
          // Same pattern as the New-solve callback: just register the new
          // file in the library and load it. No directory rescan.
          final entry = _library.addEntry(path);
          _selectAndLoad(entry);
        },
      );
    } else {
      body = HandGrid(
        node: n,
        combos: combos,
        onCellTap: (cell) => showDialog(
          context: context,
          builder: (_) => ComboDetailDialog(
            node: n,
            combos: combos,
            cell: cell,
          ),
        ),
      );
    }
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF24282B),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF4A4E52), width: 1),
      ),
      child: body,
    );
  }
}

/// Amber banner shown when the current scenario was produced by on-demand
/// chance_pending expansion. Tells the user this is a fresh subgame
/// equilibrium computed with conditional ranges — *not* a navigation into
/// an existing root-solve dump. Strategies here will differ subtly from the
/// corresponding node in a global solve.
class _SubgameBanner extends StatelessWidget {
  final ScenarioParent parent;
  final VoidCallback onBack;

  const _SubgameBanner({required this.parent, required this.onBack});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0x33D4B43F), // amber wash
        border: Border.all(color: const Color(0xFFD4B43F), width: 1.4),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          const Icon(
            Icons.alt_route,
            size: 20,
            color: Color(0xFFD4B43F),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Text(
                      'SUBGAME RE-SOLVE',
                      style: TextStyle(
                        color: Color(0xFFD4B43F),
                        fontSize: 13,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.2,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Flexible(
                      child: Text(
                        'continued from ${parent.parentLine.join(" › ")} '
                        '→ ${parent.pickedCard}',
                        style: const TextStyle(
                          color: Color(0xFFEAE6D9),
                          fontSize: 13,
                          fontFamily: 'monospace',
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 3),
                const Text(
                  'Fresh equilibrium for this branch only — solver re-ran '
                  'with the conditional ranges at this node. Strategies may '
                  'differ slightly from a global root solve.',
                  style: TextStyle(
                    color: Color(0xCCEAE6D9),
                    fontSize: 12,
                    height: 1.35,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          OutlinedButton.icon(
            icon: const Icon(Icons.arrow_back, size: 14),
            label: Text(
              'Back to ${parent.parentLabel}',
              style: const TextStyle(fontSize: 12),
            ),
            style: OutlinedButton.styleFrom(
              foregroundColor: const Color(0xFFEAE6D9),
              side: const BorderSide(color: Color(0xFFD4B43F), width: 1),
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            ),
            onPressed: onBack,
          ),
        ],
      ),
    );
  }
}

/// Inspect / Train segmented control. Visual style matches the rest of the
/// toolbar's outlined buttons; the active mode is filled.
class _ModeSwitch extends StatelessWidget {
  final _Mode mode;
  final bool enabled;
  final ValueChanged<_Mode> onChange;

  const _ModeSwitch({
    required this.mode,
    required this.enabled,
    required this.onChange,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(2),
      decoration: BoxDecoration(
        color: const Color(0xFF1B1E20),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: const Color(0xFF45525A), width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _pill(
            label: 'Inspect',
            icon: Icons.search,
            active: mode == _Mode.inspect,
            onTap: enabled ? () => onChange(_Mode.inspect) : null,
          ),
          _pill(
            label: 'Train',
            icon: Icons.school,
            active: mode == _Mode.train,
            onTap: enabled ? () => onChange(_Mode.train) : null,
          ),
        ],
      ),
    );
  }

  Widget _pill({
    required String label,
    required IconData icon,
    required bool active,
    required VoidCallback? onTap,
  }) {
    final fg = onTap == null
        ? const Color(0x55EAE6D9)
        : (active ? const Color(0xFFEAE6D9) : const Color(0xCCEAE6D9));
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: active ? const Color(0xFF1B2D40) : Colors.transparent,
          borderRadius: BorderRadius.circular(4),
        ),
        child: Row(
          children: [
            Icon(icon, size: 14, color: fg),
            const SizedBox(width: 4),
            Text(
              label,
              style: TextStyle(
                color: fg,
                fontFamily: 'monospace',
                fontSize: 12,
                fontWeight: active ? FontWeight.w800 : FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
