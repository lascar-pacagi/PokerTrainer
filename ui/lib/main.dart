/// PokerTrainer interactive policy inspector — entry point.
///
/// Loads libpokertrainer.so + initializes hand-eval tables, then opens the
/// inspector window. One GameSession lives at the root and feeds every
/// screen widget via ChangeNotifier ticks.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';

import 'ffi/actions.dart';
import 'ffi/engine.dart';
import 'game/game_session.dart';
import 'game/model_registry.dart';
import 'play/play_screen.dart';
import 'training/training_screen.dart';
import 'widgets/action_bar.dart';
import 'widgets/history_strip.dart';
import 'widgets/table_view.dart';

void main() {
  // Lazy: errors here surface as a red screen, which is fine for dev.
  final engine = PokerEngine.init();
  runApp(InspectorApp(engine: engine));
}

class InspectorApp extends StatelessWidget {
  final PokerEngine engine;
  const InspectorApp({super.key, required this.engine});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PokerTrainer · HU NLHE Inspector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF1B1F22),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2D5B7C),
          brightness: Brightness.dark,
        ),
        textTheme: Typography.whiteMountainView,
      ),
      home: InspectorHome(engine: engine),
      // Default text scaling. Each tab overrides this with its own scaler:
      // Training picks a larger scale (the chart/text panels were designed
      // for it), Inspector keeps closer to the design baseline so the action
      // bar fits without scrolling. CardView wraps itself in
      // TextScaler.noScaling so its hand-tuned glyph positions never shift.
      builder: (context, child) {
        final mq = MediaQuery.of(context);
        return MediaQuery(
          data: mq.copyWith(textScaler: const TextScaler.linear(1.0)),
          child: child!,
        );
      },
    );
  }
}

class InspectorHome extends StatefulWidget {
  final PokerEngine engine;
  const InspectorHome({super.key, required this.engine});

  @override
  State<InspectorHome> createState() => _InspectorHomeState();
}

enum _Tab { inspector, training, play }

class _InspectorHomeState extends State<InspectorHome> {
  late final GameSession  _session;
  late final ModelRegistry _models;
  bool _revealAllHoles = true;
  _Tab _tab = _Tab.inspector;

  @override
  void initState() {
    super.initState();
    _session = GameSession(widget.engine);
    _models  = ModelRegistry(widget.engine);
  }

  @override
  void dispose() {
    _session.dispose();
    _models.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: Listenable.merge([_session, _models]),
      builder: (context, _) => Scaffold(
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                _tabSwitcher(),
                const SizedBox(height: 14),
                if (_tab == _Tab.training)
                  Expanded(
                    child: MediaQuery(
                      // Training tab gets the bigger fonts — the chart cells
                      // and dense panels were designed at ~1.32×.
                      data: MediaQuery.of(context).copyWith(
                        textScaler: const TextScaler.linear(1.32),
                      ),
                      child: const TrainingScreen(),
                    ),
                  )
                else if (_tab == _Tab.play)
                  Expanded(
                    child: MediaQuery(
                      data: MediaQuery.of(context).copyWith(
                        textScaler: const TextScaler.linear(1.15),
                      ),
                      child: PlayScreen(engine: widget.engine, models: _models),
                    ),
                  )
                else ...[
                _topBar(),
                const SizedBox(height: 10),
                _agentRow(),
                const SizedBox(height: 18),
                Expanded(
                  child: MediaQuery(
                    // Inspector layout was tuned closer to design size; nudge
                    // up modestly so the action bar still fits in one screen.
                    data: MediaQuery.of(context).copyWith(
                      textScaler: const TextScaler.linear(1.15),
                    ),
                    child: Row(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      // Left rail: history.
                      SizedBox(
                        width: 280,
                        child: HistoryStrip(history: _session.history),
                      ),
                      const SizedBox(width: 18),
                      // Center: table + action bar.
                      Expanded(
                        child: SingleChildScrollView(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.center,
                            children: [
                              ConstrainedBox(
                                constraints: const BoxConstraints(maxWidth: 880),
                                child: TableView(
                                  session: _session,
                                  revealAllHoles: _revealAllHoles,
                                ),
                              ),
                              const SizedBox(height: 18),
                              ConstrainedBox(
                                constraints: const BoxConstraints(maxWidth: 880),
                                child: ActionBar(session: _session),
                              ),
                            ],
                          ),
                        ),
                      ),
                      // Right rail: reserved for the strategy panel (lights up
                      // once a model is loaded). Empty placeholder until then
                      // so the layout is stable when it arrives.
                      const SizedBox(width: 18),
                      SizedBox(
                        width: 240,
                        child: _StrategyPlaceholder(session: _session),
                      ),
                    ],
                  ),
                  ),
                ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _tabSwitcher() {
    return Row(
      children: [
        for (final t in _Tab.values)
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: _TabPill(
              label: switch (t) {
                _Tab.inspector => 'Inspector',
                _Tab.training => 'Training',
                _Tab.play => 'Play',
              },
              icon: switch (t) {
                _Tab.inspector => Icons.casino,
                _Tab.training => Icons.school,
                _Tab.play => Icons.sports_esports,
              },
              selected: _tab == t,
              onTap: () => setState(() => _tab = t),
            ),
          ),
      ],
    );
  }

  Widget _topBar() {
    final seedTxt = _session.currentSeed == null
        ? '—'
        : '0x${_session.currentSeed!.toRadixString(16).padLeft(8, "0")}';
    return Row(
      children: [
        Text(
          'PokerTrainer Inspector',
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
            fontWeight: FontWeight.w700,
            letterSpacing: 0.4,
            fontSize: 22,
          ),
        ),
        const SizedBox(width: 22),
        Text(
          'Hand #${_session.handCount}  ·  seed $seedTxt',
          style: const TextStyle(
            color: Color(0xCCEAE6D9),
            fontFamily: 'monospace',
            fontSize: 16,
          ),
        ),
        const Spacer(),
        IconButton(
          tooltip: _revealAllHoles ? 'Hide opponent cards' : 'Reveal all holes',
          icon: Icon(
            _revealAllHoles ? Icons.visibility : Icons.visibility_off,
            size: 22,
          ),
          onPressed: () => setState(() => _revealAllHoles = !_revealAllHoles),
        ),
        const SizedBox(width: 8),
        IconButton(
          tooltip: _session.autoStep
              ? 'Auto-step is ON — model plays through. Click to switch to '
                'manual step (so you can read the strategy panel before each '
                'model action).'
              : 'Auto-step is OFF — model waits for you. Click to resume '
                'auto-stepping.',
          icon: Icon(
            _session.autoStep ? Icons.fast_forward : Icons.skip_next,
            size: 22,
            color: _session.autoStep
                ? const Color(0xFFEAE6D9)
                : const Color(0xFFFFD24A),
          ),
          onPressed: () => _session.setAutoStep(!_session.autoStep),
        ),
        const SizedBox(width: 8),
        FilledButton.icon(
          icon: const Icon(Icons.shuffle, size: 20),
          label: const Text('Deal random', style: TextStyle(fontSize: 15)),
          onPressed: _session.dealRandom,
        ),
        const SizedBox(width: 10),
        FilledButton.tonalIcon(
          icon: const Icon(Icons.replay, size: 20),
          label: const Text('Re-seed', style: TextStyle(fontSize: 15)),
          onPressed: _showSeedDialog,
        ),
      ],
    );
  }

  /// Per-seat agent dropdowns + Load-model button. Lives just under the top
  /// bar so the controls that change *who* is playing are visible without
  /// scrolling.
  Widget _agentRow() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF272A2D),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFF4A4E52), width: 1),
      ),
      child: Row(
        children: [
          _agentPicker(Player.sb),
          const SizedBox(width: 28),
          _agentPicker(Player.bb),
          const Spacer(),
          if (_models.models.isNotEmpty) ...[
            Text(
              '${_models.models.length} model${_models.models.length == 1 ? "" : "s"} loaded',
              style: const TextStyle(
                color: Color(0xAAEAE6D9),
                fontSize: 13,
                fontStyle: FontStyle.italic,
              ),
            ),
            const SizedBox(width: 14),
          ],
          OutlinedButton.icon(
            icon: const Icon(Icons.upload_file, size: 18),
            label: const Text('Load model…', style: TextStyle(fontSize: 14)),
            onPressed: _engineHasInference() ? _pickAndLoadModel : null,
          ),
        ],
      ),
    );
  }

  Widget _agentPicker(Player p) {
    final agent = _session.agentFor(p);
    final currentLabel = agent is HumanAgent
        ? 'Human'
        : (agent is ModelAgent ? agent.label : 'Human');

    final items = <DropdownMenuItem<String>>[
      const DropdownMenuItem(value: 'Human', child: Text('Human')),
      for (final m in _models.models)
        DropdownMenuItem(
          value: m.label,
          child: Text(m.label, overflow: TextOverflow.ellipsis),
        ),
    ];

    return Row(
      children: [
        Text(
          '${p.shortLabel}:',
          style: TextStyle(
            color: p == Player.sb
                ? const Color(0xFF6FB3DC)
                : const Color(0xFFDCBE6F),
            fontFamily: 'monospace',
            fontWeight: FontWeight.w800,
            fontSize: 16,
          ),
        ),
        const SizedBox(width: 8),
        ConstrainedBox(
          constraints: const BoxConstraints(minWidth: 160, maxWidth: 240),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: currentLabel,
              isDense: true,
              dropdownColor: const Color(0xFF34383B),
              style: const TextStyle(
                color: Color(0xFFEAE6D9),
                fontSize: 15,
              ),
              items: items,
              onChanged: (v) {
                if (v == null) return;
                if (v == 'Human') {
                  _session.setAgent(p, const HumanAgent());
                } else {
                  final m = _models.byLabel(v);
                  if (m != null) {
                    _session.setAgent(p, ModelAgent(m.handle, label: m.label));
                  }
                }
              },
            ),
          ),
        ),
      ],
    );
  }

  bool _engineHasInference() => widget.engine.modelAvailable;

  Future<void> _pickAndLoadModel() async {
    // Default to the trainer's runs dir if it exists. file_selector ignores
    // unknown directories silently.
    String? initial;
    for (final candidate in [
      '/home/elucterio/Poker/PokerTrainer/runs',
      '${Platform.environment['HOME'] ?? ''}/Poker/PokerTrainer/runs',
    ]) {
      if (Directory(candidate).existsSync()) {
        initial = candidate;
        break;
      }
    }

    const typeGroup = XTypeGroup(
      label: 'TorchScript checkpoints',
      extensions: <String>['pt'],
    );
    final XFile? file = await openFile(
      acceptedTypeGroups: const [typeGroup],
      initialDirectory: initial,
    );
    if (file == null) return;

    try {
      final entry = _models.load(file.path);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Loaded "${entry.label}" — assign it via the SB/BB dropdown.'),
          duration: const Duration(seconds: 4),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: const Color(0xFF7A2A2A),
          content: Text('Load failed: $e'),
          duration: const Duration(seconds: 6),
        ),
      );
    }
  }

  Future<void> _showSeedDialog() async {
    final controller = TextEditingController(
      text: (_session.currentSeed ?? 0xC0FFEE).toRadixString(16),
    );
    final v = await showDialog<int?>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Deal from hex seed'),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: const InputDecoration(
            prefixText: '0x',
            hintText: 'c0ffee',
          ),
          style: const TextStyle(fontFamily: 'monospace'),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, null),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              try {
                final seed = int.parse(controller.text.trim(), radix: 16);
                Navigator.pop(ctx, seed);
              } catch (_) {
                Navigator.pop(ctx, null);
              }
            },
            child: const Text('Deal'),
          ),
        ],
      ),
    );
    if (v != null) _session.dealWithSeed(v);
  }
}

/// Strategy panel: empty card until a model is loaded, then paints either
/// Q-value bars (DMC checkpoints) or probability bars (CFR checkpoints).
///
/// Mode is auto-detected from `session.strategy.qValues`:
///   * if every value is in [0, 1] AND they sum to ~1 → "POLICY π" mode
///     (CFR PolicyNet output; bars scaled absolutely 0..1, labeled %)
///   * otherwise                                       → "Q-VALUES" mode
///     (DMC scalar EVs in BB; bars scaled by min-max range, labeled as floats)
///
/// Both modes sort by descending value (so the argmax floats to the top) and
/// highlight the argmax row with a gold star.
class _StrategyPlaceholder extends StatelessWidget {
  final GameSession session;
  const _StrategyPlaceholder({required this.session});

  @override
  Widget build(BuildContext context) {
    final hasStrategy = session.strategy != null;
    final isProbability = hasStrategy && _isProbabilityDistribution(
      session.strategy!.qValues,
    );
    final headerLabel = !hasStrategy
        ? 'STRATEGY'
        : (isProbability ? 'POLICY  π' : 'Q-VALUES');

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF272A2D),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: hasStrategy
              ? const Color(0xFF324E7A)
              : const Color(0xFF4A4E52),
          width: 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            headerLabel,
            style: const TextStyle(
              color: Color(0xCCEAE6D9),
              fontSize: 13,
              letterSpacing: 1.6,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 12),
          if (!hasStrategy)
            const Text(
              'No model is acting on this turn.\n\n'
              'Load a checkpoint (top right) and assign it to a seat to '
              'see its per-action evaluation. DMC checkpoints show '
              'Q-values; CFR PolicyNet checkpoints show action probabilities.',
              style: TextStyle(
                color: Color(0x99EAE6D9),
                fontStyle: FontStyle.italic,
                fontSize: 13,
                height: 1.4,
              ),
            )
          else
            _StrategyBars(session: session, isProbability: isProbability),
        ],
      ),
    );
  }
}

/// Heuristic: are these numbers a probability distribution over legal actions?
/// True iff every value lies in [-eps, 1+eps] AND their sum is in [1-eps, 1+eps].
/// The relaxed lower bound on values guards against tiny negative drift from
/// the masked-softmax in the CFR PolicyNet shim.
bool _isProbabilityDistribution(Float32List vs) {
  if (vs.isEmpty) return false;
  const eps = 1e-3;
  double sum = 0.0;
  for (var i = 0; i < vs.length; i++) {
    final v = vs[i];
    if (v < -eps || v > 1.0 + eps) return false;
    sum += v;
  }
  return (sum - 1.0).abs() <= 1e-2;
}

/// Bars for the seat-to-act when a model is attached. Sorted by descending
/// value. `isProbability` selects label format (% vs raw) and bar scaling
/// (absolute 0..1 vs min-max relative).
class _StrategyBars extends StatelessWidget {
  final GameSession session;
  final bool isProbability;
  const _StrategyBars({required this.session, required this.isProbability});

  @override
  Widget build(BuildContext context) {
    final obs = session.observation;
    final s   = session.strategy;
    if (obs == null || s == null) return const SizedBox.shrink();

    final pairs = <({int idx, ActionType type, double v})>[
      for (int i = 0; i < obs.legal.length; i++)
        (idx: i, type: obs.legal[i], v: s.qValues[i].toDouble()),
    ];
    pairs.sort((a, b) => b.v.compareTo(a.v));

    // Bar fill scale.
    //   probability mode → absolute 0..1 (a 50% action gets a half-full bar)
    //   q-value mode    → min-max relative (the worst action's bar is empty,
    //                     the best action's bar is full)
    final values = pairs.map((e) => e.v).toList();
    final double minV = values.reduce((a, b) => a < b ? a : b);
    final double maxV = values.reduce((a, b) => a > b ? a : b);
    final double span = (maxV - minV).abs() < 1e-6 ? 1.0 : (maxV - minV);

    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        for (final p in pairs)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 3),
            child: _StrategyBar(
              type: p.type,
              value: p.v,
              relative: isProbability
                  ? p.v.clamp(0.0, 1.0)
                  : (p.v - minV) / span,
              isArgmax: p.idx == s.argmaxLegalIdx,
              isProbability: isProbability,
            ),
          ),
      ],
    );
  }
}

class _StrategyBar extends StatelessWidget {
  final ActionType type;
  final double value;
  final double relative;          // 0..1 fill
  final bool   isArgmax;
  final bool   isProbability;

  const _StrategyBar({
    required this.type,
    required this.value,
    required this.relative,
    required this.isArgmax,
    required this.isProbability,
  });

  @override
  Widget build(BuildContext context) {
    // In probability mode the value is always non-negative, so the "negative
    // → red" cue is never used. In Q mode, negative-EV actions get red so
    // you can read the sign at a glance.
    final fill = isArgmax
        ? const Color(0xFFC9911E)
        : (isProbability
            ? const Color(0xFF2D5B7C)
            : (value >= 0
                ? const Color(0xFF2D7A4D)
                : const Color(0xFF7A2D2D)));
    final label = isProbability
        ? '${(value * 100).toStringAsFixed(value >= 0.10 ? 0 : 1)}%'
        : value.toStringAsFixed(2);
    return Stack(
      children: [
        Container(
          height: 26,
          decoration: BoxDecoration(
            color: const Color(0xFF34383B),
            borderRadius: BorderRadius.circular(5),
          ),
        ),
        FractionallySizedBox(
          widthFactor: relative.clamp(0.0, 1.0),
          child: Container(
            height: 26,
            decoration: BoxDecoration(
              color: fill,
              borderRadius: BorderRadius.circular(5),
            ),
          ),
        ),
        Positioned.fill(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: Row(
              children: [
                SizedBox(
                  width: 18,
                  child: isArgmax
                      ? const Icon(Icons.star,
                          size: 14, color: Color(0xFFFFD24A))
                      : null,
                ),
                Text(
                  type.shortLabel,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const Spacer(),
                Text(
                  label,
                  style: const TextStyle(
                    color: Colors.white,
                    fontFamily: 'monospace',
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

/// Simple pill-shaped tab toggle used by the top-of-screen Inspector/Training
/// switcher. Rolls our own rather than using a TabBar because the two screens
/// have very different state lifecycles and don't share a TabController.
class _TabPill extends StatelessWidget {
  final String label;
  final IconData icon;
  final bool selected;
  final VoidCallback onTap;

  const _TabPill({
    required this.label,
    required this.icon,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: selected
              ? const Color(0xFF1F3A56)
              : const Color(0xFF2F3236),
          border: Border.all(
            color: selected
                ? const Color(0xFF6FB3DC)
                : const Color(0xFF4A4E52),
            width: 1.4,
          ),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 16,
              color: selected
                  ? const Color(0xFF6FB3DC)
                  : const Color(0xAAEAE6D9),
            ),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                color: selected
                    ? const Color(0xFFEAE6D9)
                    : const Color(0xAAEAE6D9),
                fontWeight: FontWeight.w800,
                fontSize: 14,
                letterSpacing: 0.4,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

