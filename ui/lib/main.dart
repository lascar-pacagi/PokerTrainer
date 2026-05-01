/// PokerTrainer interactive policy inspector — entry point.
///
/// Loads libpokertrainer.so + initializes hand-eval tables, then opens the
/// inspector window. One GameSession lives at the root and feeds every
/// screen widget via ChangeNotifier ticks.
library;

import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';

import 'ffi/actions.dart';
import 'ffi/engine.dart';
import 'game/game_session.dart';
import 'game/model_registry.dart';
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
        scaffoldBackgroundColor: const Color(0xFF0B0F0D),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2D5B7C),
          brightness: Brightness.dark,
        ),
        textTheme: Typography.whiteMountainView,
      ),
      home: InspectorHome(engine: engine),
    );
  }
}

class InspectorHome extends StatefulWidget {
  final PokerEngine engine;
  const InspectorHome({super.key, required this.engine});

  @override
  State<InspectorHome> createState() => _InspectorHomeState();
}

class _InspectorHomeState extends State<InspectorHome> {
  late final GameSession  _session;
  late final ModelRegistry _models;
  bool _revealAllHoles = true;

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
                _topBar(),
                const SizedBox(height: 10),
                _agentRow(),
                const SizedBox(height: 18),
                Expanded(
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
              ],
            ),
          ),
        ),
      ),
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
        color: const Color(0xFF101010),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFF2A2A2A), width: 1),
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
              dropdownColor: const Color(0xFF1B1B1B),
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

/// Placeholder for the strategy panel — empty card until a model is loaded.
/// Once `session.strategy` is non-null, this will paint Q-bars + argmax +
/// ΔQ-vs-action; for now it just announces itself.
class _StrategyPlaceholder extends StatelessWidget {
  final GameSession session;
  const _StrategyPlaceholder({required this.session});

  @override
  Widget build(BuildContext context) {
    final hasStrategy = session.strategy != null;
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF101010),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: hasStrategy
              ? const Color(0xFF324E7A)
              : const Color(0xFF2A2A2A),
          width: 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text(
            'STRATEGY',
            style: TextStyle(
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
              'see its Q-value per legal action and the argmax pick.',
              style: TextStyle(
                color: Color(0x99EAE6D9),
                fontStyle: FontStyle.italic,
                fontSize: 13,
                height: 1.4,
              ),
            )
          else
            _StrategyBars(session: session),
        ],
      ),
    );
  }
}

/// Q-value bars for the seat-to-act, when a model is attached. Ordered by
/// descending Q so the recommended action floats to the top.
class _StrategyBars extends StatelessWidget {
  final GameSession session;
  const _StrategyBars({required this.session});

  @override
  Widget build(BuildContext context) {
    final obs = session.observation;
    final s   = session.strategy;
    if (obs == null || s == null) return const SizedBox.shrink();

    // Pair (legalIdx, action, q), then sort by q descending.
    final pairs = <({int idx, ActionType type, double q})>[
      for (int i = 0; i < obs.legal.length; i++)
        (idx: i, type: obs.legal[i], q: s.qValues[i].toDouble()),
    ];
    pairs.sort((a, b) => b.q.compareTo(a.q));

    // Bar scale: range across this decision's Q values.
    final qs = pairs.map((e) => e.q).toList();
    final minQ = qs.reduce((a, b) => a < b ? a : b);
    final maxQ = qs.reduce((a, b) => a > b ? a : b);
    final span = (maxQ - minQ).abs() < 1e-6 ? 1.0 : (maxQ - minQ);

    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        for (final p in pairs)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 3),
            child: _QBar(
              type: p.type,
              q: p.q,
              relative: (p.q - minQ) / span,
              isArgmax: p.idx == s.argmaxLegalIdx,
            ),
          ),
      ],
    );
  }
}

class _QBar extends StatelessWidget {
  final ActionType type;
  final double q;
  final double relative;   // 0..1 fill
  final bool   isArgmax;

  const _QBar({
    required this.type,
    required this.q,
    required this.relative,
    required this.isArgmax,
  });

  @override
  Widget build(BuildContext context) {
    final fill = isArgmax
        ? const Color(0xFFFFD24A)
        : (q >= 0 ? const Color(0xFF2D7A4D) : const Color(0xFF7A2D2D));
    return Stack(
      children: [
        Container(
          height: 26,
          decoration: BoxDecoration(
            color: const Color(0xFF1B1B1B),
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
                if (isArgmax) ...[
                  const Icon(Icons.star, size: 14, color: Colors.black87),
                  const SizedBox(width: 4),
                ],
                Text(
                  type.shortLabel,
                  style: TextStyle(
                    color: isArgmax ? Colors.black87 : Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const Spacer(),
                Text(
                  q.toStringAsFixed(2),
                  style: TextStyle(
                    color: isArgmax ? Colors.black87 : Colors.white,
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
