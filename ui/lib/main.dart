/// PokerTrainer interactive policy inspector — entry point.
///
/// Loads libpokertrainer.so + initializes hand-eval tables, then opens the
/// inspector window. One GameSession lives at the root and feeds every
/// screen widget via ChangeNotifier ticks.
library;

import 'package:flutter/material.dart';

import 'ffi/engine.dart';
import 'game/game_session.dart';
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
  late final GameSession _session;
  bool _revealAllHoles = true;

  @override
  void initState() {
    super.initState();
    _session = GameSession(widget.engine);
  }

  @override
  void dispose() {
    _session.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _session,
      builder: (context, _) => Scaffold(
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                _topBar(),
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
              'No model loaded.\n\n'
              'When a model is attached to a seat, this panel will show '
              'its Q-value per legal action, the argmax pick, and the ΔQ '
              'vs the action you choose.',
              style: TextStyle(
                color: Color(0x99EAE6D9),
                fontStyle: FontStyle.italic,
                fontSize: 13,
                height: 1.4,
              ),
            )
          else
            const Text(
              'Q-values render here.',
              style: TextStyle(
                color: Color(0xFFEAE6D9),
                fontSize: 14,
              ),
            ),
        ],
      ),
    );
  }
}
