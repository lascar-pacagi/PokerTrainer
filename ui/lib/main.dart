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
                const SizedBox(height: 16),
                Expanded(
                  child: Center(
                    child: SingleChildScrollView(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          ConstrainedBox(
                            constraints: const BoxConstraints(maxWidth: 880),
                            child: TableView(
                              session: _session,
                              revealAllHoles: _revealAllHoles,
                            ),
                          ),
                          const SizedBox(height: 16),
                          ConstrainedBox(
                            constraints: const BoxConstraints(maxWidth: 880),
                            child: ActionBar(session: _session),
                          ),
                          const SizedBox(height: 16),
                          ConstrainedBox(
                            constraints: const BoxConstraints(maxWidth: 880),
                            child: HistoryStrip(history: _session.history),
                          ),
                        ],
                      ),
                    ),
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
          ),
        ),
        const SizedBox(width: 18),
        Text(
          'Hand #${_session.handCount}  ·  seed $seedTxt',
          style: const TextStyle(
            color: Color(0xAAEAE6D9),
            fontFamily: 'monospace',
            fontSize: 14,
          ),
        ),
        const Spacer(),
        IconButton(
          tooltip: _revealAllHoles ? 'Hide opponent cards' : 'Reveal all holes',
          icon: Icon(_revealAllHoles ? Icons.visibility : Icons.visibility_off),
          onPressed: () => setState(() => _revealAllHoles = !_revealAllHoles),
        ),
        const SizedBox(width: 6),
        FilledButton.icon(
          icon: const Icon(Icons.shuffle, size: 18),
          label: const Text('Deal random'),
          onPressed: _session.dealRandom,
        ),
        const SizedBox(width: 8),
        FilledButton.tonalIcon(
          icon: const Icon(Icons.replay, size: 18),
          label: const Text('Re-seed'),
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
