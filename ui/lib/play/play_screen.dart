/// "Play" tab: a configurable two-seat match — human / loaded bot / Slumbot.
///
/// Reuses the inspector's table / action-bar / history widgets, driven by a
/// [MatchController] instead of a bare `GameSession`, so the same surface
/// renders both local matches and the Slumbot mirror. Kept separate from the
/// Inspector and Training tabs (those are unchanged).
library;

import 'package:flutter/material.dart';

import '../ffi/actions.dart';
import '../ffi/engine.dart';
import '../game/game_session.dart';
import '../game/model_registry.dart';
import '../widgets/action_bar.dart';
import '../widgets/history_strip.dart';
import '../widgets/table_view.dart';
import 'match_controller.dart';

class PlayScreen extends StatefulWidget {
  final PokerEngine engine;
  final ModelRegistry models;
  const PlayScreen({super.key, required this.engine, required this.models});

  @override
  State<PlayScreen> createState() => _PlayScreenState();
}

class _PlayScreenState extends State<PlayScreen> {
  late final MatchController _match;
  // Opponent cards hidden by default; the eye toggle reveals known cards
  // (your own bot's any time; Slumbot's only at showdown). Your own cards
  // are always visible regardless.
  bool _reveal = false;

  @override
  void initState() {
    super.initState();
    _match = MatchController(widget.engine);
  }

  @override
  void dispose() {
    _match.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: Listenable.merge([_match, widget.models]),
      builder: (context, _) {
        final session = _match.session;
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _controlRow(),
            const SizedBox(height: 10),
            if (_match.isSlumbotMatch) ...[
              _slumbotInfoNote(),
              const SizedBox(height: 8),
            ],
            if (_match.error != null) _errorBanner(_match.error!),
            const SizedBox(height: 6),
            Expanded(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  SizedBox(width: 260, child: HistoryStrip(history: session.history)),
                  const SizedBox(width: 16),
                  Expanded(
                    child: SingleChildScrollView(
                      child: Column(
                        children: [
                          ConstrainedBox(
                            constraints: const BoxConstraints(maxWidth: 880),
                            child: TableView(
                              session: session,
                              labelFor: (p) => _match.seatLabel(p),
                              // Your own cards always visible. An opponent's are
                              // shown when their cards are known AND either the
                              // eye is on or the hand is over — so a showdown
                              // auto-reveals the bot's hand (Slumbot only
                              // discloses it then) without needing the toggle.
                              faceDownOf: (p) => _match.isHumanSeat(p)
                                  ? false
                                  : !(_match.cardsKnown(p) &&
                                      (_reveal || session.isTerminal)),
                              bubbleFor: (p) => _match.bubbleFor(p),
                              // When the eye is on but an opponent's cards
                              // aren't in the engine yet, explain why the peek
                              // is empty. Only Slumbot withholds mid-hand —
                              // your own loaded bots reveal immediately.
                              revealHintFor: (p) => _revealHintFor(p, session),
                            ),
                          ),
                          const SizedBox(height: 16),
                          ConstrainedBox(
                            constraints: const BoxConstraints(maxWidth: 880),
                            child: _actionArea(),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  SizedBox(width: 240, child: _infoPanel()),
                ],
              ),
            ),
          ],
        );
      },
    );
  }

  /// Caption for a face-down opponent seat when the user toggled reveal but the
  /// cards aren't peekable yet. Returns null unless the eye is on, the seat is a
  /// non-human opponent, the hand is live, and the cards are genuinely unknown
  /// (Slumbot pre-showdown) — your own loaded bots are dealt locally, so
  /// `cardsKnown` is already true for them and they simply reveal.
  String? _revealHintFor(Player p, GameSession session) {
    if (!_reveal || _match.isHumanSeat(p)) return null;
    if (_match.cardsKnown(p) || session.isTerminal) return null;
    return _match.isSlumbotMatch && p == _match.slumbotSeat
        ? 'shown at showdown'
        : null;
  }

  // ─── controls ─────────────────────────────────────────────────────────────

  Widget _controlRow() {
    return Wrap(
      crossAxisAlignment: WrapCrossAlignment.center,
      spacing: 12,
      runSpacing: 8,
      children: [
        // Labels match the table's seat names (SB bottom / BB top). In a local
        // match these positions are fixed (SB acts first); in a Slumbot match
        // they rotate each hand — see _slumbotInfoNote below.
        _seatDropdown('SB', _match.seatA, (c) => _match.setSeat(isA: true, choice: c)),
        const Icon(Icons.sports_mma, size: 18),
        _seatDropdown('BB', _match.seatB, (c) => _match.setSeat(isA: false, choice: c)),
        const SizedBox(width: 8),
        FilledButton.icon(
          onPressed: _match.pending ? null : () => _match.newHand(),
          icon: const Icon(Icons.casino, size: 18),
          label: const Text('New hand'),
        ),
        IconButton(
          tooltip: _reveal
              ? "Hide opponent's cards"
              : "Reveal opponent's cards (known only — your bot anytime, "
                  "Slumbot at showdown)",
          icon: Icon(_reveal ? Icons.visibility : Icons.visibility_off, size: 22),
          onPressed: () => setState(() => _reveal = !_reveal),
        ),
        if (_match.pending)
          const Row(mainAxisSize: MainAxisSize.min, children: [
            SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2)),
            SizedBox(width: 8),
            Text('Slumbot thinking…', style: TextStyle(fontStyle: FontStyle.italic)),
          ]),
      ],
    );
  }

  Widget _seatDropdown(String label, SeatChoice current, ValueChanged<SeatChoice> onPick) {
    // id encoding: 'human' | 'slumbot' | 'model:<path>'
    String idOf(SeatChoice c) => switch (c.role) {
          SeatRole.human => 'human',
          SeatRole.slumbot => 'slumbot',
          SeatRole.model => 'model:${c.model!.path}',
        };
    final items = <DropdownMenuItem<String>>[
      const DropdownMenuItem(value: 'human', child: Text('You (human)')),
      const DropdownMenuItem(value: 'slumbot', child: Text('Slumbot')),
      for (final m in widget.models.models)
        DropdownMenuItem(value: 'model:${m.path}', child: Text(m.label)),
    ];
    return Row(mainAxisSize: MainAxisSize.min, children: [
      Text('$label: ', style: const TextStyle(color: Color(0xCCEAE6D9))),
      DropdownButton<String>(
        value: idOf(current),
        items: items,
        onChanged: (v) {
          if (v == null) return;
          if (v == 'human') {
            onPick(const SeatChoice.human());
          } else if (v == 'slumbot') {
            onPick(const SeatChoice.slumbot());
          } else if (v.startsWith('model:')) {
            final path = v.substring('model:'.length);
            final m = widget.models.models.firstWhere((e) => e.path == path);
            onPick(SeatChoice.model(m));
          }
        },
      ),
    ]);
  }

  Widget _actionArea() {
    final session = _match.session;
    if (session.isTerminal) {
      final summary = session.terminalSummary();
      final txt = summary == null
          ? 'Hand over.'
          : (summary.winner == null
              ? 'Chop.'
              : '${summary.winner == Player.sb ? "SB" : "BB"} wins '
                  '${summary.winAmountBb.toStringAsFixed(2)} bb'
                  '${summary.sbHandCategory != null ? " — ${summary.winner == Player.sb ? summary.sbHandCategory : summary.bbHandCategory}" : ""}');
      return _banner('$txt   ·   tap “New hand” to continue.');
    }
    if (_match.awaitingHumanInput) {
      // Interactive: route taps through the controller (which posts to Slumbot
      // in a Slumbot match, or applies locally otherwise).
      return ActionBar(session: session, onAction: (i) => _match.humanAct(i));
    }
    // Non-interactive: a bot or Slumbot is to act (or we're mid-network-call).
    if (_match.pending) return _banner('Slumbot acting…  (watch the table)');
    if (_match.isSlumbotMatch && !_match.handStarted) {
      return _banner('Tap “New hand” to start a hand vs Slumbot.');
    }
    return _banner('Opponent to act…');
  }

  // ─── side panel ───────────────────────────────────────────────────────────

  Widget _infoPanel() {
    final mbb = _match.mbbPerHand;
    final strat = _match.session.strategy;
    final equity = _match.session.equity;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Match', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 6),
        if (_match.isSlumbotMatch) ...[
          Text('Hands: ${_match.handsPlayed}'),
          if (mbb != null)
            Text('Win rate: ${mbb.toStringAsFixed(0)} mbb/hand',
                style: TextStyle(
                    color: mbb >= 0 ? const Color(0xFF6FBF73) : const Color(0xFFD98C8C),
                    fontWeight: FontWeight.w600)),
          if (_match.lastWinningsChips != null)
            Text('Last: ${(_match.lastWinningsChips! / TableState.chipsPerBb).toStringAsFixed(2)} bb',
                style: const TextStyle(color: Color(0xCCEAE6D9))),
          const Divider(height: 18),
        ],
        if (equity != null) ...[
          const Text('Equity', style: TextStyle(fontWeight: FontWeight.w600)),
          Text('SB ${(equity.sbWin * 100).toStringAsFixed(1)}%  ·  '
              'BB ${(equity.bbWin * 100).toStringAsFixed(1)}%',
              style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
          const SizedBox(height: 10),
        ],
        if (strat != null) ...[
          const Text('Bot strategy (to act)', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          _strategyList(strat),
        ],
      ],
    );
  }

  Widget _strategyList(StrategyView strat) {
    final obs = _match.session.observation;
    if (obs == null) return const SizedBox.shrink();
    final rows = <Widget>[];
    for (int i = 0; i < obs.legal.length && i < strat.qValues.length; i++) {
      final pick = i == strat.argmaxLegalIdx;
      rows.add(Text(
        '${pick ? "▶ " : "  "}${obs.legal[i].shortLabel}: ${strat.qValues[i].toStringAsFixed(3)}',
        style: TextStyle(
          fontFamily: 'monospace',
          fontSize: 12,
          fontWeight: pick ? FontWeight.w700 : FontWeight.w400,
          color: pick ? const Color(0xFF9BD0FF) : const Color(0xCCEAE6D9),
        ),
      ));
    }
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: rows);
  }

  Widget _banner(String text) => Container(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 16),
        decoration: BoxDecoration(
          color: const Color(0x22FFFFFF),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(text, textAlign: TextAlign.center),
      );

  /// Slumbot is the authority: it deals and the button alternates each hand, so
  /// the SB/BB seat labels rotate and the player can't choose a position. Shown
  /// only in a Slumbot match (a local match has fixed SB=bottom / BB=top seats).
  Widget _slumbotInfoNote() => Container(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
        decoration: BoxDecoration(
          color: const Color(0x166FA8DC),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: const Color(0x336FA8DC)),
        ),
        child: const Row(children: [
          Icon(Icons.info_outline, size: 16, color: Color(0xFF9BD0FF)),
          SizedBox(width: 8),
          Expanded(
            child: Text(
              'Slumbot deals and is the dealer: the button alternates every hand, '
              'so who is SB/BB (and acts first) rotates automatically — you '
              'don’t choose your seat against Slumbot.',
              style: TextStyle(fontSize: 12, color: Color(0xDDEAE6D9)),
            ),
          ),
        ]),
      );

  Widget _errorBanner(String text) => Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: const Color(0x33D98C8C),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(children: [
          const Icon(Icons.error_outline, size: 18, color: Color(0xFFD98C8C)),
          const SizedBox(width: 8),
          Expanded(child: Text(text, style: const TextStyle(fontSize: 12))),
        ]),
      );
}
