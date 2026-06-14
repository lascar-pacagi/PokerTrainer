/// Drives a two-seat "Play" match. Each seat is Human, a loaded Model, or
/// Slumbot. Two orchestration modes behind one controller:
///
///  * **Local** (no Slumbot seat): the engine is authoritative. Delegates to
///    `GameSession` as-is — it already deals, auto-steps Model seats, and
///    exposes the strategy panel. Stack = 100 bb.
///
///  * **Slumbot** (one seat = Slumbot): Slumbot is authoritative (it deals and
///    bets arbitrary amounts). We keep a `GameSession` *mirror* env at 200 bb
///    (Slumbot's format), inject Slumbot's dealt cards, and apply both sides'
///    actions to it (local discrete actions / Slumbot's exact bets via
///    `stepRaiseTo`). The same widgets render the mirror.
///
/// Slumbot's hole cards are only injected at showdown (the API discloses them
/// only then). Position alternates each hand (Slumbot reports `client_pos`).
library;

import 'package:flutter/foundation.dart';

import '../ffi/actions.dart';
import '../ffi/engine.dart';
import '../game/game_session.dart';
import '../game/model_registry.dart';
import '../slumbot/slumbot_client.dart';
import '../slumbot/slumbot_protocol.dart';

enum SeatRole { human, model, slumbot }

class SeatChoice {
  final SeatRole role;
  final LoadedModel? model;
  const SeatChoice.human() : role = SeatRole.human, model = null;
  const SeatChoice.slumbot() : role = SeatRole.slumbot, model = null;
  const SeatChoice.model(LoadedModel this.model) : role = SeatRole.model;

  bool get isSlumbot => role == SeatRole.slumbot;
  bool get isModel => role == SeatRole.model;

  String get label => switch (role) {
        SeatRole.human => 'You (human)',
        SeatRole.slumbot => 'Slumbot',
        SeatRole.model => model!.label,
      };

  SeatAgent toAgent() => role == SeatRole.model
      ? ModelAgent(model!.handle, label: model!.label)
      : const HumanAgent();
}

Player _other(Player p) => p == Player.sb ? Player.bb : Player.sb;

class MatchController extends ChangeNotifier {
  final PokerEngine engine;

  MatchController(this.engine) {
    _rebuildSession();
  }

  // ─── configuration ──────────────────────────────────────────────────────

  SeatChoice _seatA = const SeatChoice.human();    // local default: SB
  SeatChoice _seatB = const SeatChoice.slumbot();   // local default: BB / opp

  SeatChoice get seatA => _seatA;
  SeatChoice get seatB => _seatB;

  bool get isSlumbotMatch => _seatA.isSlumbot || _seatB.isSlumbot;
  SeatChoice get _localChoice => _seatA.isSlumbot ? _seatB : _seatA;

  /// Reassign a seat. Rebuilds the mirror/session (stack + mode may change)
  /// and resets the running tally.
  void setSeat({required bool isA, required SeatChoice choice}) {
    // Enforce "at most one Slumbot".
    if (choice.isSlumbot) {
      if (isA && _seatB.isSlumbot) _seatB = const SeatChoice.human();
      if (!isA && _seatA.isSlumbot) _seatA = const SeatChoice.human();
    }
    if (isA) {
      _seatA = choice;
    } else {
      _seatB = choice;
    }
    _rebuildSession();
  }

  // ─── session / mirror ───────────────────────────────────────────────────

  GameSession? _session;
  GameSession get session => _session!;

  SlumbotClient? _slumbot;

  void _rebuildSession() {
    _session?.dispose();
    _slumbot?.dispose();
    _slumbot = null;
    _resetTally();
    _error = null;
    _pending = false;

    if (isSlumbotMatch) {
      _session = GameSession(engine, startingStackChips: 200 * TableState.chipsPerBb);
      _session!.setAutoStep(false); // controller drives every seat
      _session!.beginMirrorHand(); // start blank (no placeholder deal shown)
      _slumbot = SlumbotClient();
      _handStarted = false;
    } else {
      _session = GameSession(engine); // 100 bb, engine-authoritative
      _session!.setAgent(Player.sb, _seatA.toAgent());
      _session!.setAgent(Player.bb, _seatB.toAgent());
      _handStarted = true;
    }
    notifyListeners();
  }

  // ─── running tally ──────────────────────────────────────────────────────

  int _handsPlayed = 0;
  int _netChipsLocal = 0; // Slumbot-reported winnings sum (chips), local seat
  int? _lastWinnings;

  int get handsPlayed => _handsPlayed;
  int? get lastWinningsChips => _lastWinnings;

  /// Local participant's win rate vs Slumbot in mbb/hand (null if no hands or
  /// not a Slumbot match — local matches have no external scorekeeper).
  double? get mbbPerHand {
    if (!isSlumbotMatch || _handsPlayed == 0) return null;
    final netBb = _netChipsLocal / TableState.chipsPerBb;
    return 1000.0 * netBb / _handsPlayed;
  }

  void _resetTally() {
    _handsPlayed = 0;
    _netChipsLocal = 0;
    _lastWinnings = null;
  }

  // ─── status ─────────────────────────────────────────────────────────────

  bool _pending = false; // awaiting a Slumbot HTTP round-trip
  String? _error;
  bool _handStarted = false;

  bool get pending => _pending;
  String? get error => _error;
  bool get handStarted => _handStarted;

  /// Pause between mirrored opponent actions so the user can follow the play.
  Duration replayPause = const Duration(milliseconds: 750);

  /// The engine seat the local participant occupies this hand (Slumbot match
  /// only; null for local matches where seats are fixed SB=A / BB=B).
  Player? get localSeat => isSlumbotMatch ? _localSeat : null;

  /// The engine seat Slumbot occupies this hand (Slumbot match only).
  Player? get slumbotSeat => isSlumbotMatch ? _slumbotSeat : null;

  /// Whether seat `p` is occupied by the human (their own cards stay visible).
  bool isHumanSeat(Player p) {
    if (isSlumbotMatch) {
      return p == _localSeat && _localChoice.role == SeatRole.human;
    }
    final c = p == Player.sb ? _seatA : _seatB;
    return c.role == SeatRole.human;
  }

  /// Whether seat `p`'s hole cards are currently known to us (Slumbot's only at
  /// showdown). The eye toggle reveals known opponent cards.
  bool cardsKnown(Player p) {
    final h = session.env.getHole(p);
    return h[0] != EngineCard.noCard && h[1] != EngineCard.noCard;
  }

  /// Human-readable occupant label for a seat ("You" / "Slumbot" / model name).
  String seatLabel(Player p) {
    if (isSlumbotMatch) {
      if (p == _slumbotSeat) return 'Slumbot';
      return _localChoice.role == SeatRole.model ? _localChoice.label : 'You';
    }
    final c = p == Player.sb ? _seatA : _seatB;
    return switch (c.role) {
      SeatRole.human => 'You',
      SeatRole.model => c.label,
      SeatRole.slumbot => 'Slumbot',
    };
  }

  // ─── action bubbles ("Raise to 6 bb" over the acting seat) ───────────────
  //
  // A bubble belongs to the betting round it was made on. When play advances to
  // a new street (the flop/turn/river is dealt), the previous round's labels are
  // stale and must clear — like the action chips clearing in PokerStars. We tag
  // the live bubbles with `_bubbleStreet` and drop them the moment a later
  // street is reached (a new-street action OR the board running out to it).

  final Map<Player, String?> _bubbles = {Player.sb: null, Player.bb: null};
  Street _bubbleStreet = Street.preflop;

  String? bubbleFor(Player p) => _bubbles[p];

  void _clearBubbles() {
    _bubbles[Player.sb] = null;
    _bubbles[Player.bb] = null;
    _bubbleStreet = Street.preflop;
  }

  /// Drop the current bubbles if `street` is past the one they belong to (so the
  /// preflop labels vanish when the flop arrives, etc.). Cheap and idempotent.
  void _bubbleStreetGate(Street street) {
    if (street.index > _bubbleStreet.index) {
      _clearBubbles();
      _bubbleStreet = street;
    }
  }

  String _bbAmt(int chips) {
    final bb = chips / TableState.chipsPerBb;
    return bb == bb.truncateToDouble() ? bb.toStringAsFixed(0) : bb.toStringAsFixed(1);
  }

  /// Build the bubble caption for the seat-to-act about to play `type`.
  String _bubbleLabel(ActionType type, {required int toCallChips, int betToChips = 0}) {
    final p = session.tableState.toAct;
    switch (type) {
      case ActionType.fold:
        return 'Fold';
      case ActionType.checkCall:
        return toCallChips == 0 ? 'Check' : 'Call';
      default:
        if (p != null) {
          final allInTo = session.tableState.stacksChips[p.index] +
              session.tableState.investedThisStreet[p.index];
          if (betToChips >= allInTo) return 'All-in';
        }
        return toCallChips == 0 ? 'Bet ${_bbAmt(betToChips)} bb'
                                : 'Raise to ${_bbAmt(betToChips)} bb';
    }
  }

  String _bubbleTokenLabel(SbAction a) => switch (a) {
        SbFold() => 'Fold',
        SbCheck() => 'Check',
        SbCall() => 'Call',
        SbBet(:final toChips) => _bubbleLabel(ActionType.raise100,
            toCallChips: session.tableState.toCallChips, betToChips: toChips),
      };

  /// Whether the interactive action bar should be live right now (a human
  /// seat is to act and we are not mid-network-call).
  bool get awaitingHumanInput {
    final s = _session;
    if (s == null || s.isTerminal) return false;
    final p = s.tableState.toAct;
    if (p == null) return false;
    if (!isSlumbotMatch) return s.agentFor(p) is HumanAgent;
    return _handStarted && !_pending && p == _localSeat &&
        _localChoice.role == SeatRole.human;
  }

  // ─── Slumbot per-hand state ─────────────────────────────────────────────

  Player _localSeat = Player.sb;
  Player _slumbotSeat = Player.bb;
  int _appliedTokens = 0;

  // ─── public actions ─────────────────────────────────────────────────────

  /// Start a fresh hand.
  Future<void> newHand() async {
    _error = null;
    if (!isSlumbotMatch) {
      session.dealRandom(); // engine deals + auto-steps any Model seats
      notifyListeners();
      return;
    }
    await _slumbotNewHand();
  }

  /// The local human tapped legal action `legalIdx`. In a local match this is
  /// the ordinary apply; in a Slumbot match it is translated, sent, and the
  /// reply applied to the mirror. Also reused to drive a local Model seat.
  Future<void> humanAct(int legalIdx) async {
    if (!isSlumbotMatch) {
      session.applyLegal(legalIdx); // auto-steps the opponent Model if any
      notifyListeners();
      return;
    }
    await _slumbotLocalAct(legalIdx);
  }

  // ─── Slumbot orchestration ──────────────────────────────────────────────

  Future<void> _slumbotNewHand() async {
    _pending = true;
    notifyListeners();
    try {
      final resp = await _slumbot!.newHand();
      _handStarted = true;
      // Position: client_pos==1 ⇒ client is SB/button (seat 0); ==0 ⇒ BB.
      _localSeat = resp.clientPos == 1 ? Player.sb : Player.bb;
      _slumbotSeat = _other(_localSeat);

      session.beginMirrorHand();
      _clearBubbles();
      // Assign the mirror's agents so the strategy panel lights up for a local
      // Model seat; auto-step is off so this never self-drives.
      session.setAgent(_localSeat, _localChoice.toAgent());
      session.setAgent(_slumbotSeat, const HumanAgent());

      final hero = parseCards(resp.holeCards);
      session.injectHole(_localSeat, hero[0], hero[1]);

      _appliedTokens = 0;
      await _applyServerState(resp); // replays Slumbot's initial action(s), board
    } on SlumbotException catch (e) {
      _error = e.message;
    } catch (e) {
      _error = 'unexpected: $e';
    } finally {
      _pending = false;
      notifyListeners();
    }
    await _driveLocalModelIfNeeded();
  }

  Future<void> _slumbotLocalAct(int legalIdx) async {
    final obs = session.observation;
    if (obs == null) return;
    final type = obs.legal[legalIdx];
    final sizing = obs.sizingForLegal(legalIdx);
    final toCall = session.tableState.toCallChips;
    final myStreet = session.tableState.street; // the round we are acting in
    final myLabel = _bubbleLabel(type, toCallChips: toCall, betToChips: sizing);

    // Apply our action to the mirror first (instant UI feedback), then post.
    if (type == ActionType.fold || type == ActionType.checkCall) {
      session.applyActionDirect(type);
    } else {
      session.applyRaiseToDirect(sizing); // RAISE_* / ALL_IN → exact chips
    }
    _bubbleStreetGate(myStreet);          // dropped stale labels if a new round
    _bubbles[_localSeat] = myLabel;
    _appliedTokens += 1;
    final incr = encodeAction(type, toCallChips: toCall, betToChips: sizing);

    _pending = true;
    notifyListeners();
    try {
      final resp = await _slumbot!.act(incr);
      await _applyServerState(resp); // applies Slumbot's reply, board, terminal
    } on SlumbotException catch (e) {
      _error = e.message;
    } catch (e) {
      _error = 'unexpected: $e';
    } finally {
      _pending = false;
      notifyListeners();
    }
    await _driveLocalModelIfNeeded();
  }

  /// Apply every server action token beyond what we've already mirrored, then
  /// reveal the board and (at showdown) Slumbot's cards, and bank the result.
  /// Paced: each newly-revealed opponent action is shown, then we pause, so the
  /// user can follow Slumbot's play instead of seeing it jump.
  Future<void> _applyServerState(SlumbotResponse resp) async {
    final tokens = parseActionString(resp.action);
    for (int i = _appliedTokens; i < tokens.length; i++) {
      final a = tokens[i].action;
      final actor = session.tableState.toAct; // who is about to act
      final street = session.tableState.street; // the round this action is in
      final label = _bubbleTokenLabel(a);     // computed from pre-apply state
      _bubbleStreetGate(street);               // a new round? drop the old labels
      switch (a) {
        case SbFold():
          session.applyActionDirect(ActionType.fold);
        case SbCheck():
        case SbCall():
          session.applyActionDirect(ActionType.checkCall);
        case SbBet(:final toChips):
          session.applyRaiseToDirect(toChips);
      }
      if (actor != null) _bubbles[actor] = label;
      _appliedTokens = i + 1;
      notifyListeners();                       // show this action + its bubble
      await Future<void>.delayed(replayPause);  // linger so the user reads it
    }
    _appliedTokens = tokens.length;

    session.injectBoard(boardSlots(resp.board));
    // The board may have run out to a new street with no action yet (the round
    // closed on the last token) — clear the just-shown round's labels for it.
    _bubbleStreetGate(session.tableState.street);

    if (resp.handOver) {
      if (resp.botHoleCards != null && resp.botHoleCards!.length == 2) {
        final v = parseCards(resp.botHoleCards!);
        session.injectHole(_slumbotSeat, v[0], v[1]);
      }
      _handsPlayed += 1;
      _lastWinnings = resp.winnings;
      _netChipsLocal += resp.winnings ?? 0;
      _handStarted = false;
    }
  }

  /// In a Slumbot match where the local seat is a Model, advance it: read the
  /// model's argmax (via the session's strategy cache) and play it, looping
  /// until the hand ends or it is no longer the model's turn. Bounded.
  Future<void> _driveLocalModelIfNeeded() async {
    if (!isSlumbotMatch || !_localChoice.isModel) return;
    int safety = 64;
    while (safety-- > 0 &&
        _handStarted &&
        !session.isTerminal &&
        session.tableState.toAct == _localSeat &&
        _error == null) {
      final strat = session.strategy;
      if (strat == null) break;
      await Future<void>.delayed(replayPause); // let the user see the bot's spot
      await _slumbotLocalAct(strat.argmaxLegalIdx);
    }
  }

  @override
  void dispose() {
    _session?.dispose();
    _slumbot?.dispose();
    super.dispose();
  }
}
