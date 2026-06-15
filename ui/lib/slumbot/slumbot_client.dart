/// Thin HTTP client for the public Slumbot API (https://slumbot.com/api).
///
/// No game logic here — it just performs the `new_hand` / `act` round-trips
/// and returns the parsed JSON as a [SlumbotResponse]. The Play screen's
/// `MatchController` interprets the fields (action string, board, cards,
/// position, winnings) and drives the mirror env. Login is optional (only used
/// for Slumbot-side stats); anonymous play works with an empty token.
///
/// Protocol reference: ericgjackson/slumbot2019 `slumbot_api.py`.
library;

import 'dart:convert';

import 'package:http/http.dart' as http;

const String _baseUrl = 'https://slumbot.com/api';

/// Parsed `new_hand` / `act` response. Fields absent in a given response are
/// null (e.g. `winnings`/`botHoleCards` only appear when the hand is over).
class SlumbotResponse {
  /// Session token to pass to the next `act` (and the next `new_hand`).
  final String token;

  /// The full action string for the hand so far (e.g. "b200c/kk").
  final String action;

  /// Client's hole cards as 2-char strings (["Ah","Kd"]).
  final List<String> holeCards;

  /// Board cards revealed so far (0/3/4/5 entries).
  final List<String> board;

  /// Client's position this hand. Per the API, `clientPos == 0` means the
  /// client is the **big blind** (acts last preflop); `1` means the
  /// small-blind/button (acts first preflop). Verified against the live API
  /// during integration.
  final int clientPos;

  /// Client's net chips for the hand, present iff the hand is over.
  final int? winnings;

  /// Bot's hole cards, present iff the hand reached showdown.
  final List<String>? botHoleCards;

  /// Raw decoded JSON (for fields not surfaced above / debugging).
  final Map<String, dynamic> raw;

  const SlumbotResponse({
    required this.token,
    required this.action,
    required this.holeCards,
    required this.board,
    required this.clientPos,
    required this.winnings,
    required this.botHoleCards,
    required this.raw,
  });

  bool get handOver => winnings != null;

  factory SlumbotResponse.fromJson(Map<String, dynamic> j) {
    List<String> strs(dynamic v) =>
        v == null ? const [] : (v as List).map((e) => e.toString()).toList();
    return SlumbotResponse(
      token: (j['token'] ?? '').toString(),
      action: (j['action'] ?? '').toString(),
      holeCards: strs(j['hole_cards']),
      board: strs(j['board']),
      clientPos: (j['client_pos'] ?? 0) as int,
      winnings: j['winnings'] as int?,
      botHoleCards: j['bot_hole_cards'] == null ? null : strs(j['bot_hole_cards']),
      raw: j,
    );
  }
}

/// Raised on any non-200 response or transport failure, with a readable
/// message the UI can surface.
class SlumbotException implements Exception {
  final String message;
  const SlumbotException(this.message);
  @override
  String toString() => 'SlumbotException: $message';
}

class SlumbotClient {
  final http.Client _http;
  final String baseUrl;

  /// Persistent login token (for Slumbot-side stats); empty for anonymous play.
  /// A future login flow would thread this in via the constructor.
  final String _loginToken;

  /// The single session token Slumbot hands back, rotated by every `new_hand`
  /// AND every `act`. It is carried FORWARD into the next `new_hand` — that is
  /// what makes Slumbot ALTERNATE the button each hand. Starting a hand WITHOUT
  /// a token always seats Slumbot as the button, so Slumbot acts first every
  /// hand (verified live: tokenless new_hand ⇒ client_pos always 0; carrying
  /// the token ⇒ client_pos alternates). Empty until the first reply.
  String _sessionToken = '';

  String get handToken => _sessionToken;

  SlumbotClient({
    http.Client? httpClient,
    this.baseUrl = _baseUrl,
    String loginToken = '',
  })  : _http = httpClient ?? http.Client(),
        _loginToken = loginToken;

  void dispose() => _http.close();

  /// Start a fresh hand, carrying the session token forward so Slumbot
  /// alternates the button. If Slumbot rejects the carried token, fall back to
  /// a tokenless `new_hand` so play never hard-fails — that fallback is the old
  /// always-anonymous behaviour (Slumbot simply takes the button again).
  Future<SlumbotResponse> newHand() async {
    final carried = _sessionToken.isNotEmpty ? _sessionToken : _loginToken;
    SlumbotResponse r;
    try {
      r = await _post('new_hand', {if (carried.isNotEmpty) 'token': carried});
    } on SlumbotException {
      if (carried.isEmpty) rethrow;
      _sessionToken = '';
      r = await _post(
          'new_hand', {if (_loginToken.isNotEmpty) 'token': _loginToken});
    }
    if (r.token.isNotEmpty) _sessionToken = r.token;
    return r;
  }

  /// Send the client's incremental action (e.g. "b300", "c", "k", "f").
  Future<SlumbotResponse> act(String incr) async {
    final r = await _post('act', {'token': _sessionToken, 'incr': incr});
    if (r.token.isNotEmpty) _sessionToken = r.token;
    return r;
  }

  Future<SlumbotResponse> _post(String path, Map<String, dynamic> body) async {
    late http.Response resp;
    try {
      resp = await _http.post(
        Uri.parse('$baseUrl/$path'),
        headers: const {'Content-Type': 'application/json'},
        body: jsonEncode(body),
      );
    } catch (e) {
      throw SlumbotException('network error talking to Slumbot: $e');
    }
    if (resp.statusCode != 200) {
      throw SlumbotException('$path → HTTP ${resp.statusCode}: ${resp.body}');
    }
    final Map<String, dynamic> j;
    try {
      j = jsonDecode(resp.body) as Map<String, dynamic>;
    } catch (e) {
      throw SlumbotException('$path → bad JSON: ${resp.body}');
    }
    if (j.containsKey('error_msg')) {
      throw SlumbotException('$path → ${j['error_msg']}');
    }
    return SlumbotResponse.fromJson(j);
  }
}
