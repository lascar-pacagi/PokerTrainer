/// A single playing-card chip. Painted by hand from rank/suit so we don't
/// need PNG assets. Suited colors: hearts/diamonds = red, clubs/spades = grey.
library;

import 'package:flutter/material.dart';

import '../ffi/actions.dart';

class CardView extends StatelessWidget {
  /// Engine card byte (rank*4+suit). 0xFF = unrevealed slot.
  final int card;

  /// True for the card-back style (face-down). Forces hidden display.
  final bool faceDown;

  final double width;

  const CardView({
    super.key,
    required this.card,
    this.faceDown = false,
    this.width = 44,
  });

  bool get _empty => card == EngineCard.noCard;

  @override
  Widget build(BuildContext context) {
    // CardView uses fixed-pixel positions (top: 4, left: 5, top: width*0.42)
    // alongside relative font sizes. A global textScaler > 1.0 grows the
    // glyphs but not the slots, causing the small suit to overlap the rank.
    // Force textScaler 1.0 inside the painting so rank/suit/big-suit always
    // sit where the layout math expects.
    return MediaQuery(
      data: MediaQuery.of(context).copyWith(textScaler: TextScaler.noScaling),
      child: _build(context),
    );
  }

  Widget _build(BuildContext context) {
    final h = width * 1.45;

    if (_empty) {
      return _emptySlot(h);
    }

    if (faceDown) {
      return Container(
        width: width,
        height: h,
        decoration: BoxDecoration(
          color: const Color(0xFF1A2C4A),
          borderRadius: BorderRadius.circular(width * 0.12),
          border: Border.all(color: const Color(0xFF324E7A), width: 1.5),
        ),
        child: Center(
          child: Container(
            width: width * 0.55,
            height: h * 0.7,
            decoration: BoxDecoration(
              color: const Color(0xFF233C66),
              borderRadius: BorderRadius.circular(width * 0.06),
            ),
          ),
        ),
      );
    }

    final r = EngineCard.rankOf(card);
    final s = EngineCard.suitOf(card);
    final rankCh = EngineCard.rankChars[r];
    final suitCh = _suitGlyph(s);
    final isRed  = (s == 1 || s == 2);   // d / h
    final fg     = isRed ? const Color(0xFFC72D2D) : const Color(0xFF1A1A1A);

    return Container(
      width: width,
      height: h,
      decoration: BoxDecoration(
        color: const Color(0xFFF8F4E9),
        borderRadius: BorderRadius.circular(width * 0.12),
        border: Border.all(color: const Color(0xFF999999), width: 1.0),
        boxShadow: const [
          BoxShadow(color: Color(0x33000000), blurRadius: 3, offset: Offset(1, 1)),
        ],
      ),
      child: Stack(
        children: [
          Positioned(
            top: 4, left: 5,
            child: Text(
              rankCh,
              style: TextStyle(
                color: fg,
                fontSize: width * 0.36,
                fontWeight: FontWeight.w700,
                fontFamily: 'monospace',
                height: 1.0,
              ),
            ),
          ),
          Positioned(
            top: width * 0.50, left: 5,
            child: Text(
              suitCh,
              style: TextStyle(
                color: fg,
                fontSize: width * 0.30,
                height: 1.0,
              ),
            ),
          ),
          Positioned(
            bottom: 6, right: 6,
            child: Text(
              suitCh,
              style: TextStyle(
                color: fg,
                fontSize: width * 0.55,
                height: 1.0,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _emptySlot(double h) {
    return Container(
      width: width,
      height: h,
      decoration: BoxDecoration(
        color: const Color(0x22FFFFFF),
        borderRadius: BorderRadius.circular(width * 0.12),
        border: Border.all(
          color: const Color(0x55FFFFFF),
          width: 1.0,
          style: BorderStyle.solid,
        ),
      ),
    );
  }

  String _suitGlyph(int suit) {
    switch (suit) {
      case 0: return '♣';
      case 1: return '♦';
      case 2: return '♥';
      case 3: return '♠';
    }
    return '?';
  }
}
