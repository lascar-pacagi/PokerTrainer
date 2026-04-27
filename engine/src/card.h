#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace pt {

// Card index convention: card = rank * 4 + suit, range [0, 52).
// rank: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
// suit: 0=c, 1=d, 2=h, 3=s
using Card = uint8_t;

inline constexpr Card NO_CARD = 0xFF;
inline constexpr int  NUM_RANKS = 13;
inline constexpr int  NUM_SUITS = 4;
inline constexpr int  NUM_CARDS = 52;

inline constexpr int rank_of(Card c) { return c >> 2; }
inline constexpr int suit_of(Card c) { return c & 0x3; }
inline constexpr Card make_card(int rank, int suit) {
    return static_cast<Card>((rank << 2) | suit);
}

inline constexpr std::string_view RANK_CHARS = "23456789TJQKA";
inline constexpr std::string_view SUIT_CHARS = "cdhs";

// "AsKh" → {make_card(12,3), make_card(11,2)}. Returns NO_CARD on parse error.
Card parse_card(std::string_view s);
std::string card_to_string(Card c);

// Convenience: quick 52-bit mask helpers (fits in uint64_t).
using CardMask = uint64_t;
inline constexpr CardMask bit(Card c) { return CardMask{1} << c; }

}  // namespace pt
