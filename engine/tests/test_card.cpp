#include <catch2/catch_test_macros.hpp>

#include "card.h"

using namespace pt;

TEST_CASE("Card roundtrip", "[card]") {
    for (int r = 0; r < NUM_RANKS; ++r) {
        for (int s = 0; s < NUM_SUITS; ++s) {
            const Card c = make_card(r, s);
            REQUIRE(rank_of(c) == r);
            REQUIRE(suit_of(c) == s);
            REQUIRE(card_to_string(parse_card(card_to_string(c))) == card_to_string(c));
        }
    }
}

TEST_CASE("Card parsing basics", "[card]") {
    REQUIRE(parse_card("As") == make_card(12, 3));
    REQUIRE(parse_card("2c") == make_card(0, 0));
    REQUIRE(parse_card("Td") == make_card(8, 1));
    REQUIRE(parse_card("X?") == NO_CARD);
    REQUIRE(parse_card("A")  == NO_CARD);
    REQUIRE(parse_card("")   == NO_CARD);
}

TEST_CASE("Card uniqueness", "[card]") {
    bool seen[NUM_CARDS] = {false};
    for (int r = 0; r < NUM_RANKS; ++r)
        for (int s = 0; s < NUM_SUITS; ++s) {
            Card c = make_card(r, s);
            REQUIRE(c < NUM_CARDS);
            REQUIRE_FALSE(seen[c]);
            seen[c] = true;
        }
}
