#include <catch2/catch_test_macros.hpp>

#include "card.h"
#include "hand_eval.h"

#include <array>

using namespace pt;

namespace {
// Build a 7-card hand from human-readable strings.
std::array<Card, 7> parse_hand(std::initializer_list<const char*> s) {
    std::array<Card, 7> out{};
    int i = 0;
    for (const char* p : s) out[i++] = parse_card(p);
    return out;
}

HandEvaluator& evaluator() {
    static HandEvaluator e = HandEvaluator::load_or_generate(PT_TABLES_PATH);
    return e;
}
}  // namespace

TEST_CASE("Royal flush beats everything", "[hand_eval]") {
    auto royal = parse_hand({"As", "Ks", "Qs", "Js", "Ts", "2c", "3d"});
    const uint16_t r = evaluator().evaluate7(royal.data());
    REQUIRE(r == 1);
    REQUIRE(HandEvaluator::rank_category(r) == "Straight Flush");
}

TEST_CASE("Four of a kind category", "[hand_eval]") {
    auto quads = parse_hand({"As", "Ah", "Ad", "Ac", "Kh", "2c", "3d"});
    const uint16_t r = evaluator().evaluate7(quads.data());
    REQUIRE(r >= 2);
    REQUIRE(r <= 166);
    REQUIRE(HandEvaluator::rank_category(r) == "Four of a Kind");
}

TEST_CASE("Full house vs. flush ordering", "[hand_eval]") {
    auto fh    = parse_hand({"As", "Ah", "Ad", "Kc", "Kh", "2c", "3d"});
    auto flush = parse_hand({"As", "Ks", "9s", "5s", "2s", "7d", "8c"});
    REQUIRE(evaluator().evaluate7(fh.data()) < evaluator().evaluate7(flush.data()));
}

TEST_CASE("Straight beats three of a kind", "[hand_eval]") {
    auto straight = parse_hand({"9s", "8h", "7d", "6c", "5h", "2c", "3d"});
    auto trips    = parse_hand({"9s", "9h", "9d", "5c", "2h", "3c", "4d"});
    REQUIRE(evaluator().evaluate7(straight.data()) < evaluator().evaluate7(trips.data()));
}

TEST_CASE("Wheel straight (A-2-3-4-5)", "[hand_eval]") {
    auto wheel = parse_hand({"As", "2h", "3d", "4c", "5h", "Kc", "Qd"});
    const uint16_t r = evaluator().evaluate7(wheel.data());
    REQUIRE(HandEvaluator::rank_category(r) == "Straight");
}

TEST_CASE("High card worst is near 7462", "[hand_eval]") {
    auto high = parse_hand({"7s", "5h", "4d", "3c", "2s", "9c", "Jd"});
    const uint16_t r = evaluator().evaluate7(high.data());
    REQUIRE(r >= 6186);
    REQUIRE(r <= 7462);
}

TEST_CASE("Evaluator is deterministic", "[hand_eval]") {
    auto hand = parse_hand({"Qs", "Qh", "Qd", "Kc", "Kh", "2c", "3d"});
    const auto a = evaluator().evaluate7(hand.data());
    const auto b = evaluator().evaluate7(hand.data());
    REQUIRE(a == b);
}

TEST_CASE("Worst quads beats best full house", "[hand_eval]") {
    // The old product-of-primes scheme, and a hand-rolled Cactus Kev offset
    // with the wrong kicker multiplier, both allow 2222+3 to score worse
    // than AAA+KK. Guard against a regression.
    auto low_quads = parse_hand({"2s", "2h", "2d", "2c", "3h", "4c", "5d"});
    auto best_fh   = parse_hand({"As", "Ah", "Ad", "Kc", "Kh", "3c", "4d"});
    REQUIRE(evaluator().evaluate7(low_quads.data()) <
            evaluator().evaluate7(best_fh.data()));
    REQUIRE(HandEvaluator::rank_category(evaluator().evaluate7(low_quads.data()))
            == "Four of a Kind");
    REQUIRE(HandEvaluator::rank_category(evaluator().evaluate7(best_fh.data()))
            == "Full House");
}

TEST_CASE("Cactus Kev range boundaries are exact", "[hand_eval]") {
    // Royal flush = rank 1.
    auto royal = parse_hand({"As", "Ks", "Qs", "Js", "Ts", "2c", "3d"});
    REQUIRE(evaluator().evaluate7(royal.data()) == 1);

    // Worst SF = 5-high wheel SF; should be 10 in standard numbering.
    auto wheel_sf = parse_hand({"As", "2s", "3s", "4s", "5s", "Kc", "Qd"});
    REQUIRE(evaluator().evaluate7(wheel_sf.data()) == 10);

    // Best quads = AAAA+K → rank 11.
    auto best_quads = parse_hand({"As", "Ah", "Ad", "Ac", "Kh", "2c", "3d"});
    REQUIRE(evaluator().evaluate7(best_quads.data()) == 11);

    // Worst quads = 2222+3 → rank 166. Need all other 3 cards to be 3's so
    // that 2222+3 is the best 5-card subset (not 2222+higher).
    auto worst_quads = parse_hand({"2s", "2h", "2d", "2c", "3s", "3h", "3d"});
    REQUIRE(evaluator().evaluate7(worst_quads.data()) == 166);

    // Best FH = AAA+KK → rank 167.
    auto best_fh = parse_hand({"As", "Ah", "Ad", "Kc", "Kh", "3c", "4d"});
    REQUIRE(evaluator().evaluate7(best_fh.data()) == 167);
}

TEST_CASE("Kicker ordering within a category", "[hand_eval]") {
    // AAAA+K beats AAAA+Q beats AAAA+2.
    auto aaaa_k = parse_hand({"As", "Ah", "Ad", "Ac", "Kh", "2c", "3d"});
    auto aaaa_q = parse_hand({"As", "Ah", "Ad", "Ac", "Qh", "2c", "3d"});
    auto aaaa_2 = parse_hand({"As", "Ah", "Ad", "Ac", "2h", "3c", "4d"});
    REQUIRE(evaluator().evaluate7(aaaa_k.data()) <
            evaluator().evaluate7(aaaa_q.data()));
    REQUIRE(evaluator().evaluate7(aaaa_q.data()) <
            evaluator().evaluate7(aaaa_2.data()));

    // Ace-high flush beats king-high flush, both non-straight.
    auto ahi_flush = parse_hand({"As", "Js", "9s", "5s", "2s", "3c", "4d"});
    auto khi_flush = parse_hand({"Ks", "Js", "9s", "5s", "2s", "3c", "4d"});
    REQUIRE(evaluator().evaluate7(ahi_flush.data()) <
            evaluator().evaluate7(khi_flush.data()));
}
