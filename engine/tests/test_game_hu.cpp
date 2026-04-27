#include <catch2/catch_test_macros.hpp>

#include "action.h"
#include "card.h"
#include "game_hu.h"
#include "hand_eval.h"

#include <array>

using namespace pt;

namespace {

HandEvaluator& evaluator() {
    static HandEvaluator e = HandEvaluator::load_or_generate("");
    return e;
}

// Shorthand: total chips committed by player p (all streets).
int64_t total_committed(const HUState& s, int p) {
    return s.invested_prior_streets[p] + s.invested_this_street[p];
}

}  // namespace

TEST_CASE("Initial deal posts blinds and SB to act", "[game_hu]") {
    const auto s = HUGame::deal(42);
    REQUIRE(s.street == Street::PREFLOP);
    REQUIRE(s.to_act == Player::SB);
    REQUIRE(s.invested_this_street[0] == HUState::SMALL_BLIND_CHIPS);
    REQUIRE(s.invested_this_street[1] == HUState::BIG_BLIND_CHIPS);
    REQUIRE(s.pot_chips == HUState::SMALL_BLIND_CHIPS + HUState::BIG_BLIND_CHIPS);
    REQUIRE(s.stacks[0] + s.invested_this_street[0] == s.starting_stacks[0]);
    REQUIRE(s.stacks[1] + s.invested_this_street[1] == s.starting_stacks[1]);
    REQUIRE_FALSE(s.is_terminal());
}

TEST_CASE("Dealt cards are unique and fill 9 slots", "[game_hu]") {
    const auto s = HUGame::deal(12345);
    std::array<bool, NUM_CARDS> seen{};
    auto mark = [&](Card c) {
        REQUIRE(c < NUM_CARDS);
        REQUIRE_FALSE(seen[c]);
        seen[c] = true;
    };
    mark(s.hole[0][0]); mark(s.hole[0][1]);
    mark(s.hole[1][0]); mark(s.hole[1][1]);
    for (int i = 0; i < 5; ++i) mark(s.board[i]);
}

TEST_CASE("SB folds preflop → BB wins 1 BB", "[game_hu]") {
    auto s = HUGame::deal(7);
    HUGame::step(s, ActionType::FOLD);
    REQUIRE(s.terminal == Terminal::FOLD);
    // BB wins SB's matched contribution (0.5 BB).
    REQUIRE(s.payoff_chips[0] == -HUState::SMALL_BLIND_CHIPS);
    REQUIRE(s.payoff_chips[1] ==  HUState::SMALL_BLIND_CHIPS);
    REQUIRE(s.payoff_chips[0] + s.payoff_chips[1] == 0);
}

TEST_CASE("Limp-check goes to flop with BB first to act", "[game_hu]") {
    auto s = HUGame::deal(99);
    // SB calls (limps in from 50 to 100).
    HUGame::step(s, ActionType::CHECK_CALL);
    REQUIRE(s.street == Street::PREFLOP);         // BB still has option
    REQUIRE(s.to_act == Player::BB);
    REQUIRE(s.invested_this_street[0] == 100);

    // BB checks — preflop closes.
    HUGame::step(s, ActionType::CHECK_CALL);
    REQUIRE(s.street == Street::FLOP);
    REQUIRE(s.to_act == Player::BB);              // OOP acts first postflop
    REQUIRE(s.invested_this_street[0] == 0);
    REQUIRE(s.invested_this_street[1] == 0);
    REQUIRE(s.pot_chips == 200);
}

TEST_CASE("Check-check each postflop street runs to showdown", "[game_hu]") {
    auto s = HUGame::deal(1001);
    // preflop: limp, check
    HUGame::step(s, ActionType::CHECK_CALL);
    HUGame::step(s, ActionType::CHECK_CALL);
    REQUIRE(s.street == Street::FLOP);
    // flop: check, check
    HUGame::step(s, ActionType::CHECK_CALL);
    HUGame::step(s, ActionType::CHECK_CALL);
    REQUIRE(s.street == Street::TURN);
    // turn: check, check
    HUGame::step(s, ActionType::CHECK_CALL);
    HUGame::step(s, ActionType::CHECK_CALL);
    REQUIRE(s.street == Street::RIVER);
    // river: check, check → showdown
    HUGame::step(s, ActionType::CHECK_CALL);
    HUGame::step(s, ActionType::CHECK_CALL, &evaluator());
    REQUIRE(s.terminal == Terminal::SHOWDOWN);
    // Zero-sum.
    REQUIRE(s.payoff_chips[0] + s.payoff_chips[1] == 0);
}

TEST_CASE("Preflop raise-call-check-check then fold is legal", "[game_hu]") {
    auto s = HUGame::deal(55);
    // SB raises pot.
    REQUIRE(s.action_is_legal(ActionType::RAISE_100));
    HUGame::step(s, ActionType::RAISE_100);
    REQUIRE(s.to_act == Player::BB);
    REQUIRE(s.invested_this_street[0] > HUState::BIG_BLIND_CHIPS);
    // BB calls.
    HUGame::step(s, ActionType::CHECK_CALL);
    REQUIRE(s.street == Street::FLOP);
    REQUIRE(s.invested_this_street[0] == 0);
    REQUIRE(s.invested_this_street[1] == 0);
    // Flop check-check.
    HUGame::step(s, ActionType::CHECK_CALL);
    HUGame::step(s, ActionType::CHECK_CALL);
    REQUIRE(s.street == Street::TURN);
    // Turn: BB leads, SB folds.
    HUGame::step(s, ActionType::RAISE_75);
    HUGame::step(s, ActionType::FOLD);
    REQUIRE(s.terminal == Terminal::FOLD);
    REQUIRE(s.payoff_chips[0] + s.payoff_chips[1] == 0);
}

TEST_CASE("Fold is illegal when check is available", "[game_hu]") {
    auto s = HUGame::deal(33);
    HUGame::step(s, ActionType::CHECK_CALL);  // SB limps
    // BB facing no bet → fold not legal.
    REQUIRE_FALSE(s.action_is_legal(ActionType::FOLD));
}

TEST_CASE("All-in shove terminates at showdown after running board", "[game_hu]") {
    auto s = HUGame::deal(2024);
    HUGame::step(s, ActionType::ALL_IN);       // SB shoves
    REQUIRE(s.all_in[0]);
    REQUIRE(s.action_is_legal(ActionType::CHECK_CALL));
    HUGame::step(s, ActionType::CHECK_CALL, &evaluator());  // BB calls all-in
    REQUIRE(s.terminal == Terminal::SHOWDOWN);
    // Both should have contributed matched amounts (HU equal stacks).
    REQUIRE(total_committed(s, 0) == total_committed(s, 1));
    REQUIRE(s.payoff_chips[0] + s.payoff_chips[1] == 0);
}

TEST_CASE("Payoffs are zero-sum after every fold/showdown", "[game_hu]") {
    for (uint64_t seed = 1; seed < 50; ++seed) {
        auto s = HUGame::deal(seed);
        // Play a dumb strategy: SB min-raise, BB call, then check-check to river.
        // RAISE_25 is the canonical min-raise slot post-dedup (RAISE_33 collides
        // with it on preflop-initial and is dropped from the legal mask).
        HUGame::step(s, ActionType::RAISE_25);
        HUGame::step(s, ActionType::CHECK_CALL);
        while (!s.is_terminal()) {
            HUGame::step(s, ActionType::CHECK_CALL, &evaluator());
        }
        REQUIRE(s.payoff_chips[0] + s.payoff_chips[1] == 0);
        // Payoff magnitude = matched contribution (both equal here).
        REQUIRE(std::abs(s.payoff_chips[0]) <= s.starting_stacks[0]);
    }
}

TEST_CASE("Legal actions mask excludes raises when facing shove with no chips behind", "[game_hu]") {
    auto s = HUGame::deal(101);
    HUGame::step(s, ActionType::ALL_IN);   // SB all-in
    // BB can only fold or call.
    const auto legal = s.legal_actions();
    REQUIRE(std::find(legal.begin(), legal.end(), ActionType::FOLD) != legal.end());
    REQUIRE(std::find(legal.begin(), legal.end(), ActionType::CHECK_CALL) != legal.end());
    for (int i = static_cast<int>(ActionType::RAISE_25);
         i <= static_cast<int>(ActionType::ALL_IN); ++i) {
        REQUIRE(std::find(legal.begin(), legal.end(), static_cast<ActionType>(i)) == legal.end());
    }
}
