#include <catch2/catch_test_macros.hpp>

#include "action.h"
#include "card.h"
#include "encoder.h"
#include "game_hu.h"

#include <algorithm>
#include <array>
#include <numeric>

using namespace pt;

namespace {

int count_nonzero(const float* v, int n) {
    int k = 0;
    for (int i = 0; i < n; ++i) if (v[i] != 0.f) ++k;
    return k;
}

}  // namespace

TEST_CASE("Encoded x has correct dimension and bit counts", "[encoder]") {
    const auto s = HUGame::deal(42);
    const auto e = encode(s);

    REQUIRE(e.x.size() == X_DIM);

    // Preflop: exactly 2 hole-card bits, 0 board-card bits.
    int hole_bits = 0, board_bits = 0;
    for (int i = 0; i < 52; ++i)  if (e.x[i] > 0.5f) ++hole_bits;
    for (int i = 52; i < 104; ++i) if (e.x[i] > 0.5f) ++board_bits;
    REQUIRE(hole_bits == 2);
    REQUIRE(board_bits == 0);

    // Street one-hot at [104..107]: preflop → index 104 hot.
    REQUIRE(e.x[104] == 1.f);
    REQUIRE(e.x[105] == 0.f);
    REQUIRE(e.x[106] == 0.f);
    REQUIRE(e.x[107] == 0.f);

    // Position one-hot at [108..109]: preflop SB is OOP → [1, 0].
    REQUIRE(e.x[108] == 1.f);
    REQUIRE(e.x[109] == 0.f);

    // Pot is 1.5 BB preflop (SB + BB blinds).
    REQUIRE(e.x[110] == 1.5f);
}

TEST_CASE("Postflop position flips: SB becomes IP", "[encoder]") {
    auto s = HUGame::deal(7);
    HUGame::step(s, ActionType::CHECK_CALL);   // SB limps
    HUGame::step(s, ActionType::CHECK_CALL);   // BB checks → flop
    REQUIRE(s.street == Street::FLOP);
    REQUIRE(s.to_act == Player::BB);

    const auto e = encode(s);
    // BB is to_act on flop; BB is OOP postflop → [1, 0].
    REQUIRE(e.x[108] == 1.f);
    REQUIRE(e.x[109] == 0.f);
    // Street one-hot: flop → index 105.
    REQUIRE(e.x[105] == 1.f);
    REQUIRE(e.x[104] == 0.f);
    // 3 board cards visible.
    int board_bits = 0;
    for (int i = 52; i < 104; ++i) if (e.x[i] > 0.5f) ++board_bits;
    REQUIRE(board_bits == 3);
}

TEST_CASE("Legal-action tensor has one row per legal action", "[encoder]") {
    const auto s = HUGame::deal(123);
    const auto e = encode(s);
    const auto legal = s.legal_actions();
    REQUIRE(e.a.size() == legal.size());
    REQUIRE(e.legal == legal);
    // Each row: exactly one action-type bit set in [0..NUM_ACTIONS-1].
    for (const auto& row : e.a) {
        int sum = 0;
        for (int i = 0; i < NUM_ACTIONS; ++i) if (row[i] == 1.f) ++sum;
        REQUIRE(sum == 1);
    }
}

TEST_CASE("FOLD action vector zeroes out chip-related fields", "[encoder]") {
    auto s = HUGame::deal(9);
    HUGame::step(s, ActionType::RAISE_100);    // SB raises so BB faces a bet
    const auto e = encode(s);
    // Find the row corresponding to FOLD.
    const auto it = std::find(e.legal.begin(), e.legal.end(), ActionType::FOLD);
    REQUIRE(it != e.legal.end());
    const auto idx = std::distance(e.legal.begin(), it);
    const auto& row = e.a[idx];
    REQUIRE(row[static_cast<int>(ActionType::FOLD)] == 1.f);
    REQUIRE(row[NUM_ACTIONS + 0] == 0.f);    // bet_to_bb
    REQUIRE(row[NUM_ACTIONS + 1] == 0.f);    // bet_frac_pot
    REQUIRE(row[NUM_ACTIONS + 5] == 0.f);    // is_raise
}

TEST_CASE("CHECK_CALL action vector tracks resulting stack and pot", "[encoder]") {
    auto s = HUGame::deal(14);
    HUGame::step(s, ActionType::RAISE_100);    // SB raises
    const auto e = encode(s);                  // BB to act
    const auto it = std::find(e.legal.begin(), e.legal.end(), ActionType::CHECK_CALL);
    REQUIRE(it != e.legal.end());
    const auto& row = e.a[std::distance(e.legal.begin(), it)];
    REQUIRE(row[static_cast<int>(ActionType::CHECK_CALL)] == 1.f);
    // Calling matches SB's invested total; bet_to_bb > 1 (since SB bet ≥ 2bb).
    REQUIRE(row[NUM_ACTIONS + 0] > 1.f);
    // pot_after_bb > pot_bb of state.
    REQUIRE(row[NUM_ACTIONS + 2] > e.x[110]);
    REQUIRE(row[NUM_ACTIONS + 5] == 0.f);     // is_raise = 0 for call
}

TEST_CASE("Flat action history: oldest-first rows + valid-mask", "[encoder]") {
    auto s = HUGame::deal(21);
    // RAISE_100 (not RAISE_50) because at preflop-initial the 0.25/0.33/0.50
    // fractions all snap to the min-raise-to of 2bb, so dedup admits only
    // RAISE_25. RAISE_100 produces a distinct 3bb open.
    HUGame::step(s, ActionType::RAISE_100);    // action #0: SB open
    HUGame::step(s, ActionType::CHECK_CALL);   // action #1: BB call → flop
    HUGame::step(s, ActionType::CHECK_CALL);   // action #2: BB checks flop
    REQUIRE(s.to_act == Player::SB);

    const auto e = encode(s);
    auto row = [&](int k) -> const float* {
        return &e.x[X_OFF_HIST + k * HIST_FEAT];
    };

    // Row 0 (oldest): SB's preflop raise. actor_was_us = true (SB to act now).
    REQUIRE(row(0)[static_cast<int>(ActionType::RAISE_100)] == 1.f);
    REQUIRE(row(0)[A_DIM + 0] == 1.f);        // was_us
    REQUIRE(row(0)[A_DIM + 1] == 0.f);
    REQUIRE(row(0)[A_DIM + 2] == 1.f);        // preflop (street one-hot [0])
    REQUIRE(row(0)[A_DIM + 3] == 0.f);
    REQUIRE(row(0)[A_DIM + 6] == 0.f);        // pos_norm = 0/24

    // Row 1: BB's preflop call.
    REQUIRE(row(1)[static_cast<int>(ActionType::CHECK_CALL)] == 1.f);
    REQUIRE(row(1)[A_DIM + 0] == 0.f);        // villain
    REQUIRE(row(1)[A_DIM + 1] == 1.f);
    REQUIRE(row(1)[A_DIM + 2] == 1.f);        // preflop

    // Row 2: BB's flop check — most-recent real row.
    REQUIRE(row(2)[static_cast<int>(ActionType::CHECK_CALL)] == 1.f);
    REQUIRE(row(2)[A_DIM + 0] == 0.f);
    REQUIRE(row(2)[A_DIM + 1] == 1.f);
    REQUIRE(row(2)[A_DIM + 3] == 1.f);        // flop (street one-hot [1])

    // Valid-mask: first 3 bits set, rest zero.
    for (int k = 0; k < 3;         ++k) REQUIRE(e.x[X_OFF_VALID_MASK + k] == 1.f);
    for (int k = 3; k < HIST_MAX;  ++k) REQUIRE(e.x[X_OFF_VALID_MASK + k] == 0.f);

    // Rows 3..HIST_MAX-1 must be fully zero.
    for (int k = 3; k < HIST_MAX; ++k) {
        for (int j = 0; j < HIST_FEAT; ++j) {
            REQUIRE(row(k)[j] == 0.f);
        }
    }
}

TEST_CASE("pos_norm scales with row index", "[encoder]") {
    auto s = HUGame::deal(44);
    // Build a preflop raise-war to get several history rows. First raise is
    // RAISE_100 rather than RAISE_50 because on preflop-initial the 0.50
    // fraction snaps to min-raise and dedup drops it (collides with RAISE_25).
    HUGame::step(s, ActionType::RAISE_100);   // #0 SB open
    HUGame::step(s, ActionType::RAISE_100);   // #1 BB 3bet
    HUGame::step(s, ActionType::RAISE_100);   // #2 SB 4bet
    const auto e = encode(s);
    REQUIRE(e.x[X_OFF_HIST + 0 * HIST_FEAT + A_DIM + 6] ==
            static_cast<float>(0) / static_cast<float>(HIST_MAX));
    REQUIRE(e.x[X_OFF_HIST + 1 * HIST_FEAT + A_DIM + 6] ==
            static_cast<float>(1) / static_cast<float>(HIST_MAX));
    REQUIRE(e.x[X_OFF_HIST + 2 * HIST_FEAT + A_DIM + 6] ==
            static_cast<float>(2) / static_cast<float>(HIST_MAX));
}

TEST_CASE("Encoder refuses terminal state", "[encoder]") {
    auto s = HUGame::deal(1);
    HUGame::step(s, ActionType::FOLD);
    REQUIRE(s.is_terminal());
    REQUIRE_THROWS(encode(s));
}
