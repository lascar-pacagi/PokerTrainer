#include <catch2/catch_test_macros.hpp>

#include "action.h"
#include "encoder.h"
#include "game_hu.h"

#include <algorithm>
#include <array>

using namespace pt;

namespace {

// History-row offsets within a single HIST_FEAT-wide row (v0.4).
// See encoder.cpp::encode_history_row for the canonical definition.
constexpr int HOFF_IS_REAL      = 0;
constexpr int HOFF_ACTION_ONEHOT= 1;   // bits 1..11
constexpr int HOFF_BET_TO       = 12;
constexpr int HOFF_BET_FRAC_POT = 13;
constexpr int HOFF_POT_AFTER    = 14;
constexpr int HOFF_STACK_AFTER  = 15;
constexpr int HOFF_WAS_ALL_IN   = 16;
constexpr int HOFF_IS_RAISE     = 17;
constexpr int HOFF_ACTOR_US     = 18;
constexpr int HOFF_ACTOR_VILL   = 19;

inline const float* row_at(const EncodedState& e, Street st, int slot) {
    return &e.x[STREET_OFFSETS[static_cast<int>(st)] + slot * HIST_FEAT];
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

TEST_CASE("v0.4 dim sanity", "[encoder]") {
    REQUIRE(HIST_FEAT == 20);
    REQUIRE(HIST_MAX  == 34);
    REQUIRE(STATIC_DIM == 132);
    REQUIRE(HIST_DIM  == HIST_MAX * HIST_FEAT);
    REQUIRE(X_DIM     == STATIC_DIM + HIST_DIM);
    REQUIRE(STREET_SLOTS[0] == 10);
    REQUIRE(STREET_SLOTS[1] == 8);
    REQUIRE(STREET_SLOTS[2] == 8);
    REQUIRE(STREET_SLOTS[3] == 8);
    REQUIRE(STREET_OFFSETS[0] == 132);
    REQUIRE(STREET_OFFSETS[1] == 132 + 10 * HIST_FEAT);
    REQUIRE(STREET_OFFSETS[2] == 132 + (10 + 8) * HIST_FEAT);
    REQUIRE(STREET_OFFSETS[3] == 132 + (10 + 8 + 8) * HIST_FEAT);
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

TEST_CASE("Legal-actions mask block matches the legal action set", "[encoder]") {
    const auto s = HUGame::deal(123);
    const auto e = encode(s);
    const auto legal = s.legal_actions();

    // x[X_OFF_LEGAL_MASK + k] == 1.0 iff ActionType(k) is legal.
    for (int k = 0; k < LEGAL_MASK_DIM; ++k) {
        const bool legal_k = std::find(legal.begin(), legal.end(),
                                       static_cast<ActionType>(k)) != legal.end();
        const float bit = e.x[X_OFF_LEGAL_MASK + k];
        REQUIRE(bit == (legal_k ? 1.f : 0.f));
    }
}

TEST_CASE("a-rows are pure 11-dim one-hots", "[encoder]") {
    const auto s = HUGame::deal(123);
    const auto e = encode(s);
    const auto legal = s.legal_actions();
    REQUIRE(e.a.size() == legal.size());
    REQUIRE(e.legal == legal);
    // Each row: exactly one bit set (the action's slot), all others zero.
    for (std::size_t i = 0; i < e.a.size(); ++i) {
        const auto& row = e.a[i];
        int sum = 0;
        for (int k = 0; k < A_DIM; ++k) {
            if (row[k] == 1.f) ++sum;
            else REQUIRE(row[k] == 0.f);
        }
        REQUIRE(sum == 1);
        REQUIRE(row[static_cast<int>(legal[i])] == 1.f);
    }
}

TEST_CASE("History rows: fixed-slot positioning across streets", "[encoder]") {
    auto s = HUGame::deal(21);
    // RAISE_100 (not RAISE_50) because at preflop-initial the 0.25/0.33/0.50
    // fractions all snap to min-raise, so dedup admits only RAISE_25.
    HUGame::step(s, ActionType::RAISE_100);    // preflop[0] SB open
    HUGame::step(s, ActionType::CHECK_CALL);   // preflop[1] BB call → flop
    HUGame::step(s, ActionType::CHECK_CALL);   // flop[0] BB checks
    REQUIRE(s.street == Street::FLOP);
    REQUIRE(s.to_act == Player::SB);

    const auto e = encode(s);

    // preflop[0]: SB open. Current to_act is SB → was_us = true.
    const float* p0 = row_at(e, Street::PREFLOP, 0);
    REQUIRE(p0[HOFF_IS_REAL] == 1.f);
    REQUIRE(p0[HOFF_ACTION_ONEHOT + static_cast<int>(ActionType::RAISE_100)] == 1.f);
    REQUIRE(p0[HOFF_IS_RAISE]   == 1.f);
    REQUIRE(p0[HOFF_ACTOR_US]   == 1.f);
    REQUIRE(p0[HOFF_ACTOR_VILL] == 0.f);

    // preflop[1]: BB call.
    const float* p1 = row_at(e, Street::PREFLOP, 1);
    REQUIRE(p1[HOFF_IS_REAL] == 1.f);
    REQUIRE(p1[HOFF_ACTION_ONEHOT + static_cast<int>(ActionType::CHECK_CALL)] == 1.f);
    REQUIRE(p1[HOFF_IS_RAISE]   == 0.f);
    REQUIRE(p1[HOFF_ACTOR_US]   == 0.f);
    REQUIRE(p1[HOFF_ACTOR_VILL] == 1.f);

    // preflop slots 2..7 must all be zero (padding).
    for (int slot = 2; slot < PREFLOP_SLOTS; ++slot) {
        const float* r = row_at(e, Street::PREFLOP, slot);
        for (int j = 0; j < HIST_FEAT; ++j) REQUIRE(r[j] == 0.f);
    }

    // flop[0]: BB's check. Current to_act on flop is SB → BB action → was_us = false.
    const float* f0 = row_at(e, Street::FLOP, 0);
    REQUIRE(f0[HOFF_IS_REAL] == 1.f);
    REQUIRE(f0[HOFF_ACTION_ONEHOT + static_cast<int>(ActionType::CHECK_CALL)] == 1.f);
    REQUIRE(f0[HOFF_ACTOR_US]   == 0.f);
    REQUIRE(f0[HOFF_ACTOR_VILL] == 1.f);

    // flop slots 1..5 zero, turn block all zero, river block all zero.
    for (int slot = 1; slot < FLOP_SLOTS; ++slot) {
        const float* r = row_at(e, Street::FLOP, slot);
        for (int j = 0; j < HIST_FEAT; ++j) REQUIRE(r[j] == 0.f);
    }
    for (int slot = 0; slot < TURN_SLOTS; ++slot) {
        const float* r = row_at(e, Street::TURN, slot);
        for (int j = 0; j < HIST_FEAT; ++j) REQUIRE(r[j] == 0.f);
    }
    for (int slot = 0; slot < RIVER_SLOTS; ++slot) {
        const float* r = row_at(e, Street::RIVER, slot);
        for (int j = 0; j < HIST_FEAT; ++j) REQUIRE(r[j] == 0.f);
    }
}

TEST_CASE("Street block boundary: flop starts cleanly after preflop", "[encoder]") {
    // Build a 4-action preflop, then flop.
    auto s = HUGame::deal(99);
    HUGame::step(s, ActionType::RAISE_100);    // preflop[0] SB open
    HUGame::step(s, ActionType::RAISE_100);    // preflop[1] BB 3bet
    HUGame::step(s, ActionType::RAISE_100);    // preflop[2] SB 4bet
    HUGame::step(s, ActionType::CHECK_CALL);   // preflop[3] BB call → flop
    REQUIRE(s.street == Street::FLOP);

    const auto e = encode(s);

    // First 4 preflop slots are real.
    for (int slot = 0; slot < 4; ++slot) {
        REQUIRE(row_at(e, Street::PREFLOP, slot)[HOFF_IS_REAL] == 1.f);
    }
    // Slot 4 is the first all-zero preflop slot.
    for (int slot = 4; slot < PREFLOP_SLOTS; ++slot) {
        const float* r = row_at(e, Street::PREFLOP, slot);
        for (int j = 0; j < HIST_FEAT; ++j) REQUIRE(r[j] == 0.f);
    }
    // Flop block start (offset 292) is fully zero — no spillover.
    for (int slot = 0; slot < FLOP_SLOTS; ++slot) {
        const float* r = row_at(e, Street::FLOP, slot);
        for (int j = 0; j < HIST_FEAT; ++j) REQUIRE(r[j] == 0.f);
    }
}

TEST_CASE("History rows: stack_after_bb and was_all_in are populated", "[encoder]") {
    auto s = HUGame::deal(33);
    HUGame::step(s, ActionType::ALL_IN);    // SB shoves all-in preflop
    REQUIRE(s.to_act == Player::BB);

    const auto e = encode(s);
    const float* r0 = row_at(e, Street::PREFLOP, 0);

    // preflop[0] records SB's all-in. Stack after = 0, was_all_in = 1.
    REQUIRE(r0[HOFF_IS_REAL] == 1.f);
    REQUIRE(r0[HOFF_ACTION_ONEHOT + static_cast<int>(ActionType::ALL_IN)] == 1.f);
    REQUIRE(r0[HOFF_STACK_AFTER] == 0.f);
    REQUIRE(r0[HOFF_WAS_ALL_IN]  == 1.f);

    // Sanity: bet_to_bb > 0 and pot_after_bb is non-trivial.
    REQUIRE(r0[HOFF_BET_TO]    > 0.f);
    REQUIRE(r0[HOFF_POT_AFTER] > 0.f);
}

TEST_CASE("Per-street budget overflow throws", "[encoder]") {
    // Synthetically construct an HUState whose preflop history has more
    // entries than PREFLOP_SLOTS allows. Direct field manipulation —
    // doesn't require the action to be reachable through legal play.
    auto s = HUGame::deal(7);
    AppliedAction dummy{};
    dummy.actor              = Player::SB;
    dummy.street             = Street::PREFLOP;
    dummy.type               = ActionType::CHECK_CALL;
    dummy.bet_to_chips       = 0;
    dummy.pot_after_chips    = HUState::SMALL_BLIND_CHIPS + HUState::BIG_BLIND_CHIPS;
    dummy.stack_after_chips  = s.stacks[0];
    dummy.was_all_in         = false;
    s.history.clear();
    for (int i = 0; i < PREFLOP_SLOTS + 1; ++i) s.history.push_back(dummy);

    REQUIRE_THROWS(encode(s));
}

TEST_CASE("Encoder refuses terminal state", "[encoder]") {
    auto s = HUGame::deal(1);
    HUGame::step(s, ActionType::FOLD);
    REQUIRE(s.is_terminal());
    REQUIRE_THROWS(encode(s));
}
