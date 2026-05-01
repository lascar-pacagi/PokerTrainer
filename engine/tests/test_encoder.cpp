#include <catch2/catch_test_macros.hpp>

#include "action.h"
#include "encoder.h"
#include "game_hu.h"

#include <algorithm>
#include <array>

using namespace pt;

namespace {

// History-row offsets within a single HIST_FEAT-wide row (v0.5; same width
// as v0.4). See encoder.cpp::encode_history_row for the canonical
// definition.
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

    // No truncation on a fresh deal.
    for (int st = 0; st < 4; ++st) {
        REQUIRE(e.x[X_OFF_HIST_TRUNCATED + st] == 0.f);
    }
}

TEST_CASE("v0.5 dim sanity", "[encoder]") {
    REQUIRE(HIST_FEAT == 20);
    REQUIRE(HIST_MAX  == 34);
    REQUIRE(STATIC_DIM == 136);
    REQUIRE(HIST_DIM  == HIST_MAX * HIST_FEAT);
    REQUIRE(X_DIM     == STATIC_DIM + HIST_DIM);
    REQUIRE(X_DIM     == 816);
    REQUIRE(STREET_SLOTS[0] == 10);
    REQUIRE(STREET_SLOTS[1] == 8);
    REQUIRE(STREET_SLOTS[2] == 8);
    REQUIRE(STREET_SLOTS[3] == 8);
    REQUIRE(STREET_OFFSETS[0] == 136);
    REQUIRE(STREET_OFFSETS[1] == 136 + 10 * HIST_FEAT);
    REQUIRE(STREET_OFFSETS[2] == 136 + (10 + 8) * HIST_FEAT);
    REQUIRE(STREET_OFFSETS[3] == 136 + (10 + 8 + 8) * HIST_FEAT);
    REQUIRE(X_OFF_HIST_TRUNCATED == 132);
    REQUIRE(HIST_TRUNC_DIM == 4);
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

TEST_CASE("History rows: slot 0 is the MOST RECENT action of its street",
          "[encoder]") {
    auto s = HUGame::deal(21);
    HUGame::step(s, ActionType::RAISE_100);    // preflop SB open
    HUGame::step(s, ActionType::CHECK_CALL);   // preflop BB call → flop
    HUGame::step(s, ActionType::CHECK_CALL);   // flop BB checks
    REQUIRE(s.street == Street::FLOP);
    REQUIRE(s.to_act == Player::SB);

    const auto e = encode(s);

    // Preflop sub-block:
    //   slot 0 = MOST RECENT preflop action = BB's call
    //   slot 1 = SB's open
    //   slots 2..9 padded
    const float* p0 = row_at(e, Street::PREFLOP, 0);
    REQUIRE(p0[HOFF_IS_REAL] == 1.f);
    REQUIRE(p0[HOFF_ACTION_ONEHOT + static_cast<int>(ActionType::CHECK_CALL)] == 1.f);
    REQUIRE(p0[HOFF_IS_RAISE]   == 0.f);
    // Current to_act is SB; this row is BB's action → was_us = false.
    REQUIRE(p0[HOFF_ACTOR_US]   == 0.f);
    REQUIRE(p0[HOFF_ACTOR_VILL] == 1.f);

    const float* p1 = row_at(e, Street::PREFLOP, 1);
    REQUIRE(p1[HOFF_IS_REAL] == 1.f);
    REQUIRE(p1[HOFF_ACTION_ONEHOT + static_cast<int>(ActionType::RAISE_100)] == 1.f);
    REQUIRE(p1[HOFF_IS_RAISE]   == 1.f);
    // SB's action, current to_act SB → was_us = true.
    REQUIRE(p1[HOFF_ACTOR_US]   == 1.f);
    REQUIRE(p1[HOFF_ACTOR_VILL] == 0.f);

    for (int slot = 2; slot < PREFLOP_SLOTS; ++slot) {
        const float* r = row_at(e, Street::PREFLOP, slot);
        for (int j = 0; j < HIST_FEAT; ++j) REQUIRE(r[j] == 0.f);
    }

    // Flop sub-block: slot 0 = BB's check (only flop action so far).
    const float* f0 = row_at(e, Street::FLOP, 0);
    REQUIRE(f0[HOFF_IS_REAL] == 1.f);
    REQUIRE(f0[HOFF_ACTION_ONEHOT + static_cast<int>(ActionType::CHECK_CALL)] == 1.f);
    REQUIRE(f0[HOFF_ACTOR_US]   == 0.f);
    REQUIRE(f0[HOFF_ACTOR_VILL] == 1.f);

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

    // No truncation flags.
    for (int st = 0; st < 4; ++st) {
        REQUIRE(e.x[X_OFF_HIST_TRUNCATED + st] == 0.f);
    }
}

TEST_CASE("History: 4-action preflop fills slots 0..3 newest-first",
          "[encoder]") {
    auto s = HUGame::deal(99);
    HUGame::step(s, ActionType::RAISE_100);    // SB open
    HUGame::step(s, ActionType::RAISE_100);    // BB 3bet
    HUGame::step(s, ActionType::RAISE_100);    // SB 4bet
    HUGame::step(s, ActionType::CHECK_CALL);   // BB call → flop
    REQUIRE(s.street == Street::FLOP);

    const auto e = encode(s);

    // slot 0 = BB's call (most recent), slot 3 = SB's open (oldest).
    REQUIRE(row_at(e, Street::PREFLOP, 0)
            [HOFF_ACTION_ONEHOT + static_cast<int>(ActionType::CHECK_CALL)] == 1.f);
    REQUIRE(row_at(e, Street::PREFLOP, 1)
            [HOFF_ACTION_ONEHOT + static_cast<int>(ActionType::RAISE_100)] == 1.f);
    REQUIRE(row_at(e, Street::PREFLOP, 2)
            [HOFF_ACTION_ONEHOT + static_cast<int>(ActionType::RAISE_100)] == 1.f);
    REQUIRE(row_at(e, Street::PREFLOP, 3)
            [HOFF_ACTION_ONEHOT + static_cast<int>(ActionType::RAISE_100)] == 1.f);

    // Slots 4..9 zero. Flop block all zero (no flop action yet).
    for (int slot = 4; slot < PREFLOP_SLOTS; ++slot) {
        const float* r = row_at(e, Street::PREFLOP, slot);
        for (int j = 0; j < HIST_FEAT; ++j) REQUIRE(r[j] == 0.f);
    }
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

    // preflop[0] = most recent = SB's all-in.
    REQUIRE(r0[HOFF_IS_REAL] == 1.f);
    REQUIRE(r0[HOFF_ACTION_ONEHOT + static_cast<int>(ActionType::ALL_IN)] == 1.f);
    REQUIRE(r0[HOFF_STACK_AFTER] == 0.f);
    REQUIRE(r0[HOFF_WAS_ALL_IN]  == 1.f);

    // Sanity: bet_to_bb > 0 and pot_after_bb is non-trivial.
    REQUIRE(r0[HOFF_BET_TO]    > 0.f);
    REQUIRE(r0[HOFF_POT_AFTER] > 0.f);
}

TEST_CASE("Per-street truncation: oldest dropped, flag fires, slot 0 still "
          "is the most recent", "[encoder]") {
    // Synthetically construct an HUState whose preflop history has MORE
    // entries than PREFLOP_SLOTS. Direct field manipulation — doesn't need
    // the actions to be reachable through legal play.
    auto s = HUGame::deal(7);
    s.history.clear();

    constexpr int N = PREFLOP_SLOTS + 3;   // 13 actions on preflop
    // Use distinct ActionTypes so we can verify which actions survived.
    // Cycle through CHECK_CALL/RAISE_100 to keep is_raise meaningful.
    for (int i = 0; i < N; ++i) {
        AppliedAction a{};
        a.actor             = (i % 2 == 0) ? Player::SB : Player::BB;
        a.street            = Street::PREFLOP;
        a.type              = (i == N - 1) ? ActionType::ALL_IN
                            : (i % 2 == 0  ? ActionType::RAISE_100
                                           : ActionType::CHECK_CALL);
        a.bet_to_chips      = 100 + i * 50;
        a.pot_after_chips   = 200 + i * 100;
        a.stack_after_chips = 5000 - i * 100;
        a.was_all_in        = (i == N - 1);
        s.history.push_back(a);
    }

    const auto e = encode(s);

    // Truncation bit fired for preflop only.
    REQUIRE(e.x[X_OFF_HIST_TRUNCATED + static_cast<int>(Street::PREFLOP)] == 1.f);
    REQUIRE(e.x[X_OFF_HIST_TRUNCATED + static_cast<int>(Street::FLOP)]    == 0.f);
    REQUIRE(e.x[X_OFF_HIST_TRUNCATED + static_cast<int>(Street::TURN)]    == 0.f);
    REQUIRE(e.x[X_OFF_HIST_TRUNCATED + static_cast<int>(Street::RIVER)]   == 0.f);

    // Slot 0 = most recent preflop action = the all-in.
    const float* p0 = row_at(e, Street::PREFLOP, 0);
    REQUIRE(p0[HOFF_IS_REAL]    == 1.f);
    REQUIRE(p0[HOFF_ACTION_ONEHOT + static_cast<int>(ActionType::ALL_IN)] == 1.f);
    REQUIRE(p0[HOFF_WAS_ALL_IN] == 1.f);

    // Slots 1..9 also populated (the most-recent 9 actions before the all-in).
    for (int slot = 1; slot < PREFLOP_SLOTS; ++slot) {
        const float* r = row_at(e, Street::PREFLOP, slot);
        REQUIRE(r[HOFF_IS_REAL] == 1.f);
    }

    // The earliest 3 actions were dropped; we kept N-PREFLOP_SLOTS=3 fewer.
    // Verify that the bet_to_bb of slot PREFLOP_SLOTS-1 (oldest kept) is
    // greater than what slot 0 would have been if we hadn't truncated.
    // Concretely: slot 0 is action N-1 (last), slot k is action N-1-k.
    // So slot PREFLOP_SLOTS-1 is action (N-1)-(PREFLOP_SLOTS-1) = N-PREFLOP_SLOTS = 3.
    // Action 3 had bet_to_chips = 100 + 3*50 = 250.
    const float* p_last = row_at(e, Street::PREFLOP, PREFLOP_SLOTS - 1);
    REQUIRE(p_last[HOFF_BET_TO] > 0.f);
    // action 3 had pot_after_chips = 200 + 3*100 = 500 → 5.0 bb.
    REQUIRE(std::abs(p_last[HOFF_POT_AFTER] - 5.0f) < 1e-3f);
}

TEST_CASE("Multi-street truncation flags are independent", "[encoder]") {
    auto s = HUGame::deal(11);
    s.history.clear();

    // PREFLOP: 12 actions (overflow by 2).
    for (int i = 0; i < PREFLOP_SLOTS + 2; ++i) {
        AppliedAction a{};
        a.actor   = Player::SB;
        a.street  = Street::PREFLOP;
        a.type    = ActionType::CHECK_CALL;
        a.bet_to_chips = 100;
        a.pot_after_chips = 200;
        s.history.push_back(a);
    }
    // FLOP: 3 actions (under budget — no truncation).
    for (int i = 0; i < 3; ++i) {
        AppliedAction a{};
        a.actor   = Player::BB;
        a.street  = Street::FLOP;
        a.type    = ActionType::CHECK_CALL;
        a.bet_to_chips = 100;
        a.pot_after_chips = 300;
        s.history.push_back(a);
    }
    // TURN: 9 actions (overflow by 1).
    for (int i = 0; i < TURN_SLOTS + 1; ++i) {
        AppliedAction a{};
        a.actor   = Player::SB;
        a.street  = Street::TURN;
        a.type    = ActionType::CHECK_CALL;
        a.bet_to_chips = 100;
        a.pot_after_chips = 400;
        s.history.push_back(a);
    }

    const auto e = encode(s);

    REQUIRE(e.x[X_OFF_HIST_TRUNCATED + 0] == 1.f);  // preflop
    REQUIRE(e.x[X_OFF_HIST_TRUNCATED + 1] == 0.f);  // flop
    REQUIRE(e.x[X_OFF_HIST_TRUNCATED + 2] == 1.f);  // turn
    REQUIRE(e.x[X_OFF_HIST_TRUNCATED + 3] == 0.f);  // river
}

TEST_CASE("Encoder refuses terminal state", "[encoder]") {
    auto s = HUGame::deal(1);
    HUGame::step(s, ActionType::FOLD);
    REQUIRE(s.is_terminal());
    REQUIRE_THROWS(encode(s));
}
