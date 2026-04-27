#include <catch2/catch_test_macros.hpp>

#include "action.h"
#include "env.h"
#include "game_hu.h"

using namespace pt;

TEST_CASE("Env reset produces observation with preflop SB to act", "[env]") {
    Env env(42);
    const auto obs = env.observation();
    REQUIRE(env.to_act() == Player::SB);
    REQUIRE(env.state().street == Street::PREFLOP);
    REQUIRE(!obs.legal.empty());
}

TEST_CASE("Env step by legal index matches step by action", "[env]") {
    Env a(123), b(123);
    const auto obs_a = a.observation();
    const auto idx = 0;  // first legal
    const auto act = obs_a.legal[idx];
    const auto ra = a.step(idx);
    const auto rb = b.step_action(act);
    REQUIRE(ra.done == rb.done);
    REQUIRE(ra.just_acted == rb.just_acted);
    REQUIRE(ra.reward_bb == rb.reward_bb);
    REQUIRE(a.state().pot_chips == b.state().pot_chips);
}

TEST_CASE("Terminal step reports reward and payoff", "[env]") {
    Env env(7);
    // SB folds.
    const auto r = env.step_action(ActionType::FOLD);
    REQUIRE(r.done);
    REQUIRE(r.just_acted == Player::SB);
    // SB loses 0.5 BB.
    REQUIRE(r.reward_bb == -0.5);
    const auto pays = env.payoffs_bb();
    REQUIRE(pays[0] == -0.5);
    REQUIRE(pays[1] ==  0.5);
}

TEST_CASE("Playing to showdown produces zero-sum payoffs", "[env]") {
    Env env(2024);
    while (!env.is_terminal()) {
        const auto obs = env.observation();
        // Always pick CHECK_CALL if legal, else the first legal.
        int idx = 0;
        for (int i = 0; i < static_cast<int>(obs.legal.size()); ++i) {
            if (obs.legal[i] == ActionType::CHECK_CALL) { idx = i; break; }
        }
        env.step(idx);
    }
    const auto pays = env.payoffs_bb();
    REQUIRE(pays[0] + pays[1] == 0.0);
}

TEST_CASE("Env is reproducible given same seed", "[env]") {
    Env a(2026), b(2026);
    // Replay a random-ish script; we just need identical trajectories.
    for (int step = 0; step < 3 && !a.is_terminal(); ++step) {
        const auto legal = a.state().legal_actions();
        const auto act = legal[step % legal.size()];
        a.step_action(act);
        b.step_action(act);
    }
    REQUIRE(a.state().pot_chips == b.state().pot_chips);
    REQUIRE(a.state().street == b.state().street);
}

TEST_CASE("reset(seed) overrides internal RNG", "[env]") {
    Env a(1), b(999);
    a.reset(42);
    b.reset(42);
    // Same hand seed → same hole cards and board.
    REQUIRE(a.state().hole[0] == b.state().hole[0]);
    REQUIRE(a.state().hole[1] == b.state().hole[1]);
    REQUIRE(a.state().board   == b.state().board);
}
