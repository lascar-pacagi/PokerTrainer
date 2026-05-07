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

TEST_CASE("Env clone is independent of parent", "[env][clone]") {
    Env env(7);
    env.reset(42);
    // Advance two CHECK_CALL preflop actions for a non-trivial state.
    for (int i = 0; i < 2; ++i) {
        const auto obs = env.observation();
        int idx = 0;
        for (size_t k = 0; k < obs.legal.size(); ++k)
            if (obs.legal[k] == ActionType::CHECK_CALL) { idx = static_cast<int>(k); break; }
        env.step(idx);
    }
    REQUIRE(!env.is_terminal());

    const auto hist_before = env.state().history.size();
    const auto pot_before  = env.state().pot_chips;
    const auto to_act_pre  = env.to_act();
    const auto hole_pre    = env.state().hole;
    const auto board_pre   = env.state().board;

    // Drive a clone to terminal; parent must be untouched.
    auto clone = env.clone();
    while (!clone->is_terminal()) {
        const auto obs = clone->observation();
        int idx = 0;
        for (size_t k = 0; k < obs.legal.size(); ++k)
            if (obs.legal[k] == ActionType::CHECK_CALL) { idx = static_cast<int>(k); break; }
        clone->step(idx);
    }
    REQUIRE(clone->is_terminal());

    REQUIRE(env.state().history.size() == hist_before);
    REQUIRE(env.state().pot_chips      == pot_before);
    REQUIRE(env.to_act()               == to_act_pre);
    REQUIRE(!env.is_terminal());
    REQUIRE(env.state().hole  == hole_pre);
    REQUIRE(env.state().board == board_pre);

    // Two clones with the same scripted line → same payoffs.
    auto c2 = env.clone();
    auto c3 = env.clone();
    for (auto* c : {c2.get(), c3.get()}) {
        while (!c->is_terminal()) {
            const auto obs = c->observation();
            int idx = 0;
            for (size_t k = 0; k < obs.legal.size(); ++k)
                if (obs.legal[k] == ActionType::CHECK_CALL) { idx = static_cast<int>(k); break; }
            c->step(idx);
        }
    }
    REQUIRE(c2->payoffs_bb() == c3->payoffs_bb());
    REQUIRE(c2->payoffs_bb() == clone->payoffs_bb());
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
