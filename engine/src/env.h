#pragma once

#include "action.h"
#include "encoder.h"
#include "game_hu.h"
#include "hand_eval.h"

#include <array>
#include <cstdint>
#include <memory>

namespace pt {

// Gym-style wrapper around HUState.
//
// Conventions:
//   - reset() deals a fresh hand using the next RNG draw. The Env holds its
//     own mt19937_64 seeded in the constructor; this gives reproducible hand
//     streams across actors.
//   - observation() returns the EncodedState for the player currently to act.
//     Illegal to call on a terminal state (throws). The `legal_idx` field
//     doubles as the action mask for the policy head.
//   - step() applies an action either by index into the legal-actions vector
//     of the last observation, or directly by ActionType.
//   - Mid-hand rewards are zero. Terminal reward is the chip delta in big
//     blinds for the player who just acted (signed). The counter-party's
//     reward is the negation; DouZero-style trajectories typically back-fill
//     per-player terminal returns from `payoffs_bb()` at hand end.
class Env {
public:
    struct StepResult {
        bool    done;
        Player  just_acted;
        double  reward_bb;   // 0 mid-hand; signed chip delta / BB on terminal
    };

    explicit Env(uint64_t seed = 0xC0FFEE,
                 int64_t  starting_stack_chips = HUState::BIG_BLIND_CHIPS * HUState::DEFAULT_STACK_BB);
    ~Env();
    Env(Env&&) noexcept;
    Env& operator=(Env&&) noexcept;
    Env(const Env&)            = delete;
    Env& operator=(const Env&) = delete;

    // Start a new hand. If `hand_seed` is null, uses the Env's internal RNG.
    void reset();
    void reset(uint64_t hand_seed);

    // Encoded state for the player-to-act. Throws if terminal.
    EncodedState observation() const;

    // Apply by index into the most-recent `observation().legal` vector.
    StepResult step(int legal_action_idx);

    // Apply by ActionType (bypasses reliance on a cached observation).
    StepResult step_action(ActionType a);

    const HUState& state() const { return state_; }

    // Mutable view of the underlying HUState, intended for synthetic spot
    // construction (overwriting hole/board cards from a UI). Use with care:
    // mutating chip-tracking fields or the action history will desync the
    // RNG-deal-then-replay invariants. The inspector use case overwrites
    // cards immediately after a fresh reset() and before the first step,
    // which is safe.
    HUState& mutable_state() { return state_; }

    bool is_terminal() const { return state_.is_terminal(); }
    Player to_act() const { return state_.to_act; }

    // Chip delta in BB for each player after a terminal hand. Both zero mid-hand.
    std::array<double, NUM_PLAYERS> payoffs_bb() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    HUState state_;
};

}  // namespace pt
