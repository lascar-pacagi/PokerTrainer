#include "env.h"

#include <memory>
#include <random>
#include <stdexcept>

namespace pt {

struct Env::Impl {
    std::mt19937_64                rng;
    std::shared_ptr<HandEvaluator> evaluator;   // shared across clones
    int64_t                        starting_stack_chips;

    Impl(uint64_t seed, int64_t stack_chips)
        : rng(seed),
          evaluator(std::make_shared<HandEvaluator>(
              HandEvaluator::load_or_generate(""))),
          starting_stack_chips(stack_chips) {}

    // Cheap copy ctor: shared_ptr bumps refcount; mt19937_64 is byte-copied.
    // Used by Env::clone(); explicitly defined so the intent is documented.
    Impl(const Impl& other) = default;
};

Env::Env(uint64_t seed, int64_t starting_stack_chips)
    : impl_(std::make_unique<Impl>(seed, starting_stack_chips)) {
    reset();
}

Env::~Env() = default;
Env::Env(Env&&) noexcept = default;
Env& Env::operator=(Env&&) noexcept = default;

// Builds nothing (no HandEvaluator); clone() fills impl_/state_ right after.
Env::Env(CloneTag) noexcept {}

std::unique_ptr<Env> Env::clone() const {
    // Construct via the private CloneTag ctor, which builds NOTHING — crucially
    // it does NOT run the public ctor, whose Impl build calls
    // HandEvaluator::load_or_generate() (≈2 ms). The old code did
    // `new Env(0, stack)` and threw that fresh evaluator away, paying the full
    // table build on every clone — a silent tax on all CFR traversal. Now we
    // just copy the Impl (shared_ptr evaluator → refcount bump, RNG byte-copy)
    // and the small HUState by value.
    auto out = std::unique_ptr<Env>(new Env(CloneTag{}));
    out->impl_  = std::make_unique<Impl>(*impl_);
    out->state_ = state_;
    return out;
}

void Env::reset() {
    const uint64_t hand_seed = impl_->rng();
    state_ = HUGame::deal(hand_seed, impl_->starting_stack_chips);
}

void Env::reset(uint64_t hand_seed) {
    state_ = HUGame::deal(hand_seed, impl_->starting_stack_chips);
}

EncodedState Env::observation() const {
    return encode(state_);
}

Env::StepResult Env::step_action(ActionType a) {
    if (state_.is_terminal())
        throw std::runtime_error("step on terminal state");
    const Player actor = state_.to_act;
    HUGame::step(state_, a, impl_->evaluator.get());

    StepResult out{};
    out.just_acted = actor;
    out.done       = state_.is_terminal();
    if (out.done) {
        const int ai = static_cast<int>(actor);
        out.reward_bb = static_cast<double>(state_.payoff_chips[ai]) /
                        static_cast<double>(HUState::BIG_BLIND_CHIPS);
    } else {
        out.reward_bb = 0.0;
    }
    return out;
}

Env::StepResult Env::step_raise_to(int64_t bet_to_chips) {
    if (state_.is_terminal())
        throw std::runtime_error("step on terminal state");
    const Player actor = state_.to_act;
    HUGame::step_raise_to(state_, bet_to_chips, impl_->evaluator.get());

    StepResult out{};
    out.just_acted = actor;
    out.done       = state_.is_terminal();
    if (out.done) {
        const int ai = static_cast<int>(actor);
        out.reward_bb = static_cast<double>(state_.payoff_chips[ai]) /
                        static_cast<double>(HUState::BIG_BLIND_CHIPS);
    } else {
        out.reward_bb = 0.0;
    }
    return out;
}

Env::StepResult Env::step(int legal_action_idx) {
    const auto legal = state_.legal_actions();
    if (legal_action_idx < 0 || legal_action_idx >= static_cast<int>(legal.size()))
        throw std::runtime_error("legal_action_idx out of range");
    return step_action(legal[legal_action_idx]);
}

std::array<double, NUM_PLAYERS> Env::payoffs_bb() const {
    std::array<double, NUM_PLAYERS> out{};
    if (!state_.is_terminal()) return out;
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        out[i] = static_cast<double>(state_.payoff_chips[i]) /
                 static_cast<double>(HUState::BIG_BLIND_CHIPS);
    }
    return out;
}

}  // namespace pt
