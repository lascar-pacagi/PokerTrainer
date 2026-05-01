#pragma once

#include "action.h"
#include "game_hu.h"

#include <array>
#include <cstdint>
#include <vector>

namespace pt {

// Tensor shapes — MUST agree byte-for-byte with trainer/dmc/features.py.
// See docs/STATE_ENCODING.md for the full field-by-field layout.
//
// Reverse-chronological layout (v0.5): the action-history block is
// partitioned into four contiguous sub-blocks (preflop / flop / turn /
// river). Within each sub-block, slot 0 is the MOST RECENT action of that
// street, slot 1 the next-most-recent, etc. Slots past the actual count are
// zero-padded (is_real = 0).
//
// If a street has more actions than its slot budget, the OLDEST actions are
// dropped (slots 0..budget-1 still hold the most-recent budget actions),
// and a per-street bit in the static block fires so the network can read
// "history truncated on this street." This avoids the v0.4 fail-fast on
// pathological min-raise wars during high-ε exploration.
inline constexpr int A_DIM             = 11;   // per-action feature vector (pure one-hot, NUM_ACTIONS slots)
inline constexpr int HIST_FEAT         = 20;   // per-history-row feature width
                                               // (1 is_real + 11 action one-hot + 3 chip scalars +
                                               //  1 stack_after + 1 was_all_in + 1 is_raise +
                                               //  2 actor)
inline constexpr int LEGAL_MASK_DIM    = 11;   // x carries a state-level legal-actions mask
inline constexpr int HIST_TRUNC_DIM    = 4;    // per-street truncation bit (preflop/flop/turn/river)

// Per-street row budgets. Sum = HIST_MAX. Streets that exceed budget are
// silently truncated (oldest dropped) and the corresponding bit in the
// HIST_TRUNCATED block of x fires. Sizing: 100k uniform-random hands
// observed worst-case [9, 7, 7, 6] per-street action counts at peak ε;
// these budgets sit a safe margin above that.
inline constexpr int PREFLOP_SLOTS     = 10;
inline constexpr int FLOP_SLOTS        = 8;
inline constexpr int TURN_SLOTS        = 8;
inline constexpr int RIVER_SLOTS       = 8;
inline constexpr std::array<int, 4> STREET_SLOTS = {
    PREFLOP_SLOTS, FLOP_SLOTS, TURN_SLOTS, RIVER_SLOTS,
};
inline constexpr int HIST_MAX          = PREFLOP_SLOTS + FLOP_SLOTS + TURN_SLOTS + RIVER_SLOTS;  // 34

inline constexpr int STATIC_DIM        = 136;  // hole/board/street/pos/scalars + LEGAL_MASK + HIST_TRUNCATED
inline constexpr int HIST_DIM          = HIST_MAX * HIST_FEAT;  // 680
inline constexpr int X_DIM             = STATIC_DIM + HIST_DIM;  // 816
inline constexpr int N_ACTIONS_CLIP    = 6;    // n_actions_this_street clip for x[]

// Within-x region offsets (handy constants for both tests and callers).
inline constexpr int X_OFF_LEGAL_MASK     = 121;             // 11-bit legal-actions mask
inline constexpr int X_OFF_HIST_TRUNCATED = 132;             // 4-bit per-street truncation flag
inline constexpr int X_OFF_PREFLOP        = STATIC_DIM;                                          // 136
inline constexpr int X_OFF_FLOP           = X_OFF_PREFLOP + PREFLOP_SLOTS * HIST_FEAT;           // 336
inline constexpr int X_OFF_TURN           = X_OFF_FLOP    + FLOP_SLOTS    * HIST_FEAT;           // 496
inline constexpr int X_OFF_RIVER          = X_OFF_TURN    + TURN_SLOTS    * HIST_FEAT;           // 656
inline constexpr std::array<int, 4> STREET_OFFSETS = {
    X_OFF_PREFLOP, X_OFF_FLOP, X_OFF_TURN, X_OFF_RIVER,
};

// View of the encoded state for a single decision point.
struct EncodedState {
    std::array<float, X_DIM> x{};
    // Per-legal-action features + their ActionType indices. Same order.
    std::vector<std::array<float, A_DIM>> a;
    std::vector<ActionType>               legal;
};

// Build the encoded tensors for the currently-to-act player. `state` must be
// non-terminal. Throws if called on a terminal state, or if any street's
// action count exceeds its slot budget.
EncodedState encode(const HUState& state);

// Encode a single candidate action as a pure A_DIM-float one-hot. Does not
// depend on state — kept as a free function for callers that already pass an
// action and want a raw row.
void encode_action_vector(float* out, ActionType a);

// Encode a historical action's HIST_FEAT-float row. `pot_before_chips` is the
// pot immediately before this action (from the previous history entry or
// blinds). `actor_was_us` is 1.0 if the historical actor == current to_act.
// Slot position is determined by the caller (street block + per-street
// counter), so this function no longer takes a row index.
void encode_history_row(float* out,
                        const AppliedAction& act,
                        int64_t pot_before_chips,
                        bool actor_was_us);

}  // namespace pt
