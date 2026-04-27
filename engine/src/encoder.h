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
// Flat-sequence layout (v0.2): the action history lives inside `x` as
// HIST_MAX rows × HIST_FEAT floats, followed by a HIST_MAX-bit valid-mask.
// There is no longer a separate `z` tensor.
inline constexpr int A_DIM             = 17;   // per-action feature vector (11 one-hot + 6 scalars)
inline constexpr int HIST_MAX          = 24;   // max actions per hand we encode
inline constexpr int HIST_FEAT         = 25;   // per-history-row feature width (A_DIM + 2 actor + 4 street + 1 pos_norm + 1 pad)
inline constexpr int STATIC_DIM        = 121;  // hole/board/street/pos/scalars
inline constexpr int HIST_DIM          = HIST_MAX * HIST_FEAT;  // 600
inline constexpr int VALID_DIM         = HIST_MAX;               // 24
inline constexpr int X_DIM             = STATIC_DIM + HIST_DIM + VALID_DIM;  // 745
inline constexpr int N_ACTIONS_CLIP    = 6;    // n_actions_this_street clip for x[]

// Within-x region offsets (handy constants for both tests and callers).
inline constexpr int X_OFF_HIST        = STATIC_DIM;                         // 121
inline constexpr int X_OFF_VALID_MASK  = STATIC_DIM + HIST_DIM;              // 721

// View of the encoded state for a single decision point.
struct EncodedState {
    std::array<float, X_DIM> x{};
    // Per-legal-action features + their ActionType indices. Same order.
    std::vector<std::array<float, A_DIM>> a;
    std::vector<ActionType>               legal;
};

// Build the encoded tensors for the currently-to-act player. `state` must be
// non-terminal. Throws if called on a terminal state.
EncodedState encode(const HUState& state);

// Encode a single candidate action's 16-float vector relative to `state`
// (state is the pre-action state; the function does NOT mutate it).
void encode_action_vector(float* out, const HUState& state, ActionType a);

// Encode a historical action's HIST_FEAT-float row. `pot_before_chips` is the
// pot immediately before this action (from the previous history entry or
// blinds). `actor_was_us` is 1.0 if the historical actor == current to_act.
// `row_idx` is the oldest-first position of this row in `x[X_OFF_HIST..]`.
void encode_history_row(float* out,
                        const AppliedAction& act,
                        int64_t pot_before_chips,
                        bool actor_was_us,
                        int row_idx);

}  // namespace pt
