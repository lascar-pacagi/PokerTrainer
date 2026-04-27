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
// Flat-sequence layout (v0.3): `a` rows are pure 11-dim action one-hots
// (no scalars). `x` carries an explicit 11-bit legal-actions mask between
// the chip-state scalars and the history block. History rows are still
// HIST_MAX × HIST_FEAT, but bit 0 of each row is now `is_real` — the
// external valid-mask block has been removed.
inline constexpr int A_DIM             = 11;   // per-action feature vector (pure one-hot, NUM_ACTIONS slots)
inline constexpr int HIST_MAX          = 24;   // max actions per hand we encode
inline constexpr int HIST_FEAT         = 25;   // per-history-row feature width
                                               // (1 is_real + 11 action one-hot + 3 chip scalars +
                                               //  1 stack_after + 1 was_all_in + 1 is_raise +
                                               //  2 actor + 4 street + 1 pos_norm)
inline constexpr int LEGAL_MASK_DIM    = 11;   // x carries a state-level legal-actions mask
inline constexpr int STATIC_DIM        = 132;  // hole/board/street/pos/scalars + LEGAL_MASK_DIM
inline constexpr int HIST_DIM          = HIST_MAX * HIST_FEAT;  // 600
inline constexpr int X_DIM             = STATIC_DIM + HIST_DIM;  // 732
inline constexpr int N_ACTIONS_CLIP    = 6;    // n_actions_this_street clip for x[]

// Within-x region offsets (handy constants for both tests and callers).
inline constexpr int X_OFF_LEGAL_MASK  = 121;             // 11-bit legal-actions mask
inline constexpr int X_OFF_HIST        = STATIC_DIM;       // 132 — start of history rows

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

// Encode a single candidate action as a pure A_DIM-float one-hot. Does not
// depend on state under v0.3 — kept as a free function for callers that
// already pass an action and want a raw row.
void encode_action_vector(float* out, ActionType a);

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
