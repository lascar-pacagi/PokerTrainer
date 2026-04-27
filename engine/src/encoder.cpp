#include "encoder.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace pt {
namespace {

inline float bb(int64_t chips) {
    return static_cast<float>(chips) / static_cast<float>(HUState::BIG_BLIND_CHIPS);
}

}  // anonymous namespace

void encode_action_vector(float* out, ActionType a) {
    // v0.3: pure A_DIM=11 one-hot. No state-dependent scalars.
    for (int i = 0; i < A_DIM; ++i) out[i] = 0.f;
    out[static_cast<int>(a)] = 1.f;
}

void encode_history_row(float* out,
                        const AppliedAction& act,
                        int64_t pot_before_chips,
                        bool actor_was_us,
                        int row_idx) {
    // v0.3 layout (HIST_FEAT = 25):
    //   [0]      is_real (1.0 — bit 0 of every populated row)
    //   [1..11]  action one-hot (NUM_ACTIONS slots)
    //   [12]     bet_to_bb
    //   [13]     bet_frac_pot   (vs pot BEFORE this action; 0 if pot was 0)
    //   [14]     pot_after_bb
    //   [15]     stack_after_bb (from AppliedAction.stack_after_chips)
    //   [16]     was_all_in
    //   [17]     is_raise
    //   [18..19] actor flags (us / villain) relative to current to_act
    //   [20..23] street one-hot (PREFLOP/FLOP/TURN/RIVER)
    //   [24]     pos_norm (row_idx / HIST_MAX)
    for (int i = 0; i < HIST_FEAT; ++i) out[i] = 0.f;

    out[0] = 1.f;
    out[1 + static_cast<int>(act.type)] = 1.f;

    out[12] = bb(act.bet_to_chips);
    out[13] = (pot_before_chips > 0)
                ? static_cast<float>(act.bet_to_chips) /
                  static_cast<float>(pot_before_chips)
                : 0.f;
    out[14] = bb(act.pot_after_chips);
    out[15] = bb(act.stack_after_chips);
    out[16] = act.was_all_in     ? 1.f : 0.f;
    out[17] = is_raise(act.type) ? 1.f : 0.f;

    out[18] = actor_was_us ? 1.f : 0.f;
    out[19] = actor_was_us ? 0.f : 1.f;

    out[20 + static_cast<int>(act.street)] = 1.f;

    out[24] = static_cast<float>(row_idx) / static_cast<float>(HIST_MAX);
}

EncodedState encode(const HUState& s) {
    if (s.is_terminal())
        throw std::runtime_error("encode() called on terminal state");

    EncodedState e;
    float* x = e.x.data();

    const int pi = static_cast<int>(s.to_act);
    const int vi = 1 - pi;

    // ─── hole_cards (52) ────────────────────────────────────────────────────
    for (int c : s.hole[pi]) x[c] = 1.f;

    // ─── board_cards (52) ───────────────────────────────────────────────────
    const int n_board = HUGame::n_visible_board(s.street);
    for (int i = 0; i < n_board; ++i) x[52 + s.board[i]] = 1.f;

    // ─── street one-hot (4) ─────────────────────────────────────────────────
    if (s.street != Street::SHOWDOWN)
        x[104 + static_cast<int>(s.street)] = 1.f;

    // ─── position (2): OOP=BB, IP=SB/BTN ────────────────────────────────────
    const bool we_are_sb = (s.to_act == Player::SB);
    const bool we_are_ip = (s.street == Street::PREFLOP) ? !we_are_sb : we_are_sb;
    x[108] = we_are_ip ? 0.f : 1.f;   // OOP
    x[109] = we_are_ip ? 1.f : 0.f;   // IP

    // ─── chip-state scalars (11) ────────────────────────────────────────────
    const int64_t eff_stack = std::min(s.stacks[0] + s.invested_this_street[0],
                                       s.stacks[1] + s.invested_this_street[1]);
    const int64_t tc        = s.to_call_chips();

    x[110] = bb(s.pot_chips);
    x[111] = bb(s.stacks[pi]);
    x[112] = bb(s.stacks[vi]);
    x[113] = bb(eff_stack);
    x[114] = s.pot_chips > 0
               ? static_cast<float>(eff_stack) / static_cast<float>(s.pot_chips)
               : static_cast<float>(eff_stack);
    x[115] = bb(tc);
    x[116] = s.pot_chips > 0
               ? static_cast<float>(tc) / static_cast<float>(s.pot_chips)
               : 0.f;
    x[117] = bb(s.invested_this_street[pi]);
    x[118] = bb(s.invested_this_street[vi]);
    x[119] = static_cast<float>(std::min(s.actions_this_street, N_ACTIONS_CLIP));
    x[120] = (s.actions_this_street == 0) ? 1.f : 0.f;

    // ─── legal-actions mask (11) ────────────────────────────────────────────
    // x[X_OFF_LEGAL_MASK + k] = 1.0 iff ActionType(k) is legal in this state.
    const uint16_t lmask = s.legal_actions_mask();
    for (int k = 0; k < LEGAL_MASK_DIM; ++k) {
        x[X_OFF_LEGAL_MASK + k] = (lmask & (uint16_t{1} << k)) ? 1.f : 0.f;
    }

    // ─── legal actions → `a` tensor (pure one-hot rows) ─────────────────────
    e.legal = s.legal_actions();
    e.a.resize(e.legal.size());
    for (std::size_t i = 0; i < e.legal.size(); ++i) {
        encode_action_vector(e.a[i].data(), e.legal[i]);
    }

    // ─── flat action history (oldest-first) ─────────────────────────────────
    // Rows written at x[X_OFF_HIST + k*HIST_FEAT]. Bit 0 of each populated
    // row is `is_real = 1`; padded rows stay all-zero with is_real = 0,
    // which is how the network distinguishes real vs padding under v0.3
    // (no separate valid-mask block).
    const int64_t blinds_pot = HUState::SMALL_BLIND_CHIPS + HUState::BIG_BLIND_CHIPS;
    const int hist_n = static_cast<int>(s.history.size());
    const int start  = std::max(0, hist_n - HIST_MAX);
    const int take   = hist_n - start;
    for (int k = 0; k < take; ++k) {
        const int j = start + k;
        const AppliedAction& act = s.history[j];
        const int64_t pot_before = (j == 0) ? blinds_pot : s.history[j - 1].pot_after_chips;
        float* row = &x[X_OFF_HIST + k * HIST_FEAT];
        const bool actor_was_us = (act.actor == s.to_act);
        encode_history_row(row, act, pot_before, actor_was_us, k);
    }

    return e;
}

}  // namespace pt
