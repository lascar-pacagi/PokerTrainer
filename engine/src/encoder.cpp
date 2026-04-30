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
    // Pure A_DIM=11 one-hot. No state-dependent scalars.
    for (int i = 0; i < A_DIM; ++i) out[i] = 0.f;
    out[static_cast<int>(a)] = 1.f;
}

void encode_history_row(float* out,
                        const AppliedAction& act,
                        int64_t pot_before_chips,
                        bool actor_was_us) {
    // v0.4 layout (HIST_FEAT = 20):
    //   [0]      is_real (1.0 — bit 0 of every populated row)
    //   [1..11]  action one-hot (NUM_ACTIONS slots)
    //   [12]     bet_to_bb
    //   [13]     bet_frac_pot   (vs pot BEFORE this action; 0 if pot was 0)
    //   [14]     pot_after_bb
    //   [15]     stack_after_bb (from AppliedAction.stack_after_chips)
    //   [16]     was_all_in
    //   [17]     is_raise
    //   [18..19] actor flags (us / villain) relative to current to_act
    //
    // Removed vs v0.3: per-row street one-hot (slot implies street under
    // fixed-position layout) and pos_norm (slot implies position).
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

    // ─── fixed-position action history ──────────────────────────────────────
    // Each action is written into its street's sub-block at the per-street
    // counter offset. Slot k of street s ALWAYS means "the kth action of
    // street s" — no chronological mixing across streets.
    //
    // Padded slots (within a sub-block, after that street's last real action)
    // stay all-zero — is_real = 0 distinguishes them.
    //
    // Budget-overflow path: if any street has more actions than its slot
    // budget allows, throw. Fail-fast so a wrong budget assumption surfaces
    // immediately rather than silently dropping rows.
    const int64_t blinds_pot = HUState::SMALL_BLIND_CHIPS + HUState::BIG_BLIND_CHIPS;
    std::array<int, 4> per_street_count = {0, 0, 0, 0};

    for (std::size_t j = 0; j < s.history.size(); ++j) {
        const AppliedAction& act = s.history[j];
        const int street_idx = static_cast<int>(act.street);
        if (street_idx < 0 || street_idx >= 4) {
            throw std::runtime_error("encode(): history action has invalid street");
        }

        const int slot = per_street_count[street_idx];
        if (slot >= STREET_SLOTS[street_idx]) {
            throw std::runtime_error(
                "encode(): per-street action budget exceeded — bump the "
                "relevant *_SLOTS constant in encoder.h (street=" +
                std::to_string(street_idx) + ", slot=" +
                std::to_string(slot) + ", history_size=" +
                std::to_string(s.history.size()) + ")");
        }

        const int64_t pot_before = (j == 0) ? blinds_pot : s.history[j - 1].pot_after_chips;
        float* row = &x[STREET_OFFSETS[street_idx] + slot * HIST_FEAT];
        const bool actor_was_us = (act.actor == s.to_act);
        encode_history_row(row, act, pot_before, actor_was_us);

        per_street_count[street_idx] = slot + 1;
    }

    return e;
}

}  // namespace pt
