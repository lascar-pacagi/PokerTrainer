#include "encoder.h"

#include <algorithm>
#include <stdexcept>

namespace pt {
namespace {

inline float bb(int64_t chips) {
    return static_cast<float>(chips) / static_cast<float>(HUState::BIG_BLIND_CHIPS);
}

// Resolve an action's effect on the state WITHOUT mutating.
struct Resolved {
    int64_t bet_to_chips;
    int64_t pot_after_chips;
    int64_t stack_after_chips;
    bool    is_all_in;
    bool    is_raise;
};

Resolved resolve(const HUState& s, ActionType a) {
    const int pi      = static_cast<int>(s.to_act);
    const int64_t me  = s.invested_this_street[pi];
    const int64_t sk  = s.stacks[pi];
    const int64_t tc  = s.to_call_chips();
    Resolved r{};
    r.is_raise = is_raise(a);

    switch (a) {
        case ActionType::FOLD:
            r.bet_to_chips      = me;
            r.pot_after_chips   = s.pot_chips;
            r.stack_after_chips = sk;
            r.is_all_in         = false;
            break;
        case ActionType::CHECK_CALL: {
            const int64_t pay = std::min(tc, sk);
            r.bet_to_chips      = me + pay;
            r.pot_after_chips   = s.pot_chips + pay;
            r.stack_after_chips = sk - pay;
            r.is_all_in         = (r.stack_after_chips == 0 && pay > 0);
            break;
        }
        default: {  // RAISE_* or ALL_IN
            int64_t raise_to;
            if (a == ActionType::ALL_IN) {
                raise_to = me + sk;
            } else {
                const int idx = static_cast<int>(a) - static_cast<int>(ActionType::RAISE_25);
                raise_to = s.raise_to_from_fraction(RAISE_FRACTIONS[idx]);
            }
            const int64_t add = raise_to - me;
            r.bet_to_chips      = raise_to;
            r.pot_after_chips   = s.pot_chips + add;
            r.stack_after_chips = sk - add;
            r.is_all_in         = (r.stack_after_chips == 0);
            break;
        }
    }
    return r;
}

// Core A_DIM-float action-vector fill (shared between legal-action encoding
// and history-row encoding). Matches the `a` layout in docs/STATE_ENCODING.md.
//
// Layout:
//   [0 .. NUM_ACTIONS-1]     action_type one-hot
//   [NUM_ACTIONS + 0]        bet_to_bb
//   [NUM_ACTIONS + 1]        bet_frac_pot
//   [NUM_ACTIONS + 2]        pot_after_bb
//   [NUM_ACTIONS + 3]        stack_after_bb
//   [NUM_ACTIONS + 4]        is_all_in
//   [NUM_ACTIONS + 5]        is_raise
void fill_action_vector(float* out,
                        ActionType a,
                        int64_t bet_to_chips,
                        int64_t pot_before_chips,
                        int64_t pot_after_chips,
                        int64_t stack_after_chips,
                        bool is_all_in_flag,
                        bool is_raise_flag) {
    for (int i = 0; i < A_DIM; ++i) out[i] = 0.f;
    out[static_cast<int>(a)] = 1.f;
    out[NUM_ACTIONS + 0] = bb(bet_to_chips);
    out[NUM_ACTIONS + 1] = pot_before_chips > 0
                ? static_cast<float>(bet_to_chips) / static_cast<float>(pot_before_chips)
                : 0.f;
    out[NUM_ACTIONS + 2] = bb(pot_after_chips);
    out[NUM_ACTIONS + 3] = bb(stack_after_chips);
    out[NUM_ACTIONS + 4] = is_all_in_flag ? 1.f : 0.f;
    out[NUM_ACTIONS + 5] = is_raise_flag  ? 1.f : 0.f;
    if (a == ActionType::FOLD) {
        out[NUM_ACTIONS + 0] = 0.f;
        out[NUM_ACTIONS + 1] = 0.f;
    }
}

}  // anonymous namespace

void encode_action_vector(float* out, const HUState& s, ActionType a) {
    const Resolved r = resolve(s, a);
    fill_action_vector(out, a,
                       r.bet_to_chips,
                       s.pot_chips,
                       r.pot_after_chips,
                       r.stack_after_chips,
                       r.is_all_in,
                       r.is_raise);
}

void encode_history_row(float* out,
                        const AppliedAction& act,
                        int64_t pot_before_chips,
                        bool actor_was_us,
                        int row_idx) {
    for (int i = 0; i < HIST_FEAT; ++i) out[i] = 0.f;
    // [0 .. A_DIM-1] — same as `a` row. History rows zero `stack_after` and
    // `is_all_in` (not reconstructable without replay; layout parity matters).
    fill_action_vector(out, act.type,
                       act.bet_to_chips,
                       pot_before_chips,
                       act.pot_after_chips,
                       /*stack_after_chips=*/0,
                       /*is_all_in_flag=*/false,
                       is_raise(act.type));
    // [A_DIM .. A_DIM+1] — actor one-hot (relative to current to_act player).
    out[A_DIM + 0] = actor_was_us ? 1.f : 0.f;
    out[A_DIM + 1] = actor_was_us ? 0.f : 1.f;
    // [A_DIM+2 .. A_DIM+5] — street one-hot.
    out[A_DIM + 2 + static_cast<int>(act.street)] = 1.f;
    // [A_DIM+6] — normalized row position.
    out[A_DIM + 6] = static_cast<float>(row_idx) / static_cast<float>(HIST_MAX);
    // [A_DIM+7] — reserved pad (zero).
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

    // ─── scalars (11) ───────────────────────────────────────────────────────
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

    // ─── legal actions → `a` tensor ─────────────────────────────────────────
    e.legal = s.legal_actions();
    e.a.resize(e.legal.size());
    for (std::size_t i = 0; i < e.legal.size(); ++i) {
        encode_action_vector(e.a[i].data(), s, e.legal[i]);
    }

    // ─── flat action history (oldest-first) + valid-mask ────────────────────
    // Rows written at x[X_OFF_HIST + k*HIST_FEAT]. If the hand has >HIST_MAX
    // actions (very unlikely — the bound is ~24), we keep the MOST RECENT
    // HIST_MAX so the current decision still sees its own street.
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
        x[X_OFF_VALID_MASK + k] = 1.f;
    }
    // Remaining rows and mask bits stay zero.

    return e;
}

}  // namespace pt
