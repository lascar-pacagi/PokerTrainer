// C ABI for Flutter dart:ffi (and any other C-speaking host).
//
// Design notes:
//   * Opaque handles (`void*`-shaped pointers) — Dart bindings pack these as
//     `Pointer<Void>` equivalents and never dereference.
//   * Caller-owned output buffers for bulk data (x, a). The host knows sizes
//     from pt_x_dim() / pt_a_dim() / pt_env_n_legal().
//   * Negative returns = error, non-negative = data. No exceptions leak.
//   * Model-query surface is compiled in only when PT_FFI_INFER is defined
//     (requires linking against pt_infer + libtorch). Without it, the
//     model_* symbols still exist as stubs that return -1 so host code can
//     runtime-feature-detect without re-linking.

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "action.h"
#include "card.h"
#include "encoder.h"
#include "env.h"
#include "hand_eval.h"

#ifdef PT_FFI_INFER
#include "inference.h"
#endif

namespace {

// Cache the last observation per Env so pt_env_step(i) maps i→ActionType via
// the same `legal` vector the host saw. Avoids the host re-encoding just to
// step. Invalidated on step/reset.
struct EnvBox {
    pt::Env                           env;
    std::unique_ptr<pt::EncodedState> last_obs;
    bool                              last_obs_valid = false;

    explicit EnvBox(uint64_t seed, int64_t stack)
        : env(seed, stack) {}

    // Wrap an Env constructed elsewhere (e.g. by Env::clone()).
    explicit EnvBox(pt::Env&& env_in)
        : env(std::move(env_in)) {}
};

#ifdef PT_FFI_INFER
struct ModelBox {
    pt::ModelInference model;
    ModelBox(const std::string& path, const std::string& device)
        : model(path, device) {}
};
#endif

inline pt::EncodedState& ensure_obs(EnvBox& box) {
    if (!box.last_obs_valid) {
        box.last_obs        = std::make_unique<pt::EncodedState>(box.env.observation());
        box.last_obs_valid  = true;
    }
    return *box.last_obs;
}

}  // namespace

extern "C" {

// ─── Constants exposed to the host ──────────────────────────────────────────
int32_t pt_x_dim(void)       { return pt::X_DIM;       }
int32_t pt_a_dim(void)       { return pt::A_DIM;       }
int32_t pt_num_actions(void) { return pt::NUM_ACTIONS; }
int32_t pt_hist_max(void)    { return pt::HIST_MAX;    }
int32_t pt_hist_feat(void)   { return pt::HIST_FEAT;   }

// Per-street history sub-block introspection (v0.4 fixed-position layout).
// `street` is 0=PREFLOP, 1=FLOP, 2=TURN, 3=RIVER. Returns the within-x offset
// where that street's row block starts (negative on out-of-range street).
int32_t pt_x_off_street(int32_t street) {
    if (street < 0 || street >= 4) return -1;
    return pt::STREET_OFFSETS[street];
}

// How many history rows are reserved for `street` (the slot budget). Returns
// -1 on out-of-range street.
int32_t pt_street_slots(int32_t street) {
    if (street < 0 || street >= 4) return -1;
    return pt::STREET_SLOTS[street];
}

// ─── Card helpers (kept from Phase-0 stub for cross-checking) ───────────────
int32_t pt_card_to_string(uint8_t card, char* out, int32_t out_cap) {
    const auto s = pt::card_to_string(card);
    if (out_cap < static_cast<int32_t>(s.size())) return -1;
    for (std::size_t i = 0; i < s.size(); ++i) out[i] = s[i];
    return static_cast<int32_t>(s.size());
}

uint8_t pt_make_card(int32_t rank, int32_t suit) {
    if (rank < 0 || rank >= pt::NUM_RANKS || suit < 0 || suit >= pt::NUM_SUITS) {
        return pt::NO_CARD;
    }
    return pt::make_card(rank, suit);
}

// ─── Hand-evaluator tables (process-global) ─────────────────────────────────
static pt::HandEvaluator* g_eval = nullptr;

int32_t pt_eval_init(const char* tables_path) {
    try {
        static pt::HandEvaluator e = pt::HandEvaluator::load_or_generate(tables_path);
        g_eval = &e;
        return 0;
    } catch (...) {
        return -1;
    }
}

int32_t pt_eval_7(const uint8_t* cards7) {
    if (!g_eval) return -1;
    return static_cast<int32_t>(g_eval->evaluate7(cards7));
}

// Category label for a rank ("Straight Flush", "Full House", ...). Writes a
// null-terminated string into out (truncated if out_cap is too small).
// Returns the full length needed (excluding null), so callers can size
// buffers if they want; returns -1 on out-of-range rank.
int32_t pt_eval_rank_category(int32_t rank, char* out, int32_t out_cap) {
    if (rank < 0 || rank > 65535) return -1;
    const auto sv = pt::HandEvaluator::rank_category(static_cast<uint16_t>(rank));
    const int32_t needed = static_cast<int32_t>(sv.size());
    if (out && out_cap > 0) {
        const int32_t copy = std::min(needed, out_cap - 1);
        for (int32_t i = 0; i < copy; ++i) out[i] = sv[i];
        out[copy] = '\0';
    }
    return needed;
}

// ─── Env lifecycle ──────────────────────────────────────────────────────────
void* pt_env_create(uint64_t seed, int64_t starting_stack_chips) {
    try {
        // EnvBox's ctor constructs Env(seed, stack), whose ctor already calls
        // reset() (deals the first hand). A second reset() here would consume an
        // extra RNG draw, so the FFI's first hand for a seed would differ from a
        // pybind/C++ Env(seed) — drop it for cross-binding reproducibility.
        auto* b = new EnvBox(seed, starting_stack_chips);
        return static_cast<void*>(b);
    } catch (...) {
        return nullptr;
    }
}

void pt_env_destroy(void* handle) {
    delete static_cast<EnvBox*>(handle);
}

// Deep-copy an Env (shared HandEvaluator). Returns a fresh handle the caller
// must destroy with pt_env_destroy. The cloned EnvBox starts with no cached
// last_obs (callers should observe before stepping). Returns nullptr on error.
void* pt_env_clone(const void* handle) {
    if (!handle) return nullptr;
    const auto* src = static_cast<const EnvBox*>(handle);
    try {
        auto cloned = src->env.clone();   // std::unique_ptr<pt::Env>
        return static_cast<void*>(new EnvBox(std::move(*cloned)));
    } catch (...) {
        return nullptr;
    }
}

int32_t pt_env_reset(void* handle) {
    if (!handle) return -1;
    auto* b = static_cast<EnvBox*>(handle);
    try {
        b->env.reset();
        b->last_obs_valid = false;
        return 0;
    } catch (...) {
        return -1;
    }
}

int32_t pt_env_reset_seed(void* handle, uint64_t hand_seed) {
    if (!handle) return -1;
    auto* b = static_cast<EnvBox*>(handle);
    try {
        b->env.reset(hand_seed);
        b->last_obs_valid = false;
        return 0;
    } catch (...) {
        return -1;
    }
}

// ─── Env queries ────────────────────────────────────────────────────────────
int32_t pt_env_is_terminal(const void* handle) {
    if (!handle) return -1;
    return static_cast<const EnvBox*>(handle)->env.is_terminal() ? 1 : 0;
}

int32_t pt_env_to_act(const void* handle) {
    if (!handle) return -1;
    const auto* b = static_cast<const EnvBox*>(handle);
    if (b->env.is_terminal()) return -1;
    return static_cast<int32_t>(b->env.to_act());
}

int32_t pt_env_n_legal(void* handle) {
    if (!handle) return -1;
    auto* b = static_cast<EnvBox*>(handle);
    if (b->env.is_terminal()) return 0;
    try {
        return static_cast<int32_t>(ensure_obs(*b).legal.size());
    } catch (...) {
        return -1;
    }
}

// Copies x (X_DIM floats), the flat legal-action feature matrix
// (n_legal × A_DIM floats), and the ActionType indices per legal action.
// Any out_* pointer may be null to skip that output. Returns n_legal or
// negative on error. The host pre-sizes buffers using pt_env_n_legal().
int32_t pt_env_observe(void* handle,
                       float*   out_x,
                       float*   out_a_flat,
                       int32_t* out_legal,
                       int32_t  out_cap_legal) {
    if (!handle) return -1;
    auto* b = static_cast<EnvBox*>(handle);
    if (b->env.is_terminal()) return -1;

    try {
        const auto& obs = ensure_obs(*b);
        const int32_t n = static_cast<int32_t>(obs.legal.size());
        if (out_cap_legal >= 0 && n > out_cap_legal) return -2;

        if (out_x) {
            std::memcpy(out_x, obs.x.data(), pt::X_DIM * sizeof(float));
        }
        if (out_a_flat) {
            for (int32_t i = 0; i < n; ++i) {
                std::memcpy(out_a_flat + static_cast<std::size_t>(i) * pt::A_DIM,
                            obs.a[i].data(),
                            pt::A_DIM * sizeof(float));
            }
        }
        if (out_legal) {
            for (int32_t i = 0; i < n; ++i) {
                out_legal[i] = static_cast<int32_t>(obs.legal[i]);
            }
        }
        return n;
    } catch (...) {
        return -1;
    }
}

// Apply a step by legal-action index (index into the last-observed
// legal vector — matches what pt_env_observe returned).
// If out_reward_bb is non-null, fills the chip-delta-in-BB for the player
// who just acted (signed). Returns 1 if terminal after step, 0 mid-hand,
// negative on error.
int32_t pt_env_step(void* handle, int32_t legal_idx, double* out_reward_bb) {
    if (!handle) return -1;
    auto* b = static_cast<EnvBox*>(handle);
    if (b->env.is_terminal()) return -1;
    try {
        // Refresh obs if caller never read it, so the index is valid.
        (void)ensure_obs(*b);
        const auto sr = b->env.step(legal_idx);
        b->last_obs_valid = false;
        if (out_reward_bb) *out_reward_bb = sr.reward_bb;
        return sr.done ? 1 : 0;
    } catch (...) {
        return -1;
    }
}

// Fills 2 doubles (SB, BB) in BB units. Mid-hand values are zero.
void pt_env_payoffs_bb(const void* handle, double* out_sb_bb, double* out_bb_bb) {
    if (!handle || (!out_sb_bb && !out_bb_bb)) return;
    try {
        const auto p = static_cast<const EnvBox*>(handle)->env.payoffs_bb();
        if (out_sb_bb) *out_sb_bb = p[0];
        if (out_bb_bb) *out_bb_bb = p[1];
    } catch (...) {
        if (out_sb_bb) *out_sb_bb = 0.0;
        if (out_bb_bb) *out_bb_bb = 0.0;
    }
}

// ─── Inspector / replay surface ─────────────────────────────────────────────
// One-shot table-state read for the UI. All chip values are in CHIPS (1 BB =
// 100 chips). Mid-hand payoffs are zero. Layout is fixed so Dart Struct /
// Python ctypes mirrors can pin field offsets.
//
// Field order is int64-block-then-int32-block to keep all int64 fields
// 8-byte-aligned without internal padding holes. Total size = 120 bytes.
struct PtStateSnapshot {
    int64_t  pot_chips;
    int64_t  stacks_chips[2];           // [SB, BB]
    int64_t  invested_this_street[2];   // [SB, BB]
    int64_t  invested_prior_streets[2]; // [SB, BB]
    int64_t  to_call_chips;
    int64_t  min_raise_to_chips;
    int64_t  payoff_chips[2];           // [SB, BB] — zero mid-hand
    int32_t  street;       // 0=PRE, 1=FLOP, 2=TURN, 3=RIVER, 4=SHOWDOWN
    int32_t  to_act;       // 0=SB, 1=BB, -1 if terminal
    int32_t  terminal;     // 0=NONE, 1=FOLD, 2=SHOWDOWN
    int32_t  is_terminal;  // 0/1
    int32_t  all_in[2];    // [SB, BB]
    int32_t  history_size; // total applied actions so far
    int32_t  _padding;     // explicit pad to 8-byte alignment
};
static_assert(sizeof(PtStateSnapshot) == 120, "PtStateSnapshot layout drift");

// One historical action, suitable for rendering the history strip in the UI.
// Total size = 40 bytes.
struct PtHistoryEntry {
    int64_t  bet_to_chips;
    int64_t  pot_after_chips;
    int64_t  stack_after_chips;
    int32_t  actor;          // 0=SB, 1=BB
    int32_t  street;         // 0..3
    int32_t  type;           // ActionType
    int32_t  was_all_in;     // 0/1
};
static_assert(sizeof(PtHistoryEntry) == 40, "PtHistoryEntry layout drift");

// One-shot snapshot of table state. Returns 0 on success, -1 on bad handle.
int32_t pt_env_state_snapshot(const void* handle, PtStateSnapshot* out) {
    if (!handle || !out) return -1;
    try {
        const auto& s = static_cast<const EnvBox*>(handle)->env.state();
        out->pot_chips                   = s.pot_chips;
        out->stacks_chips[0]             = s.stacks[0];
        out->stacks_chips[1]             = s.stacks[1];
        out->invested_this_street[0]     = s.invested_this_street[0];
        out->invested_this_street[1]     = s.invested_this_street[1];
        out->invested_prior_streets[0]   = s.invested_prior_streets[0];
        out->invested_prior_streets[1]   = s.invested_prior_streets[1];
        out->to_call_chips               = s.is_terminal() ? 0 : s.to_call_chips();
        out->min_raise_to_chips          = s.is_terminal() ? 0 : s.min_raise_to_chips();
        out->payoff_chips[0]             = s.payoff_chips[0];
        out->payoff_chips[1]             = s.payoff_chips[1];
        out->street                      = static_cast<int32_t>(s.street);
        out->to_act                      = s.is_terminal() ? -1 : static_cast<int32_t>(s.to_act);
        out->terminal                    = static_cast<int32_t>(s.terminal);
        out->is_terminal                 = s.is_terminal() ? 1 : 0;
        out->all_in[0]                   = s.all_in[0] ? 1 : 0;
        out->all_in[1]                   = s.all_in[1] ? 1 : 0;
        out->history_size                = static_cast<int32_t>(s.history.size());
        out->_padding                    = 0;
        return 0;
    } catch (...) {
        return -1;
    }
}

// Read one history entry by index. Returns 0 on success, -1 on bad handle or
// out-of-range idx. The caller renders the history strip by walking
// 0..history_size-1 from the snapshot.
int32_t pt_env_history_at(const void* handle, int32_t idx, PtHistoryEntry* out) {
    if (!handle || !out) return -1;
    try {
        const auto& s = static_cast<const EnvBox*>(handle)->env.state();
        if (idx < 0 || idx >= static_cast<int32_t>(s.history.size())) return -1;
        const auto& a = s.history[idx];
        out->bet_to_chips      = a.bet_to_chips;
        out->pot_after_chips   = a.pot_after_chips;
        out->stack_after_chips = a.stack_after_chips;
        out->actor             = static_cast<int32_t>(a.actor);
        out->street            = static_cast<int32_t>(a.street);
        out->type              = static_cast<int32_t>(a.type);
        out->was_all_in        = a.was_all_in ? 1 : 0;
        return 0;
    } catch (...) {
        return -1;
    }
}

// For each ActionType slot (0..NUM_ACTIONS-1), fill the absolute bet-to
// amount in chips that THAT slot would resolve to in the current state.
// 0 chips for FOLD, CHECK_CALL fills with the to-call amount, illegal slots
// fill with -1 (so the UI can label "Raise to X BB" without recomputing).
// Returns the number of slots written (NUM_ACTIONS) or -1 on error.
int32_t pt_env_legal_sizings(void* handle, int64_t* out_bet_to_chips) {
    if (!handle || !out_bet_to_chips) return -1;
    auto* b = static_cast<EnvBox*>(handle);
    if (b->env.is_terminal()) return -1;
    try {
        const auto& s = b->env.state();
        const uint16_t lmask = s.legal_actions_mask();
        const int64_t  to_call = s.to_call_chips();
        for (int k = 0; k < pt::NUM_ACTIONS; ++k) {
            const auto a = static_cast<pt::ActionType>(k);
            if (!(lmask & (uint16_t{1} << k))) {
                out_bet_to_chips[k] = -1;
                continue;
            }
            if (a == pt::ActionType::FOLD) {
                out_bet_to_chips[k] = 0;
            } else if (a == pt::ActionType::CHECK_CALL) {
                // Match the current to-call (chips above own current invest).
                out_bet_to_chips[k] = s.invested_this_street[static_cast<int>(s.to_act)] + to_call;
            } else if (a == pt::ActionType::ALL_IN) {
                // Shove effective stack — total invest after = current invest + remaining stack.
                const int pi = static_cast<int>(s.to_act);
                out_bet_to_chips[k] = s.invested_this_street[pi] + s.stacks[pi];
            } else {
                // RAISE_25..RAISE_300 — let the engine resolve fraction.
                const double frac = pt::RAISE_FRACTIONS[
                    k - static_cast<int>(pt::ActionType::RAISE_25)];
                out_bet_to_chips[k] = s.raise_to_from_fraction(frac);
            }
        }
        return pt::NUM_ACTIONS;
    } catch (...) {
        return -1;
    }
}

// Apply by ActionType (not by legal_idx). Mirrors pt_env_step but useful for
// scripted replay where the host already knows the action it wants to take.
// Throws-equivalent (returns -1) on illegal action.
int32_t pt_env_step_action(void* handle, int32_t action_type, double* out_reward_bb) {
    if (!handle) return -1;
    auto* b = static_cast<EnvBox*>(handle);
    if (b->env.is_terminal()) return -1;
    if (action_type < 0 || action_type >= pt::NUM_ACTIONS) return -1;
    try {
        const auto sr = b->env.step_action(static_cast<pt::ActionType>(action_type));
        b->last_obs_valid = false;
        if (out_reward_bb) *out_reward_bb = sr.reward_bb;
        return sr.done ? 1 : 0;
    } catch (...) {
        return -1;
    }
}

// Apply a raise to an explicit chip target (total invested this street by the
// actor after the action). Lets a host mirror an external agent whose bet
// sizes need not match the discrete RAISE_* abstraction (e.g. Slumbot).
// Returns 1 if the hand is terminal after the step, 0 otherwise, -1 on a bad
// handle / terminal env / illegal target.
int32_t pt_env_step_raise_to(void* handle, int64_t bet_to_chips, double* out_reward_bb) {
    if (!handle) return -1;
    auto* b = static_cast<EnvBox*>(handle);
    if (b->env.is_terminal()) return -1;
    try {
        const auto sr = b->env.step_raise_to(bet_to_chips);
        b->last_obs_valid = false;
        if (out_reward_bb) *out_reward_bb = sr.reward_bb;
        return sr.done ? 1 : 0;
    } catch (...) {
        return -1;
    }
}

namespace {

// Validate that the 4 hole + 5 board card slots have no duplicates. NO_CARD
// slots are skipped. Returns true iff every dealt card is unique.
inline bool cards_are_unique(const pt::HUState& s) {
    pt::CardMask mask = 0;
    auto check = [&](pt::Card c) -> bool {
        if (c == pt::NO_CARD) return true;
        if (c >= pt::NUM_CARDS) return false;
        const pt::CardMask bit = pt::bit(c);
        if (mask & bit) return false;
        mask |= bit;
        return true;
    };
    for (int p = 0; p < pt::NUM_PLAYERS; ++p) {
        for (int i = 0; i < 2; ++i) {
            if (!check(s.hole[p][i])) return false;
        }
    }
    for (int i = 0; i < 5; ++i) {
        if (!check(s.board[i])) return false;
    }
    return true;
}

}  // namespace

// Overwrite a player's hole cards. Validates that the new pair is distinct
// AND that combining with the existing other-player hole + board produces no
// duplicates. Returns 0 on success, -1 on validation failure or bad handle.
// Invalidates the cached observation.
int32_t pt_env_set_hole(void* handle, int32_t player,
                        uint8_t c0, uint8_t c1) {
    if (!handle || player < 0 || player >= pt::NUM_PLAYERS) return -1;
    if (c0 == c1 && c0 != pt::NO_CARD) return -1;
    if (c0 != pt::NO_CARD && c0 >= pt::NUM_CARDS) return -1;
    if (c1 != pt::NO_CARD && c1 >= pt::NUM_CARDS) return -1;
    auto* b = static_cast<EnvBox*>(handle);
    try {
        auto& s = b->env.mutable_state();
        const pt::Card prev0 = s.hole[player][0];
        const pt::Card prev1 = s.hole[player][1];
        s.hole[player][0] = c0;
        s.hole[player][1] = c1;
        if (!cards_are_unique(s)) {
            s.hole[player][0] = prev0;
            s.hole[player][1] = prev1;
            return -1;
        }
        b->last_obs_valid = false;
        return 0;
    } catch (...) {
        return -1;
    }
}

// Overwrite the board. cards5 is exactly 5 entries (NO_CARD = 0xFF for unset
// slots). Returns 0 on success, -1 on validation failure.
int32_t pt_env_set_board(void* handle, const uint8_t* cards5) {
    if (!handle || !cards5) return -1;
    for (int i = 0; i < 5; ++i) {
        if (cards5[i] != pt::NO_CARD && cards5[i] >= pt::NUM_CARDS) return -1;
    }
    auto* b = static_cast<EnvBox*>(handle);
    try {
        auto& s = b->env.mutable_state();
        std::array<pt::Card, 5> prev = s.board;
        for (int i = 0; i < 5; ++i) s.board[i] = cards5[i];
        if (!cards_are_unique(s)) {
            s.board = prev;
            return -1;
        }
        b->last_obs_valid = false;
        return 0;
    } catch (...) {
        return -1;
    }
}

// Wipe all 4 hole-card slots and all 5 board slots to NO_CARD. Useful when
// the host wants to set up a synthetic spot from scratch — call clear, then
// pt_env_set_hole / pt_env_set_board can be issued in any order without
// transient duplicate-card conflicts against the original random deal.
// Returns 0 on success, -1 on bad handle. Invalidates the cached observation.
int32_t pt_env_clear_cards(void* handle) {
    if (!handle) return -1;
    auto* b = static_cast<EnvBox*>(handle);
    try {
        auto& s = b->env.mutable_state();
        for (int p = 0; p < pt::NUM_PLAYERS; ++p) {
            s.hole[p][0] = pt::NO_CARD;
            s.hole[p][1] = pt::NO_CARD;
        }
        for (int i = 0; i < 5; ++i) s.board[i] = pt::NO_CARD;
        b->last_obs_valid = false;
        return 0;
    } catch (...) {
        return -1;
    }
}

// Reads. Returns 0 on success, -1 on bad handle / out-of-range player.
int32_t pt_env_get_hole(const void* handle, int32_t player,
                        uint8_t* out_c0, uint8_t* out_c1) {
    if (!handle || player < 0 || player >= pt::NUM_PLAYERS) return -1;
    if (!out_c0 || !out_c1) return -1;
    const auto& s = static_cast<const EnvBox*>(handle)->env.state();
    *out_c0 = s.hole[player][0];
    *out_c1 = s.hole[player][1];
    return 0;
}

// Fills exactly 5 bytes; unrevealed slots will read NO_CARD (0xFF) per
// engine convention only if the engine itself cleared them; for spots
// constructed via pt_env_set_board the caller's 0xFF entries round-trip.
int32_t pt_env_get_board(const void* handle, uint8_t* out_cards5) {
    if (!handle || !out_cards5) return -1;
    const auto& s = static_cast<const EnvBox*>(handle)->env.state();
    for (int i = 0; i < 5; ++i) out_cards5[i] = s.board[i];
    return 0;
}

// ─── Model surface (gated on PT_FFI_INFER at compile time) ─────────────────
// When the inference target isn't linked in, the symbols still exist and
// return -1 / null. Hosts can runtime-detect via pt_model_available().

int32_t pt_model_available(void) {
#ifdef PT_FFI_INFER
    return 1;
#else
    return 0;
#endif
}

void* pt_model_load(const char* path, const char* device) {
#ifdef PT_FFI_INFER
    if (!path) return nullptr;
    try {
        return static_cast<void*>(new ModelBox(
            std::string(path),
            std::string(device ? device : "cpu")));
    } catch (...) {
        return nullptr;
    }
#else
    (void)path; (void)device;
    return nullptr;
#endif
}

void pt_model_destroy(void* handle) {
#ifdef PT_FFI_INFER
    delete static_cast<ModelBox*>(handle);
#else
    (void)handle;
#endif
}

// Scores the current (non-terminal) env observation and returns the argmax
// legal-action index. If out_q is non-null, fills n_legal q-values in the
// same order as pt_env_observe's legal output.
int32_t pt_model_query(void* model_handle, void* env_handle, float* out_q) {
#ifdef PT_FFI_INFER
    if (!model_handle || !env_handle) return -1;
    auto* m = static_cast<ModelBox*>(model_handle);
    auto* b = static_cast<EnvBox*>(env_handle);
    if (b->env.is_terminal()) return -1;
    try {
        const auto& obs = ensure_obs(*b);
        const int n = static_cast<int>(obs.legal.size());
        if (out_q) {
            return m->model.score_and_argmax(obs, out_q);
        }
        // Caller doesn't want q — use a local buffer.
        std::unique_ptr<float[]> buf(new float[n]);
        return m->model.score_and_argmax(obs, buf.get());
    } catch (...) {
        return -1;
    }
#else
    (void)model_handle; (void)env_handle; (void)out_q;
    return -1;
#endif
}

}  // extern "C"
