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

// ─── Env lifecycle ──────────────────────────────────────────────────────────
void* pt_env_create(uint64_t seed, int64_t starting_stack_chips) {
    try {
        auto* b = new EnvBox(seed, starting_stack_chips);
        b->env.reset();  // ensure first observation is ready
        return static_cast<void*>(b);
    } catch (...) {
        return nullptr;
    }
}

void pt_env_destroy(void* handle) {
    delete static_cast<EnvBox*>(handle);
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
