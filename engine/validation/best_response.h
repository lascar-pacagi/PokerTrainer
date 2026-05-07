#pragma once
//
// Independent best-response computation against a pt-solver dump.
//
// Algorithm (per BR-er): post-order DFS through the action tree, propagating
// the opponent's per-combo reach probabilities. At each node we compute V[h]
// for the BR-er — the expected value for hand `h` given that the opponent
// follows the strategy embedded in the JSON, and the BR-er plays the
// EV-maximising action.
//
// **Precision regime — read this before interpreting results:**
//
//   * **All 5 board cards pinned (no chance walk):** validator is exact.
//     Agreement with pt-solver's claimed exploitability is at floating-
//     point precision (~1e-6 chips on a 100-pot).
//
//   * **Chance walked (flop or turn pinned only):** validator uses
//     iso-aware BR — derives suit-isomorphism classes from the chance
//     node's board, enumerates the iso variants of each emitted rep
//     card, and picks up each variant's contribution by combo-index
//     remap on the rep's value table. Residuals are typically
//     ~0.01-0.03 chips on a 100-pot for SRP-class spots, traceable
//     to chip-value normalization noise compounded through nested
//     chance levels rather than algorithmic disagreement. The
//     residual is small enough that any real BR-deviation > 0.1%
//     pot would surface clearly.
//
//     Caveats:
//     * Requires σ-symmetric ranges (PokerStove class names like
//       `AKs`, not suit-locked combos like `AhKh`). For asymmetric
//       ranges, σ(combo) may fall outside the range and the
//       corresponding iso variant is dropped — same approximate
//       behaviour pt-solver itself exhibits.
//     * Iso classes are derived from board suit-counts only (suits
//       with the same count are interchangeable), matching the
//       crate's iso strategy.
//
// Recurrence (player p is BR-er, q = opponent of p, h = hand of p):
//   * Terminal:        V[h] = Σ_h_o oppReach[h_o] · payoff(h, h_o, board, kind)
//   * Chance:          V[h] = mean over child cards c of V_child(h)
//   * Action, p moves: V[h] = max_a V_child_a[h]
//   * Action, q moves: V[h] = Σ_a V_child_a[h]      (note: SUM, not weighted —
//     the per-action-per-combo reach factor is applied to oppReach when
//     descending into child_a, so the "weight" is already baked into
//     V_child_a's terminal payoff integrals.)
//
// Exploitability is reported as (V_OOP_BR − V_OOP_eq) + (V_IP_BR − V_IP_eq),
// matching the convention used in the literature (Burch et al. 2014, Brown
// & Sandholm 2017). At true equilibrium the sum is 0; in practice it
// equals what pt-solver reports as `exploitability` to within solver
// convergence error.

#include "hand_eval.h"
#include "scenario.h"

#include <cstdint>
#include <vector>

namespace pt::validation {

struct BRConfig {
    /// Sampling controls for chance-node traversal:
    ///   * 0  → exhaustive (current behaviour). Walks every chance child.
    ///         Exact but O(N_card_branches) per chance node.
    ///   * >0 → Monte Carlo. At each chance node, sample at most this many
    ///         random children uniformly without replacement. Standard
    ///         error scales as 1/√samples; 100 samples ≈ 10% relative,
    ///         10000 ≈ 1%. Sampling makes BR feasible on full river-depth
    ///         dumps for SRP-class spots.
    int      samples = 0;
    /// Seed for the chance-node RNG. Ignored if `samples == 0`.
    /// Defaults to 0 = "use std::random_device" (set explicitly for
    /// reproducibility across runs).
    uint64_t seed    = 0;
};

struct BRResult {
    /// V_BR[h] for OOP, in chips. Computed assuming OOP plays max-EV
    /// at every OOP action node; IP follows the JSON's strategy.
    std::vector<double> oop_br_values;
    std::vector<double> oop_eq_values;
    /// V_BR[h] for IP, computed symmetrically.
    std::vector<double> ip_br_values;
    std::vector<double> ip_eq_values;
    /// Range-weighted mean BR value for OOP (Σ root_weight[h]·V_BR[h] / Σ).
    double              oop_br_aggregate;
    double              ip_br_aggregate;
    /// Range-weighted equilibrium values, taken from the root node's
    /// strategy/ev arrays. These are what pt-solver itself produces.
    double              oop_eq_aggregate;
    double              ip_eq_aggregate;
    /// BR-measured exploitability =
    ///   ((oop_br_aggregate − oop_eq_aggregate)
    /// +  (ip_br_aggregate  − ip_eq_aggregate)) / 2.
    /// **Average** of per-player BR gains, matching pt-solver's
    /// `compute_exploitability` convention (utility.rs:286). At true
    /// equilibrium both deltas are 0; in practice this matches
    /// `Scenario::exploitability` within solver convergence error.
    double              br_exploitability;
    /// Per-player BR gains for diagnostics.
    double              oop_br_gain;     // oop_br_aggregate − oop_eq_aggregate
    double              ip_br_gain;      //  ip_br_aggregate −  ip_eq_aggregate
};

/// Run BR for both players against `scenario`.
///
/// `eval` is reused for showdown computations; share one across many calls.
///
/// Time complexity: O(combos² · num_terminal_nodes) for the showdown
/// integral, plus O(combos · num_action_nodes) for tree traversal.
/// For SRP-class spots (~150 combos × ~3000 nodes) this runs in a few
/// seconds.
BRResult compute_best_response(
    const Scenario&      scenario,
    const HandEvaluator& eval,
    BRConfig             config = {});

}  // namespace pt::validation
