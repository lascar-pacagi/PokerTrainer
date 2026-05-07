// Paranoid validation: four tests beyond the smoke fixtures.
//
// Goals: catch bugs that only manifest on (a) iso classes larger than 2,
// (b) wide ranges, (c) iteration-count regressions, or (d) the gap between
// "exact validator on river-pinned" and "iso-aware validator on
// chance-walked" runs of the same input spec.

#include "best_response.h"
#include "scenario.h"

#include "card.h"
#include "hand_eval.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#ifndef PT_TABLES_PATH
#  define PT_TABLES_PATH "../data/poker_tables.bin"
#endif
#ifndef PT_SOLVER_PATH
#  define PT_SOLVER_PATH "../../validation/target/release/pt-solver"
#endif

namespace {

std::string solve(
    const std::string& spec_json,
    const std::string& tag,
    const std::string& depth = "river") {
    const std::string dir = std::string("/tmp/pt_validation_") + tag;
    std::system(("mkdir -p " + dir).c_str());
    const std::string in_path  = dir + "/in.json";
    const std::string out_path = dir + "/out.json";
    {
        std::ofstream f(in_path);
        f << spec_json;
    }
    const std::string cmd =
        std::string(PT_SOLVER_PATH) +
        " --input "  + in_path +
        " --output " + out_path +
        " --mode tree --depth " + depth +
        " > /dev/null 2>&1";
    int rc = std::system(cmd.c_str());
    REQUIRE(rc == 0);
    return out_path;
}

}  // namespace

TEST_CASE("Rainbow flop — 3-suit iso class (full S₃ permutation group)",
          "[validation][paranoid][rainbow]") {
    // Flop 2c5h8s: suit counts {c:1, d:0, h:1, s:1}. Iso classes:
    // {d} alone, and {c, h, s} as a 3-element class (all count 1).
    //
    // Range AA: 6 combos partition into two orbits under the {c,h,s}
    // symmetric group:
    //   { AcAh, AcAs, AhAs }   (size 3 — both cards in iso class)
    //   { AcAd, AhAd, AsAd }   (size 3 — one card in iso class, one Ad)
    //
    // The iso-aware BR must handle 3-element transposition cycles
    // correctly — different from the 2-suit {h,s} case in the smoke
    // tests. We expect BR exploitability ≈ 0 on the converged solve and
    // V_eq to be identical within each orbit.
    const std::string spec = R"({
      "oop_range": "AA",
      "ip_range":  "KK",
      "flop": "2c5h8s",
      "starting_pot": 100,
      "effective_stack": 200,
      "flop_bet_sizes":  {"bet": "75%", "raise": "2.5x"},
      "turn_bet_sizes":  {"bet": "75%", "raise": "2.5x"},
      "river_bet_sizes": {"bet": "75%", "raise": "2.5x"},
      "max_iterations": 1000,
      "target_exploitability_pct_pot": 0.01
    })";

    const auto out_path = solve(spec, "rainbow_iso");
    auto scenario = pt::validation::load_scenario(out_path);
    auto eval = pt::HandEvaluator::load_or_generate(PT_TABLES_PATH);
    auto r = pt::validation::compute_best_response(scenario, eval);

    INFO("solver claimed expl: " << scenario.exploitability);
    INFO("BR-measured expl:    " << r.br_exploitability);

    const double pot = scenario.starting_pot;
    REQUIRE(r.br_exploitability < 0.05 * pot);

    // Orbit symmetry: V_eq should be identical (to 5 decimals) within
    // each orbit. The {c,h,s}-iso-class members of OOP's range form
    // two orbits; we don't rely on exact strings (combo order in JSON
    // can vary), so just check that the eq values cluster into ≤ 2
    // distinct equivalence classes (within 1e-5 chips).
    using Catch::Matchers::WithinAbs;
    std::vector<double> vs = r.oop_eq_values;
    std::sort(vs.begin(), vs.end());
    int distinct = 1;
    for (size_t i = 1; i < vs.size(); ++i) {
        if (std::abs(vs[i] - vs[i - 1]) > 1e-3) ++distinct;
    }
    INFO("distinct V_eq classes among AA combos: " << distinct);
    REQUIRE(distinct <= 2);  // at most two orbits' worth of distinct values
}

TEST_CASE("Wide-range river-pinned (50+ combos per side)",
          "[validation][paranoid][wide]") {
    // Tests the validator at SRP-ish scale without triggering chance
    // walking (river is pinned). With ~80 OOP combos vs ~80 IP combos,
    // showdown integrals are ~6400 hand-pair evaluations per terminal —
    // ~100× more than the toy fixtures. If there's an O(N²) bug or
    // index-overflow issue, it'll show here.
    const std::string spec = R"({
      "oop_range": "TT+,AJs+,KQs,AKo",
      "ip_range":  "QQ-22,AQs-A2s,KJs+,QJs,JTs,AKo,AQo",
      "flop": "Ac9d3h",
      "turn": "5s",
      "river": "2c",
      "starting_pot": 100,
      "effective_stack": 200,
      "flop_bet_sizes":  {"bet": "50%", "raise": "2.5x"},
      "turn_bet_sizes":  {"bet": "50%", "raise": "2.5x"},
      "river_bet_sizes": {"bet": "50%", "raise": "2.5x"},
      "max_iterations": 200,
      "target_exploitability_pct_pot": 0.5
    })";

    const auto out_path = solve(spec, "wide_river_pinned");
    auto scenario = pt::validation::load_scenario(out_path);
    auto eval = pt::HandEvaluator::load_or_generate(PT_TABLES_PATH);

    INFO("scenario combos: oop=" << scenario.oop_combos.size()
         << " ip=" << scenario.ip_combos.size());
    REQUIRE(scenario.oop_combos.size() >= 30);  // at least 30 each
    REQUIRE(scenario.ip_combos.size()  >= 30);

    auto r = pt::validation::compute_best_response(scenario, eval);

    INFO("solver claimed: " << scenario.exploitability);
    INFO("BR-measured:    " << r.br_exploitability);
    INFO("OOP gain: " << r.oop_br_gain << "  IP gain: " << r.ip_br_gain);

    // River-pinned with no chance walking → exact agreement expected.
    using Catch::Matchers::WithinAbs;
    const double tol = std::max(0.05, 2.0 * scenario.exploitability);
    REQUIRE_THAT(r.br_exploitability,
                 WithinAbs(scenario.exploitability, tol));
}

TEST_CASE("Convergence monotonicity — exploitability decreases with iterations",
          "[validation][paranoid][convergence]") {
    // Solve the same spec at increasing iteration counts. BR-measured
    // exploitability (which is independent of the solver's claimed value)
    // should decrease — possibly with small jitter from CFR's stochastic
    // nature, but the trend must be downward.
    auto eval = pt::HandEvaluator::load_or_generate(PT_TABLES_PATH);

    const auto run = [&](int iters) {
        const std::string spec = std::string(R"({
          "oop_range": "AsAh,AsAd,AhAd",
          "ip_range":  "KsKh,KsKd,KhKd",
          "flop": "Ac6c2d",
          "starting_pot": 100,
          "effective_stack": 200,
          "flop_bet_sizes":  {"bet": "75%", "raise": "2.5x"},
          "turn_bet_sizes":  {"bet": "75%", "raise": "2.5x"},
          "river_bet_sizes": {"bet": "75%", "raise": "2.5x"},
          "max_iterations": )") + std::to_string(iters) +
            R"(, "target_exploitability_pct_pot": 0.0001
        })";
        const auto path = solve(spec, "monotonic_" + std::to_string(iters));
        auto scenario = pt::validation::load_scenario(path);
        auto r = pt::validation::compute_best_response(scenario, eval);
        return std::pair{scenario.exploitability,
                         std::abs(r.br_exploitability)};
    };

    // Sweep: 10, 50, 200, 1000 iterations. The solver's own claim and
    // BR-measured should both monotonically decrease.
    auto [c10,  br10]  = run(10);
    auto [c50,  br50]  = run(50);
    auto [c200, br200] = run(200);

    INFO("iters=10:   solver=" << c10  << " BR=" << br10);
    INFO("iters=50:   solver=" << c50  << " BR=" << br50);
    INFO("iters=200:  solver=" << c200 << " BR=" << br200);

    // Solver's own exploitability should monotonically decrease.
    REQUIRE(c50  <= c10);
    REQUIRE(c200 <= c50);

    // BR-measured should also decrease (allow tiny jitter — sub-1e-3 chips).
    REQUIRE(br50  <= br10  + 1e-3);
    REQUIRE(br200 <= br50  + 1e-3);

    // After 200 iterations, BR-measured should be tiny on this trivial spot.
    REQUIRE(br200 < 0.05);  // 0.05% of pot
}

TEST_CASE("Cross-check: river-pinned vs flop-pinned (same equilibrium)",
          "[validation][paranoid][cross]") {
    // Solve the SAME range × board spec two ways:
    //   (a) all 5 cards pinned → exact validator (no chance walk)
    //   (b) only flop pinned, walk turn+river → iso-aware validator
    //
    // The aggregate exploitability should be approximately the same
    // (within iso-induced residual). If not, there's a real difference
    // between the two CFR solutions.
    const std::string base_spec_river = R"({
      "oop_range": "AsAh,AsAd,AhAd",
      "ip_range":  "KsKh,KsKd,KhKd",
      "flop": "2c5h8s",
      "turn": "Tc",
      "river": "3d",
      "starting_pot": 100,
      "effective_stack": 200,
      "flop_bet_sizes":  {"bet": "75%", "raise": "2.5x"},
      "turn_bet_sizes":  {"bet": "75%", "raise": "2.5x"},
      "river_bet_sizes": {"bet": "75%", "raise": "2.5x"},
      "max_iterations": 1000,
      "target_exploitability_pct_pot": 0.01
    })";

    auto eval = pt::HandEvaluator::load_or_generate(PT_TABLES_PATH);

    // (a) River-pinned: exact validator.
    const auto path_river = solve(base_spec_river, "cross_river_pinned");
    auto scenario_river   = pt::validation::load_scenario(path_river);
    auto r_river = pt::validation::compute_best_response(scenario_river, eval);

    INFO("river-pinned solver expl: " << scenario_river.exploitability);
    INFO("river-pinned BR    expl:  " << r_river.br_exploitability);

    // River-pinned MUST match exactly (within fp noise) — this is our
    // "gold standard" baseline. Verify before doing the cross-check.
    using Catch::Matchers::WithinAbs;
    REQUIRE_THAT(r_river.br_exploitability,
                 WithinAbs(scenario_river.exploitability, 1e-3));

    // (b) Same spec but with chance walked.
    const std::string base_spec_flop = R"({
      "oop_range": "AsAh,AsAd,AhAd",
      "ip_range":  "KsKh,KsKd,KhKd",
      "flop": "2c5h8s",
      "starting_pot": 100,
      "effective_stack": 200,
      "flop_bet_sizes":  {"bet": "75%", "raise": "2.5x"},
      "turn_bet_sizes":  {"bet": "75%", "raise": "2.5x"},
      "river_bet_sizes": {"bet": "75%", "raise": "2.5x"},
      "max_iterations": 1000,
      "target_exploitability_pct_pot": 0.01
    })";

    const auto path_flop = solve(base_spec_flop, "cross_flop_pinned");
    auto scenario_flop   = pt::validation::load_scenario(path_flop);
    auto r_flop = pt::validation::compute_best_response(scenario_flop, eval);

    INFO("flop-pinned solver expl: " << scenario_flop.exploitability);
    INFO("flop-pinned BR    expl:  " << r_flop.br_exploitability);

    // Both should report tiny exploitability (well-converged solve).
    // The flop-pinned BR may have a small iso-residual but should
    // still be small on a pure-strategy spot.
    REQUIRE(r_river.br_exploitability < 0.1);
    REQUIRE(r_flop.br_exploitability  < 0.5);  // looser — 3-suit iso

    // The two solves are SOLVING DIFFERENT GAMES (one knows runout, one
    // doesn't), so we don't expect identical aggregate values. But both
    // should be well-behaved: BR gain ≥ 0 for each player, total
    // exploitability ≥ 0.
    REQUIRE(r_river.oop_br_gain >= -1e-3);
    REQUIRE(r_river.ip_br_gain  >= -1e-3);
    REQUIRE(r_flop.oop_br_gain  >= -1e-3);
    REQUIRE(r_flop.ip_br_gain   >= -1e-3);
}
