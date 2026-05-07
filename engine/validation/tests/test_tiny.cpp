// Closed-form sanity tests for the BR validator.
//
// These run a tiny pt-solver job, then check the BR validator agrees with
// pt-solver's claimed exploitability. Tiny games admit hand-checkable
// answers (AA vs KK on a board where AA always wins is a pure-strategy
// equilibrium with EV gap of zero in either direction).
//
// We shell out to pt-solver rather than vendoring its binary output —
// keeps the dev cycle simple at the cost of requiring pt-solver to be
// built. CMake passes its expected path via PT_SOLVER_PATH.

#include "best_response.h"
#include "scenario.h"

#include "card.h"
#include "hand_eval.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
#ifndef PT_VALIDATION_DIR
#  define PT_VALIDATION_DIR "."
#endif

namespace {

/// Write a SpotInput JSON to disk, run pt-solver --depth river on it,
/// return the path to the output JSON.
std::string solve(const std::string& spec_json, const std::string& tag) {
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
        " --mode tree --depth river > /dev/null 2>&1";
    int rc = std::system(cmd.c_str());
    REQUIRE(rc == 0);
    return out_path;
}

}  // namespace

TEST_CASE("AA vs KK on AKQ rainbow river — AA always wins; zero exploitability",
          "[validation][tiny]") {
    // Pure-strategy equilibrium: AA bets pot, KK has 0 equity, KK should
    // fold. BR should agree the result is non-exploitable.
    //
    // To keep the tree TINY, we pin all three streets and make the pot
    // close to all-in. With turn=Ks river=Qd we have AKQ-suited-broken:
    // every relevant AA combo beats every KK combo at showdown.
    const std::string spec = R"({
      "oop_range": "AsAh,AsAd,AhAd",
      "ip_range": "KsKh,KsKd,KhKd",
      "flop": "Ac6c2d",
      "turn": "Ks",
      "river": "Qd",
      "starting_pot": 100,
      "effective_stack": 200,
      "flop_bet_sizes":  {"bet": "75%", "raise": "2.5x"},
      "turn_bet_sizes":  {"bet": "75%", "raise": "2.5x"},
      "river_bet_sizes": {"bet": "75%", "raise": "2.5x"},
      "max_iterations": 100,
      "target_exploitability_pct_pot": 0.5
    })";

    const auto out_path = solve(spec, "aa_vs_kk_pure");
    auto scenario = pt::validation::load_scenario(out_path);
    auto eval     = pt::HandEvaluator::load_or_generate(PT_TABLES_PATH);
    auto r        = pt::validation::compute_best_response(scenario, eval);

    INFO("solver claimed exploitability: " << scenario.exploitability);
    INFO("BR-measured exploitability:    " << r.br_exploitability);
    INFO("OOP eq=" << r.oop_eq_aggregate << " br=" << r.oop_br_aggregate);
    INFO("IP  eq=" << r.ip_eq_aggregate  << " br=" << r.ip_br_aggregate);

    // Both BR gains should be small — KK has no positive deviation, AA's
    // strategy is also forced. The exploitability should be at most a few
    // percent of pot.
    const double pot = scenario.starting_pot;
    REQUIRE(r.oop_br_gain   < 0.05 * pot);
    REQUIRE(r.ip_br_gain    < 0.05 * pot);
    REQUIRE(r.br_exploitability < 0.05 * pot);

    // Solver and BR should AGREE within a tighter band: within ~2× the
    // solver's own claimed exploitability (allows for floating-point and
    // CFR convergence noise).
    using Catch::Matchers::WithinAbs;
    const double tol = std::max(0.5, 2.0 * scenario.exploitability);
    REQUIRE_THAT(r.br_exploitability,
                 WithinAbs(scenario.exploitability, tol));
}

TEST_CASE("Polar vs bluff-catcher river — mixed equilibrium",
          "[validation][tiny][mixed]") {
    // OOP has half AA (nuts) + half 22 (busted bluffs). IP has KK
    // (pure bluff-catcher). On A22-specific river (turn=Ad river=2d so the
    // board is AcKc2dAd2d but this might tie with AA — hmm let me pick a
    // simpler board). Use river=2c to give 22 a full house but it's still
    // beat by AA's higher full house.
    //
    // Actually for clarity let's use a board where AA = nut full house
    // and 22 = lower full house, KK = third pair. Board: Ac6h2d 2h 6d.
    // AA: AsAh on this board → A6622 + A → AA full of 6s, very strong.
    //   But hmm, 2h on board makes board 2-pair already; AA = aces
    //   full of 2s? AA + 22 in board = AA22 + 6 = full house aces full.
    //   Let me just say it's strong.
    // Compared to KK: KK + Ac62 26 → KK with 2s pair: just two pair.
    // Compared to 22: 22 with board 2622A2d6d → quads twos. Wait that's
    // even better. Let me avoid quads with a different board.
    //
    // Simpler: turn river both blanks. Board: Ad6h2c 8s Js (no pair).
    // AA = top pair on Ad: pair of aces.
    // 22 = third pair: 2s.
    // KK = second pair: kings.
    // KK > 22 but loses to AA. AA wins everything.
    //
    // Now AA vs KK is the "polar vs bluff-catcher" structure: AA has
    // value, KK has bluff-catcher equity vs 22 only. With OOP holding
    // half AA and half 22, OOP polarizes; KK must call enough to deny
    // bluffs but not too much to avoid paying off value.
    const std::string spec = R"({
      "oop_range": "AsAh,2c2s",
      "ip_range": "KsKh",
      "flop": "Ad6h2c",
      "turn": "8s",
      "river": "Js",
      "starting_pot": 100,
      "effective_stack": 200,
      "flop_bet_sizes":  {"bet": "100%", "raise": "2.5x"},
      "turn_bet_sizes":  {"bet": "100%", "raise": "2.5x"},
      "river_bet_sizes": {"bet": "100%", "raise": "2.5x"},
      "max_iterations": 200,
      "target_exploitability_pct_pot": 0.5
    })";

    const auto out_path = solve(spec, "polar_vs_bc");
    auto scenario = pt::validation::load_scenario(out_path);
    auto eval     = pt::HandEvaluator::load_or_generate(PT_TABLES_PATH);
    auto r        = pt::validation::compute_best_response(scenario, eval);

    INFO("solver claimed exploitability: " << scenario.exploitability);
    INFO("BR-measured exploitability:    " << r.br_exploitability);
    INFO("OOP eq=" << r.oop_eq_aggregate << " br=" << r.oop_br_aggregate);
    INFO("IP  eq=" << r.ip_eq_aggregate  << " br=" << r.ip_br_aggregate);

    // For a properly converged solve the BR-measured exploitability and
    // the solver's claim should match within a small multiple of the
    // solver's own convergence error.
    using Catch::Matchers::WithinAbs;
    const double tol = std::max(1.0, 3.0 * scenario.exploitability);
    REQUIRE_THAT(r.br_exploitability,
                 WithinAbs(scenario.exploitability, tol));

    // Both BR gains should be small (within solver tolerance).
    const double pot = scenario.starting_pot;
    REQUIRE(r.oop_br_gain   < 0.10 * pot);
    REQUIRE(r.ip_br_gain    < 0.10 * pot);
}
