// Verify the BR validator correctly handles card-blocking effects.
//
// The trickiest place this can go wrong is at terminal showdown nodes:
// if BR-er holds AhKh and opponent's range is "AKs", the AhKh combo
// blocks 1 of the 4 AKs combos. Failing to drop it inflates BR's reach
// integral. This test pins a tiny range where the blocker math is
// hand-computable.

#include "best_response.h"
#include "scenario.h"

#include "card.h"
#include "hand_eval.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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

TEST_CASE("AKs vs AKs — heavy blockers", "[validation][blockers]") {
    // Both players have a tiny AKs range. The boards are picked so neither
    // hand has any showdown advantage by suit, and the blocker effect is
    // strong: AhKh blocks 1/4 of opp's AKs combos.
    //
    // Equilibrium: with identical ranges and no showdown asymmetry, both
    // players should split the pot in expectation. BR should agree.
    const std::string spec = R"({
      "oop_range": "AhKh,AsKs,AdKd,AcKc",
      "ip_range": "AhKh,AsKs,AdKd,AcKc",
      "flop": "Th9c4d",
      "turn": "2s",
      "river": "7h",
      "starting_pot": 100,
      "effective_stack": 200,
      "flop_bet_sizes":  {"bet": "50%", "raise": "2.5x"},
      "turn_bet_sizes":  {"bet": "50%", "raise": "2.5x"},
      "river_bet_sizes": {"bet": "50%", "raise": "2.5x"},
      "max_iterations": 100,
      "target_exploitability_pct_pot": 0.5
    })";

    const auto out_path = solve(spec, "aks_vs_aks");
    auto scenario = pt::validation::load_scenario(out_path);
    auto eval     = pt::HandEvaluator::load_or_generate(PT_TABLES_PATH);
    auto r        = pt::validation::compute_best_response(scenario, eval);

    INFO("solver claimed: " << scenario.exploitability);
    INFO("BR-measured:    " << r.br_exploitability);

    // With identical ranges and no advantage, the equilibrium values
    // should be roughly equal. Both BR gains should be tiny.
    const double pot = scenario.starting_pot;
    REQUIRE(r.oop_br_gain   < 0.05 * pot);
    REQUIRE(r.ip_br_gain    < 0.05 * pot);

    using Catch::Matchers::WithinAbs;
    const double tol = std::max(0.5, 2.0 * scenario.exploitability);
    REQUIRE_THAT(r.br_exploitability,
                 WithinAbs(scenario.exploitability, tol));
}
