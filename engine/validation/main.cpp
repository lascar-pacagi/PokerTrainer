// pt-validate — independent best-response check on a pt-solver dump.
//
// Reads a `--mode tree --depth river` JSON output from pt-solver, computes
// the best-response value for each player against the embedded strategies,
// and reports the gap between BR-measured exploitability and the value
// pt-solver wrote into the same file. At true equilibrium the gap is 0;
// in practice it should match the solver's own claimed exploitability
// within solver convergence error and floating-point tolerance.
//
// CLI:
//   pt-validate --input scenario.json
//                [--tables /path/to/poker_tables.bin]
//                [--tolerance 0.01]   # fraction of pot for pass/fail
//
// Exit status: 0 on pass (BR_exploitability ≤ tolerance · pot), 1 on
// disagreement, 2 on argument or load error.

#include "best_response.h"
#include "hand_eval.h"
#include "scenario.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage() {
    std::cerr <<
        "usage: pt-validate --input scenario.json\n"
        "                  [--tables /path/to/poker_tables.bin]\n"
        "                  [--tolerance 0.01]\n"
        "                  [--samples N]   # 0 = exhaustive (default), >0 = MC\n"
        "                  [--seed S]      # ignored if --samples 0\n";
}

struct Args {
    std::string   input;
    std::string   tables;
    double        tolerance = 0.01;  // 1% of pot
    int           samples   = 0;     // 0 = exhaustive at chance
    std::uint64_t seed      = 0;     // 0 = random_device
};

bool parse_args(int argc, char** argv, Args& out) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto take = [&](const char* flag, std::string& dst) {
            if (a == flag && i + 1 < argc) {
                dst = argv[++i];
                return true;
            }
            return false;
        };
        if (take("--input",   out.input))   continue;
        if (take("--tables",  out.tables))  continue;
        if (a == "--tolerance" && i + 1 < argc) {
            out.tolerance = std::stod(argv[++i]);
            continue;
        }
        if (a == "--samples" && i + 1 < argc) {
            out.samples = std::stoi(argv[++i]);
            continue;
        }
        if (a == "--seed" && i + 1 < argc) {
            out.seed = std::stoull(argv[++i]);
            continue;
        }
        if (a == "-h" || a == "--help") {
            print_usage();
            return false;
        }
        std::cerr << "unknown arg: " << a << "\n";
        print_usage();
        return false;
    }
    if (out.input.empty()) {
        std::cerr << "--input is required\n";
        print_usage();
        return false;
    }
    if (out.tables.empty()) {
        // Default to engine/data/poker_tables.bin via PT_TABLES_PATH if set,
        // otherwise the standard relative location.
        const char* env = std::getenv("PT_TABLES_PATH");
        out.tables = env ? env :
            "/home/elucterio/Poker/PokerTrainer/engine/data/poker_tables.bin";
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 2;

    pt::validation::Scenario scenario;
    try {
        scenario = pt::validation::load_scenario(args.input);
    } catch (const std::exception& e) {
        std::cerr << "load failed: " << e.what() << "\n";
        return 2;
    }

    std::cout << "Loaded scenario: " << args.input << "\n"
              << "  pot = " << scenario.starting_pot
              <<  ", stack = " << scenario.effective_stack << "\n"
              << "  flop = ";
    for (auto c : scenario.flop) std::cout << pt::card_to_string(c);
    if (scenario.turn)  std::cout << " turn=" << pt::card_to_string(*scenario.turn);
    if (scenario.river) std::cout << " river=" << pt::card_to_string(*scenario.river);
    std::cout << "\n"
              << "  combos: oop=" << scenario.oop_combos.size()
              << ", ip=" << scenario.ip_combos.size() << "\n"
              << "  nodes: " << scenario.nodes.size() << "\n"
              << "  pt-solver claimed exploitability: "
              << scenario.exploitability << " chips\n";

    pt::HandEvaluator eval;
    try {
        eval = pt::HandEvaluator::load_or_generate(args.tables);
    } catch (const std::exception& e) {
        std::cerr << "hand-eval load failed: " << e.what() << "\n"
                  << "  (try --tables /path/to/poker_tables.bin)\n";
        return 2;
    }

    pt::validation::BRConfig config;
    config.samples = args.samples;
    config.seed    = args.seed;
    if (config.samples > 0) {
        std::cout << "  MC sampling: up to " << config.samples
                  << " random children per chance node";
        if (config.seed != 0) {
            std::cout << " (seed " << config.seed << ")";
        }
        std::cout << "\n";
    }

    pt::validation::BRResult r;
    try {
        r = pt::validation::compute_best_response(scenario, eval, config);
    } catch (const std::exception& e) {
        std::cerr << "BR computation failed: " << e.what() << "\n";
        return 2;
    }

    std::cout << "\n── Best-response results ──────────────────────────────\n";
    std::cout << "  OOP equilibrium value:  " << r.oop_eq_aggregate << "\n";
    std::cout << "  OOP best-response:      " << r.oop_br_aggregate << "\n";
    std::cout << "  OOP BR gain:            " << r.oop_br_gain << "\n";
    std::cout << "  IP  equilibrium value:  " << r.ip_eq_aggregate << "\n";
    std::cout << "  IP  best-response:      " << r.ip_br_aggregate << "\n";
    std::cout << "  IP  BR gain:            " << r.ip_br_gain << "\n";
    // Per-combo dump for diagnostics.
    if (r.oop_br_values.size() <= 20) {
        std::cout << "\n  per-combo OOP V (chip):\n";
        for (size_t i = 0; i < r.oop_br_values.size(); ++i) {
            const double eq_v = i < r.oop_eq_values.size() ? r.oop_eq_values[i] : 0.0;
            std::cout << "    " << scenario.oop_combos[i].original
                      << ": eq=" << eq_v
                      << " br=" << r.oop_br_values[i]
                      << " gain=" << (r.oop_br_values[i] - eq_v) << "\n";
        }
        std::cout << "\n  per-combo IP V (chip):\n";
        for (size_t i = 0; i < r.ip_br_values.size(); ++i) {
            const double eq_v = i < r.ip_eq_values.size() ? r.ip_eq_values[i] : 0.0;
            std::cout << "    " << scenario.ip_combos[i].original
                      << ": eq=" << eq_v
                      << " br=" << r.ip_br_values[i]
                      << " gain=" << (r.ip_br_values[i] - eq_v) << "\n";
        }
    }
    std::cout << "\n  BR-measured exploitability: " << r.br_exploitability
              << " chips\n";
    std::cout << "  pt-solver claimed:          " << scenario.exploitability
              << " chips\n";

    const double pot = scenario.starting_pot;
    const double pass_threshold = args.tolerance * pot;
    const double agreement_gap  =
        std::abs(r.br_exploitability - scenario.exploitability);

    std::cout << "\n  Tolerance: " << args.tolerance * 100 << "% of pot = "
              << pass_threshold << " chips\n";
    std::cout << "  |BR exploitability - claimed|: " << agreement_gap
              << " chips\n";

    bool pass = agreement_gap <= pass_threshold &&
                r.br_exploitability <= pass_threshold * 2.0;
    std::cout << (pass ? "\n  ✓ PASS"
                       : "\n  ✗ FAIL")
              << "  — independent BR "
              << (pass ? "agrees with" : "disagrees with")
              << " pt-solver\n";
    return pass ? 0 : 1;
}
