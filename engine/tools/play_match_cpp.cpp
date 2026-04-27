// Minimal C++ harness: load a TorchScript model, play N hands vs check-fold,
// print total payoff in BB and the first hand's action trace.
//
// This is the "does the whole stack actually run?" smoke for Phase B. A
// trained model should win essentially every hand vs check-fold (the +34k
// mbb/100 we measured in Python); if C++ reproduces that ballpark we know
// LibTorch + pte::Env + pte::ModelInference are wired correctly.
//
// Usage:
//   build/play_match_cpp --model runs/cpu_long_50k/model.pt \
//                        --tables engine/data/poker_tables.bin \
//                        --n-hands 200 --seed 12345
//
// Tables path defaults to the repo copy; model path is required.
#include "env.h"
#include "encoder.h"
#include "action.h"
#include "hand_eval.h"
#include "inference.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Mirror of trainer/evaluate/policies.py::CheckFoldPolicy. FOLD if legal
// (there's a bet to call), else CHECK_CALL.
int check_fold_choice(const std::vector<pt::ActionType>& legal) {
    auto fold_it = std::find(legal.begin(), legal.end(), pt::ActionType::FOLD);
    if (fold_it != legal.end()) {
        return static_cast<int>(std::distance(legal.begin(), fold_it));
    }
    auto cc_it = std::find(legal.begin(), legal.end(), pt::ActionType::CHECK_CALL);
    if (cc_it == legal.end()) {
        throw std::runtime_error("check-fold: neither FOLD nor CHECK_CALL legal");
    }
    return static_cast<int>(std::distance(legal.begin(), cc_it));
}

struct Args {
    std::string model_path;
    std::string tables_path = "engine/data/poker_tables.bin";
    int         n_hands     = 200;
    uint64_t    seed        = 0xC0FFEEu;
    bool        verbose_first_hand = true;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (k == "--model")   a.model_path  = next("--model");
        else if (k == "--tables")  a.tables_path = next("--tables");
        else if (k == "--n-hands") a.n_hands     = std::atoi(next("--n-hands"));
        else if (k == "--seed")    a.seed        = std::strtoull(next("--seed"), nullptr, 0);
        else if (k == "--quiet")   a.verbose_first_hand = false;
        else {
            std::fprintf(stderr, "unknown arg: %s\n", k.c_str());
            std::exit(2);
        }
    }
    if (a.model_path.empty()) {
        std::fprintf(stderr,
                     "usage: play_match_cpp --model <path.pt> [--tables <bin>] "
                     "[--n-hands N] [--seed S] [--quiet]\n");
        std::exit(2);
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    // Hand evaluator tables — must be loaded before any Env is created.
    pt::HandEvaluator::load_or_generate(args.tables_path);

    pt::ModelInference model(args.model_path, "cpu");

    // Model sits in SB for even hand index, BB for odd — seat alternation
    // baseline (no duplicate dealing here; that's trivariate work for the
    // Python harness where we already have it).
    pt::Env env(args.seed);

    double total_bb_model = 0.0;
    double total_bb_opp   = 0.0;

    for (int h = 0; h < args.n_hands; ++h) {
        env.reset();
        const pt::Player model_seat = (h % 2 == 0) ? pt::Player::SB : pt::Player::BB;
        const bool       verbose    = (h == 0 && args.verbose_first_hand);

        if (verbose) {
            std::printf("=== hand 0: model seat=%s ===\n",
                        model_seat == pt::Player::SB ? "SB" : "BB");
        }

        while (!env.is_terminal()) {
            const pt::Player to_act = env.to_act();
            pt::EncodedState obs    = env.observation();

            int idx;
            if (to_act == model_seat) {
                std::vector<float> q(obs.a.size());
                idx = model.score_and_argmax(obs, q.data());
                if (verbose) {
                    const auto nm = pt::action_name(obs.legal[idx]);
                    std::printf("  [model] legal=%zu pick=%.*s q=%.2f\n",
                                obs.legal.size(),
                                static_cast<int>(nm.size()), nm.data(),
                                q[idx]);
                }
            } else {
                idx = check_fold_choice(obs.legal);
                if (verbose) {
                    const auto nm = pt::action_name(obs.legal[idx]);
                    std::printf("  [checkfold] pick=%.*s\n",
                                static_cast<int>(nm.size()), nm.data());
                }
            }
            env.step(idx);
        }

        const auto payoff = env.payoffs_bb();
        const double bb_model = (model_seat == pt::Player::SB) ? payoff[0] : payoff[1];
        const double bb_opp   = -bb_model;
        total_bb_model += bb_model;
        total_bb_opp   += bb_opp;
    }

    const double mbb_per_100 = (total_bb_model / args.n_hands) * 100.0 * 1000.0;
    std::printf("[cpp] %d hands: model=%+.3f BB  opp=%+.3f BB  rate=%+.1f mbb/100\n",
                args.n_hands, total_bb_model, total_bb_opp, mbb_per_100);
    return 0;
}
