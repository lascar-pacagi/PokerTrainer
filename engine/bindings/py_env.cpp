// Python bindings for PokerTrainer's HU NLHE engine.
//
// Exposes:
//   - ActionType, Player, Street, Terminal  (enums)
//   - HandEvaluator                          (evaluate5/7 + category labels)
//   - EncodedState                           (x, z, a as numpy arrays + legal list)
//   - HUState                                (read/write — for synthetic spot construction)
//   - deal / step / encode                   (HUGame free functions for combo enumeration)
//   - StepResult                             (done, just_acted, reward_bb)
//   - Env                                    (reset, step, observation, payoffs_bb)
//
// Tensor contract (must match docs/STATE_ENCODING.md, encoding v0.5):
//   obs.x : float32 ndarray shape (X_DIM,) = (816,)
//            — chip-state scalars + 11-bit legal-actions mask + 4-bit
//              per-street truncation flag + reverse-chronological action
//              history (4 sub-blocks: preflop/flop/turn/river; slot 0 of
//              each = most recent action of that street).
//   obs.a : float32 ndarray shape (n_legal, A_DIM) = (n_legal, 11)
//            — pure action one-hot rows (one 1.0 per row, no scalars).
//   obs.legal : list[ActionType] of length n_legal

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "action.h"
#include "card.h"
#include "encoder.h"
#include "env.h"
#include "game_hu.h"
#include "hand_eval.h"

namespace py = pybind11;

namespace {

// Convert std::array<float, N> to a new, owned float32 ndarray of shape (N,).
// IMPORTANT: construct with explicit shape vector. The single-int constructor
// `py::array_t<float>(N)` silently produces zero-filled output for some
// pybind11 versions / array sizes (we hit this on pybind11 2.9.1 with N=745
// and N=11). Always pass a shape list.
template <std::size_t N>
py::array_t<float> arr_to_numpy(const std::array<float, N>& a) {
    py::array_t<float> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(N)});
    auto buf = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < N; ++i) buf(i) = a[i];
    return out;
}

// Pack vector<array<float, A_DIM>> into a (n_legal, A_DIM) ndarray.
py::array_t<float> legal_actions_to_numpy(
        const std::vector<std::array<float, pt::A_DIM>>& rows) {
    py::array_t<float> out({static_cast<py::ssize_t>(rows.size()),
                            static_cast<py::ssize_t>(pt::A_DIM)});
    auto buf = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < pt::A_DIM; ++j)
            buf(i, j) = rows[i][j];
    return out;
}

}  // anonymous namespace

PYBIND11_MODULE(pokertrainer_engine, m) {
    m.doc() = "PokerTrainer HU NLHE engine — Phase 0";

    // ─── Module constants ───────────────────────────────────────────────────
    m.attr("X_DIM")             = pt::X_DIM;
    m.attr("A_DIM")             = pt::A_DIM;
    m.attr("HIST_MAX")          = pt::HIST_MAX;
    m.attr("HIST_FEAT")         = pt::HIST_FEAT;
    m.attr("STATIC_DIM")        = pt::STATIC_DIM;
    m.attr("LEGAL_MASK_DIM")    = pt::LEGAL_MASK_DIM;
    m.attr("X_OFF_LEGAL_MASK")    = pt::X_OFF_LEGAL_MASK;
    m.attr("X_OFF_HIST_TRUNCATED") = pt::X_OFF_HIST_TRUNCATED;
    m.attr("HIST_TRUNC_DIM")      = pt::HIST_TRUNC_DIM;
    m.attr("X_OFF_PREFLOP")     = pt::X_OFF_PREFLOP;
    m.attr("X_OFF_FLOP")        = pt::X_OFF_FLOP;
    m.attr("X_OFF_TURN")        = pt::X_OFF_TURN;
    m.attr("X_OFF_RIVER")       = pt::X_OFF_RIVER;
    m.attr("PREFLOP_SLOTS")     = pt::PREFLOP_SLOTS;
    m.attr("FLOP_SLOTS")        = pt::FLOP_SLOTS;
    m.attr("TURN_SLOTS")        = pt::TURN_SLOTS;
    m.attr("RIVER_SLOTS")       = pt::RIVER_SLOTS;
    // STREET_SLOTS / STREET_OFFSETS as Python tuples for callers that want
    // index-by-street access without selecting per-name attrs.
    m.attr("STREET_SLOTS") = py::make_tuple(
        pt::PREFLOP_SLOTS, pt::FLOP_SLOTS, pt::TURN_SLOTS, pt::RIVER_SLOTS);
    m.attr("STREET_OFFSETS") = py::make_tuple(
        pt::X_OFF_PREFLOP, pt::X_OFF_FLOP, pt::X_OFF_TURN, pt::X_OFF_RIVER);
    m.attr("NUM_ACTIONS")       = pt::NUM_ACTIONS;
    m.attr("NUM_PLAYERS")    = pt::NUM_PLAYERS;
    m.attr("BIG_BLIND_CHIPS")   = pt::HUState::BIG_BLIND_CHIPS;
    m.attr("SMALL_BLIND_CHIPS") = pt::HUState::SMALL_BLIND_CHIPS;

    // ─── Enums ──────────────────────────────────────────────────────────────
    py::enum_<pt::ActionType>(m, "ActionType")
        .value("FOLD",       pt::ActionType::FOLD)
        .value("CHECK_CALL", pt::ActionType::CHECK_CALL)
        .value("RAISE_25",   pt::ActionType::RAISE_25)
        .value("RAISE_33",   pt::ActionType::RAISE_33)
        .value("RAISE_50",   pt::ActionType::RAISE_50)
        .value("RAISE_75",   pt::ActionType::RAISE_75)
        .value("RAISE_100",  pt::ActionType::RAISE_100)
        .value("RAISE_150",  pt::ActionType::RAISE_150)
        .value("RAISE_200",  pt::ActionType::RAISE_200)
        .value("RAISE_300",  pt::ActionType::RAISE_300)
        .value("ALL_IN",     pt::ActionType::ALL_IN);

    py::enum_<pt::Player>(m, "Player")
        .value("SB", pt::Player::SB)
        .value("BB", pt::Player::BB);

    py::enum_<pt::Street>(m, "Street")
        .value("PREFLOP",  pt::Street::PREFLOP)
        .value("FLOP",     pt::Street::FLOP)
        .value("TURN",     pt::Street::TURN)
        .value("RIVER",    pt::Street::RIVER)
        .value("SHOWDOWN", pt::Street::SHOWDOWN);

    py::enum_<pt::Terminal>(m, "Terminal")
        .value("NONE",     pt::Terminal::NONE)
        .value("FOLD",     pt::Terminal::FOLD)
        .value("SHOWDOWN", pt::Terminal::SHOWDOWN);

    // ─── Card helpers ───────────────────────────────────────────────────────
    m.def("make_card",     &pt::make_card);
    m.def("parse_card",    &pt::parse_card);
    m.def("card_to_string",&pt::card_to_string);
    m.def("rank_of",       [](pt::Card c) { return pt::rank_of(c); });
    m.def("suit_of",       [](pt::Card c) { return pt::suit_of(c); });

    // ─── HandEvaluator ──────────────────────────────────────────────────────
    py::class_<pt::HandEvaluator>(m, "HandEvaluator")
        .def_static("load_or_generate", &pt::HandEvaluator::load_or_generate)
        .def("evaluate7", [](const pt::HandEvaluator& self, py::list cards) {
            if (cards.size() != 7) throw std::runtime_error("evaluate7 needs 7 cards");
            pt::Card buf[7];
            for (int i = 0; i < 7; ++i) buf[i] = cards[i].cast<pt::Card>();
            return self.evaluate7(buf);
        })
        .def("evaluate5", [](const pt::HandEvaluator& self, py::list cards) {
            if (cards.size() != 5) throw std::runtime_error("evaluate5 needs 5 cards");
            pt::Card buf[5];
            for (int i = 0; i < 5; ++i) buf[i] = cards[i].cast<pt::Card>();
            return self.evaluate5(buf);
        })
        .def_static("rank_category", &pt::HandEvaluator::rank_category);

    // ─── EncodedState ───────────────────────────────────────────────────────
    py::class_<pt::EncodedState>(m, "EncodedState")
        .def_property_readonly("x",
            [](const pt::EncodedState& self) { return arr_to_numpy(self.x); })
        .def_property_readonly("a",
            [](const pt::EncodedState& self) {
                return legal_actions_to_numpy(self.a);
            })
        .def_property_readonly("legal",
            [](const pt::EncodedState& self) { return self.legal; })
        .def_property_readonly("legal_idx",
            [](const pt::EncodedState& self) {
                py::array_t<int8_t> out(std::vector<py::ssize_t>{
                    static_cast<py::ssize_t>(self.legal.size())});
                auto buf = out.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < buf.shape(0); ++i)
                    buf(i) = static_cast<int8_t>(self.legal[i]);
                return out;
            });

    // ─── AppliedAction (one entry of state.history) ─────────────────────────
    // Exposed read-only — the action history is append-only from the
    // engine's POV. Token-based encodings (trainer/cfr/tokenize.py) walk
    // this list chronologically to derive per-action chip-state info.
    py::class_<pt::AppliedAction>(m, "AppliedAction")
        .def_readonly("actor",              &pt::AppliedAction::actor)
        .def_readonly("street",             &pt::AppliedAction::street)
        .def_readonly("type",               &pt::AppliedAction::type)
        .def_readonly("bet_to_chips",       &pt::AppliedAction::bet_to_chips)
        .def_readonly("pot_after_chips",    &pt::AppliedAction::pot_after_chips)
        .def_readonly("stack_after_chips",  &pt::AppliedAction::stack_after_chips)
        .def_readonly("was_all_in",         &pt::AppliedAction::was_all_in);

    // ─── HUState (read/write — see Phase 2 v2 plan for combo enumeration) ───
    // Fields are read/write so callers can construct synthetic spots:
    // deal() a state, then overwrite `hole` with a candidate combo and
    // (optionally) replay scripted action history via step().
    py::class_<pt::HUState>(m, "HUState")
        .def_readwrite("street",              &pt::HUState::street)
        .def_readwrite("to_act",              &pt::HUState::to_act)
        .def_readwrite("pot_chips",           &pt::HUState::pot_chips)
        .def_readwrite("stacks",              &pt::HUState::stacks)
        .def_readwrite("starting_stacks",     &pt::HUState::starting_stacks)
        .def_readwrite("invested_this_street",&pt::HUState::invested_this_street)
        .def_readwrite("invested_prior_streets",&pt::HUState::invested_prior_streets)
        .def_readwrite("hole",                &pt::HUState::hole)
        .def_readwrite("board",               &pt::HUState::board)
        .def_readwrite("terminal",            &pt::HUState::terminal)
        .def_readwrite("payoff_chips",        &pt::HUState::payoff_chips)
        .def_readwrite("actions_this_street", &pt::HUState::actions_this_street)
        .def_readwrite("all_in",              &pt::HUState::all_in)
        // Full action history. Returned by-value (a copy of the vector) —
        // tokenization is not in a hot inner loop, so the copy cost is fine.
        // Each element is a pt::AppliedAction (bound above).
        .def_property_readonly("history",
            [](const pt::HUState& self) { return self.history; })
        .def_property_readonly("history_size",
            [](const pt::HUState& self) {
                return static_cast<int>(self.history.size());
            })
        .def("to_call_chips",    &pt::HUState::to_call_chips)
        .def("min_raise_to_chips", &pt::HUState::min_raise_to_chips)
        .def("legal_actions",    &pt::HUState::legal_actions)
        .def("is_terminal",      &pt::HUState::is_terminal);

    // ─── HUGame free functions + encode (Phase 2 v2) ────────────────────────
    // These expose the stateless engine primitives directly for use cases
    // (e.g., model-derived range enumeration) where the Env wrapper's
    // single-RNG / single-state constraint is in the way.
    m.def("deal", &pt::HUGame::deal,
          py::arg("rng_seed"),
          py::arg("starting_stack_chips") =
              pt::HUState::BIG_BLIND_CHIPS * pt::HUState::DEFAULT_STACK_BB,
          "Deal a fresh hand and return a new HUState. Mirrors HUGame::deal.");

    // step() mutates `state` in place; the HandEvaluator is only needed at
    // showdown (it ranks 7-card hands), so it's optional and may be None for
    // preflop/early-postflop replay. Throws on illegal action.
    m.def("step",
          [](pt::HUState& state, pt::ActionType action,
             const pt::HandEvaluator* eval) {
              pt::HUGame::step(state, action, eval);
          },
          py::arg("state"),
          py::arg("action"),
          py::arg("evaluator") = nullptr,
          "Apply `action` to `state` in place. Pass `evaluator` only when a "
          "showdown could be reached.");

    m.def("encode", &pt::encode, py::arg("state"),
          "Build the EncodedState (x, a, legal) for the to-act player. "
          "Throws if `state` is terminal.");

    // ─── StepResult ─────────────────────────────────────────────────────────
    py::class_<pt::Env::StepResult>(m, "StepResult")
        .def_readonly("done",       &pt::Env::StepResult::done)
        .def_readonly("just_acted", &pt::Env::StepResult::just_acted)
        .def_readonly("reward_bb",  &pt::Env::StepResult::reward_bb);

    // ─── Env ────────────────────────────────────────────────────────────────
    py::class_<pt::Env>(m, "Env")
        .def(py::init<uint64_t, int64_t>(),
             py::arg("seed") = 0xC0FFEE,
             py::arg("starting_stack_chips") =
                 pt::HUState::BIG_BLIND_CHIPS * pt::HUState::DEFAULT_STACK_BB)
        .def("reset",         py::overload_cast<>(&pt::Env::reset))
        .def("reset",         py::overload_cast<uint64_t>(&pt::Env::reset), py::arg("hand_seed"))
        .def("observation",   &pt::Env::observation)
        .def("step",          &pt::Env::step, py::arg("legal_action_idx"))
        .def("step_action",   &pt::Env::step_action, py::arg("action"))
        .def("step_raise_to", &pt::Env::step_raise_to, py::arg("bet_to_chips"),
             "Raise to an explicit chip target (mirrors the FFI path the UI "
             "uses for Slumbot's arbitrary bet sizes).")
        .def("state",         &pt::Env::state, py::return_value_policy::reference_internal)
        .def("is_terminal",   &pt::Env::is_terminal)
        .def("to_act",        &pt::Env::to_act)
        .def("payoffs_bb",    &pt::Env::payoffs_bb)
        .def("clone",         &pt::Env::clone,
             "Deep-copy this Env (shared HandEvaluator). Used by CFR-style "
             "tree traversal that needs to branch into multiple actions.");
}
