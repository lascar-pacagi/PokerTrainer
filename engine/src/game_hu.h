#pragma once

#include "action.h"
#include "card.h"

#include <array>
#include <cstdint>
#include <vector>

namespace pt {

// Player indices in heads-up.
//   0 = BTN/SB — acts FIRST preflop, acts LAST postflop (IP on flop/turn/river).
//   1 = BB     — acts LAST preflop,  acts FIRST postflop (OOP on flop/turn/river).
enum class Player : uint8_t { SB = 0, BB = 1 };
inline constexpr int NUM_PLAYERS = 2;

inline constexpr Player other(Player p) {
    return p == Player::SB ? Player::BB : Player::SB;
}

enum class Street : uint8_t { PREFLOP = 0, FLOP = 1, TURN = 2, RIVER = 3, SHOWDOWN = 4 };

// Reason the hand ended.
enum class Terminal : uint8_t {
    NONE = 0,
    FOLD = 1,       // opponent folded
    SHOWDOWN = 2,   // ran out the board, now compare hands
};

// One action applied to the state (for history + replay).
struct AppliedAction {
    Player actor;
    Street street;
    ActionType type;
    int64_t bet_to_chips;     // absolute chip target at the moment of action (0 for fold/check)
    int64_t pot_after_chips;  // total pot after action resolved
};

// HU NLHE game state. Chips are tracked in integer "chip units"; a big blind
// is `BIG_BLIND_CHIPS` chips so we can represent sub-bb sizes precisely.
struct HUState {
    static constexpr int64_t BIG_BLIND_CHIPS   = 100;   // 1 bb = 100 chips
    static constexpr int64_t SMALL_BLIND_CHIPS = 50;    // 0.5 bb = 50 chips
    static constexpr int64_t DEFAULT_STACK_BB  = 100;   // 100 bb starting stacks

    // ─── Invariants ─────────────────────────────────────────────────────────
    //  - stacks[i] >= 0; invested_this_street[i] + invested_prior_streets[i]
    //    + stacks[i] = starting_stacks[i].
    //  - pot_chips = sum over players of money committed to the pot
    //    (both prior streets and this street's current invested amounts).
    //  - to_call_for(p) = max_invested_this_street - invested_this_street[p].
    //  - legal_actions() excludes FOLD when to_call is 0; excludes CHECK_CALL
    //    never (check and call share the slot).

    // Static configuration
    std::array<int64_t, NUM_PLAYERS> starting_stacks;

    // Dealt cards
    std::array<std::array<Card, 2>, NUM_PLAYERS> hole;  // [player][0..1]
    std::array<Card, 5> board;                          // unused slots = NO_CARD

    // Chip-tracking
    std::array<int64_t, NUM_PLAYERS> stacks;                   // remaining behind
    std::array<int64_t, NUM_PLAYERS> invested_this_street;     // chips put in this street
    std::array<int64_t, NUM_PLAYERS> invested_prior_streets;   // committed on earlier streets
    int64_t pot_chips;

    // Flow control
    Street    street;
    Player    to_act;
    int64_t   last_raise_size;   // most recent raise-increment (for min-raise rule)
    int       actions_this_street;

    // Termination
    Terminal  terminal;
    std::array<int64_t, NUM_PLAYERS> payoff_chips;  // signed chip delta vs. starting stack

    // Whether each player is all-in (cannot act further).
    std::array<bool, NUM_PLAYERS> all_in;

    // History of applied actions.
    std::vector<AppliedAction> history;

    // ─── Accessors ──────────────────────────────────────────────────────────
    int64_t to_call_chips() const;
    int64_t min_raise_to_chips() const;
    int64_t current_bet_chips() const;   // max of invested_this_street across players
    bool is_terminal() const { return terminal != Terminal::NONE; }
    bool is_all_in_showdown() const;     // both players all-in with no further action

    // Compute discrete raise target from pot fraction. Snaps up to min-raise
    // and down to all-in as needed. Returns -1 if the raise is not legal.
    int64_t raise_to_from_fraction(double pot_fraction) const;

    // Bitset of legal actions (indexed by ActionType). Never returns empty
    // while the state is non-terminal.
    uint16_t legal_actions_mask() const;
    bool action_is_legal(ActionType a) const;
    std::vector<ActionType> legal_actions() const;
};

// Forward reference for the main engine class.
class HandEvaluator;

// Factory + stepping functions (pure wrt the HUState argument).
class HUGame {
public:
    // Initial state: deal hole cards, deal the 5 board cards (hidden until the
    // corresponding street), post blinds, set to_act = SB, street = PREFLOP.
    static HUState deal(uint64_t rng_seed, int64_t starting_stack_chips = HUState::BIG_BLIND_CHIPS * HUState::DEFAULT_STACK_BB);

    // Apply `action` to `state` (mutates in place). Throws std::runtime_error
    // if action is illegal. On terminal outcomes, fills payoff_chips.
    static void step(HUState& state, ActionType action, const HandEvaluator* eval = nullptr);

    // Utility: return the card on the board that is visible on `street`.
    // PREFLOP -> 0, FLOP -> 3, TURN -> 4, RIVER/SHOWDOWN -> 5.
    static int n_visible_board(Street s);

    // Showdown resolution: compare 7-card hands and distribute pot.
    static void resolve_showdown(HUState& state, const HandEvaluator& eval);
};

}  // namespace pt
