#include "game_hu.h"

#include "hand_eval.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace pt {

// ─── HUState accessors ──────────────────────────────────────────────────────

int64_t HUState::current_bet_chips() const {
    return std::max(invested_this_street[0], invested_this_street[1]);
}

int64_t HUState::to_call_chips() const {
    const int64_t me = invested_this_street[static_cast<int>(to_act)];
    return current_bet_chips() - me;
}

int64_t HUState::min_raise_to_chips() const {
    // Raise-to amount ≥ current_bet + max(last_raise_size, 1 BB).
    const int64_t min_incr = std::max<int64_t>(last_raise_size, BIG_BLIND_CHIPS);
    return current_bet_chips() + min_incr;
}

bool HUState::is_all_in_showdown() const {
    return all_in[0] && all_in[1];
}

int64_t HUState::raise_to_from_fraction(double pot_fraction) const {
    // Per docs/STATE_ENCODING.md: bet_to = to_call + fraction * (pot + to_call),
    // then snap up to min-raise and down to effective all-in.
    const int64_t tc       = to_call_chips();
    const int64_t cur_bet  = current_bet_chips();
    const int64_t me_inv   = invested_this_street[static_cast<int>(to_act)];
    const int64_t my_stack = stacks[static_cast<int>(to_act)];

    // raise_to = cur_bet + pot_fraction * (pot_chips + to_call). The
    // pot-fraction is applied to the pot *after* the hypothetical call.
    int64_t raise_to = cur_bet +
                       static_cast<int64_t>(pot_fraction *
                                            static_cast<double>(pot_chips + tc));

    // Max raise-to = all-in (our full chips pushed in).
    const int64_t all_in_to = me_inv + my_stack;
    raise_to = std::min(raise_to, all_in_to);

    // Min raise-to = min_raise_to_chips(), unless that exceeds all-in in which
    // case the raise is only legal as an all-in under NLHE "incomplete raise"
    // rules. We return the all-in amount in that case.
    const int64_t min_rt = min_raise_to_chips();
    if (raise_to < min_rt) raise_to = std::min(min_rt, all_in_to);

    return raise_to;
}

uint16_t HUState::legal_actions_mask() const {
    if (is_terminal()) return 0;

    uint16_t mask = 0;
    const int64_t tc        = to_call_chips();
    const int64_t my_stack  = stacks[static_cast<int>(to_act)];
    const int64_t me_inv    = invested_this_street[static_cast<int>(to_act)];
    const int64_t all_in_to = me_inv + my_stack;

    // CHECK_CALL always legal (covers both check and call).
    mask |= (1u << static_cast<int>(ActionType::CHECK_CALL));

    // FOLD: legal only when facing a bet.
    if (tc > 0) mask |= (1u << static_cast<int>(ActionType::FOLD));

    // Raises: legal only if we still have chips behind and can actually raise
    // above current_bet (min-raise = current_bet + min_increment, capped to all-in).
    if (my_stack > tc) {
        // We have chips to raise. Each of RAISE_25..RAISE_300 and ALL_IN is
        // a candidate. Duplicate slots — where the snapped raise_to collides
        // with an already-legal slot — are dropped so the agent sees each
        // distinct sizing exactly once. This prevents "degenerate-equivalence"
        // policy stasis (two slots producing byte-identical trajectories would
        // make MC gradient signal ambiguous between them).
        mask |= (1u << static_cast<int>(ActionType::ALL_IN));

        const int64_t cur_bet = current_bet_chips();
        // First slot admitted is ALL_IN at bet-to = all_in_to. Collect
        // already-admitted raise-to amounts so later fractions can dedup.
        std::array<int64_t, 9> admitted{};   // up to 8 fractions + 1 all-in
        int n_admitted = 0;
        admitted[n_admitted++] = all_in_to;

        for (int i = 0; i < static_cast<int>(RAISE_FRACTIONS.size()); ++i) {
            const int64_t rt = raise_to_from_fraction(RAISE_FRACTIONS[i]);
            if (rt <= cur_bet || rt > all_in_to) continue;
            bool dup = false;
            for (int j = 0; j < n_admitted; ++j) {
                if (admitted[j] == rt) { dup = true; break; }
            }
            if (dup) continue;
            admitted[n_admitted++] = rt;
            mask |= (1u << (static_cast<int>(ActionType::RAISE_25) + i));
        }
    }

    return mask;
}

bool HUState::action_is_legal(ActionType a) const {
    return (legal_actions_mask() >> static_cast<int>(a)) & 1u;
}

std::vector<ActionType> HUState::legal_actions() const {
    std::vector<ActionType> out;
    out.reserve(NUM_ACTIONS);
    const uint16_t m = legal_actions_mask();
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        if ((m >> i) & 1u) out.push_back(static_cast<ActionType>(i));
    }
    return out;
}

// ─── HUGame ─────────────────────────────────────────────────────────────────

namespace {

// Fisher–Yates-based deterministic deal using std::mt19937_64.
std::array<Card, 9> deal_cards(uint64_t seed) {
    std::array<Card, NUM_CARDS> deck;
    for (int i = 0; i < NUM_CARDS; ++i) deck[i] = static_cast<Card>(i);
    std::mt19937_64 rng(seed);
    for (int i = NUM_CARDS - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        const int j = dist(rng);
        std::swap(deck[i], deck[j]);
    }
    std::array<Card, 9> out;
    for (int i = 0; i < 9; ++i) out[i] = deck[i];
    return out;
}

// Close out the street if both players have acted and invested amounts match
// (or someone is all-in with no further action possible). Return true iff we
// advanced to a new street (or terminal).
bool maybe_close_street(HUState& s, const HandEvaluator* eval) {
    // Both players all-in → deal remaining board and go to showdown.
    if (s.is_all_in_showdown()) {
        while (s.street != Street::SHOWDOWN) {
            s.invested_prior_streets[0] += s.invested_this_street[0];
            s.invested_prior_streets[1] += s.invested_this_street[1];
            s.invested_this_street = {0, 0};
            s.street = static_cast<Street>(static_cast<int>(s.street) + 1);
            s.actions_this_street = 0;
            s.last_raise_size = HUState::BIG_BLIND_CHIPS;
        }
        s.terminal = Terminal::SHOWDOWN;
        if (eval) HUGame::resolve_showdown(s, *eval);
        return true;
    }

    // Has the betting round closed? A round closes when:
    //  - invested_this_street is equal for both players
    //  - AND at least one action has been taken this street, OR preflop
    //    BB has had a chance to act after SB-call
    //
    // Preflop: SB posts 50, BB posts 100. SB is first to act. If SB limps (call to 100),
    // BB has the OPTION to check or raise. So invested amounts match after SB-call, but
    // BB has not acted → round not yet closed.
    //
    // We track whether both players have acted on this street. A call that
    // matches the opening bet counts as an action. After any raise, the
    // opponent must re-act.

    if (s.invested_this_street[0] != s.invested_this_street[1]) return false;

    // Special case preflop: the BB has not acted yet if actions_this_street
    // is exactly 1 and the SB just called (limped).
    if (s.street == Street::PREFLOP) {
        // SB acted once by calling; BB still has option. Don't close.
        if (s.actions_this_street < 2) return false;
    } else {
        // Postflop: BB is first to act (OOP). Round closes when both players
        // have acted and sizes match. For a check-check, actions_this_street
        // must be 2.
        if (s.actions_this_street < 2) return false;
    }

    // Round closed. Advance street or end hand.
    s.invested_prior_streets[0] += s.invested_this_street[0];
    s.invested_prior_streets[1] += s.invested_this_street[1];
    s.invested_this_street = {0, 0};
    s.actions_this_street = 0;
    s.last_raise_size = HUState::BIG_BLIND_CHIPS;

    if (s.street == Street::RIVER) {
        s.street = Street::SHOWDOWN;
        s.terminal = Terminal::SHOWDOWN;
        if (eval) HUGame::resolve_showdown(s, *eval);
        return true;
    }

    s.street = static_cast<Street>(static_cast<int>(s.street) + 1);
    // Postflop: BB acts first.
    s.to_act = Player::BB;
    return true;
}

}  // anonymous namespace

HUState HUGame::deal(uint64_t rng_seed, int64_t starting_stack_chips) {
    HUState s{};
    s.starting_stacks = {starting_stack_chips, starting_stack_chips};
    s.stacks = s.starting_stacks;
    s.invested_this_street = {0, 0};
    s.invested_prior_streets = {0, 0};
    s.pot_chips = 0;
    s.street = Street::PREFLOP;
    s.to_act = Player::SB;
    s.last_raise_size = HUState::BIG_BLIND_CHIPS;
    s.actions_this_street = 0;
    s.terminal = Terminal::NONE;
    s.payoff_chips = {0, 0};
    s.all_in = {false, false};

    // Deal 9 cards: [SB_c1, SB_c2, BB_c1, BB_c2, flop1, flop2, flop3, turn, river]
    const auto cards = deal_cards(rng_seed);
    s.hole[0] = {cards[0], cards[1]};
    s.hole[1] = {cards[2], cards[3]};
    s.board = {cards[4], cards[5], cards[6], cards[7], cards[8]};

    // Post blinds.
    const int sb_idx = static_cast<int>(Player::SB);
    const int bb_idx = static_cast<int>(Player::BB);
    s.stacks[sb_idx]               -= HUState::SMALL_BLIND_CHIPS;
    s.invested_this_street[sb_idx]  = HUState::SMALL_BLIND_CHIPS;
    s.stacks[bb_idx]               -= HUState::BIG_BLIND_CHIPS;
    s.invested_this_street[bb_idx]  = HUState::BIG_BLIND_CHIPS;
    s.pot_chips = HUState::SMALL_BLIND_CHIPS + HUState::BIG_BLIND_CHIPS;
    // Note: last_raise_size is 1 BB (the blind delta), so the first legal
    // raise-to is 2 BB.

    return s;
}

int HUGame::n_visible_board(Street s) {
    switch (s) {
        case Street::PREFLOP:  return 0;
        case Street::FLOP:     return 3;
        case Street::TURN:     return 4;
        case Street::RIVER:
        case Street::SHOWDOWN: return 5;
    }
    return 0;
}

void HUGame::step(HUState& s, ActionType action, const HandEvaluator* eval) {
    if (s.is_terminal()) throw std::runtime_error("step on terminal state");
    if (!s.action_is_legal(action))
        throw std::runtime_error("illegal action: " + std::string(action_name(action)));

    const int pi = static_cast<int>(s.to_act);
    AppliedAction rec{};
    rec.actor = s.to_act;
    rec.street = s.street;
    rec.type = action;
    rec.bet_to_chips = 0;

    switch (action) {
        case ActionType::FOLD: {
            s.terminal = Terminal::FOLD;
            // Winner = the non-folding player. Signed payoff = ± matched
            // contribution; any over-commitment above matched is returned to
            // the over-committer (zero-sum net transfer).
            const Player winner = other(s.to_act);
            const int wi = static_cast<int>(winner);
            const int li = static_cast<int>(s.to_act);
            const int64_t wc = s.invested_prior_streets[wi] + s.invested_this_street[wi];
            const int64_t lc = s.invested_prior_streets[li] + s.invested_this_street[li];
            const int64_t matched = std::min(wc, lc);
            s.payoff_chips[wi] =  matched;
            s.payoff_chips[li] = -matched;
            s.pot_chips = 0;
            rec.pot_after_chips = 0;
            rec.stack_after_chips = s.stacks[pi];   // unchanged by fold
            rec.was_all_in = false;                 // a fold cannot be all-in
            s.history.push_back(rec);
            return;
        }

        case ActionType::CHECK_CALL: {
            const int64_t tc = s.to_call_chips();
            const int64_t pay = std::min(tc, s.stacks[pi]);
            s.stacks[pi] -= pay;
            s.invested_this_street[pi] += pay;
            s.pot_chips += pay;
            if (s.stacks[pi] == 0) s.all_in[pi] = true;
            rec.bet_to_chips = s.invested_this_street[pi];
            break;
        }

        default: {  // RAISE_25..RAISE_300, ALL_IN
            // Resolve target bet-to.
            int64_t raise_to;
            if (action == ActionType::ALL_IN) {
                raise_to = s.invested_this_street[pi] + s.stacks[pi];
            } else {
                const int idx = static_cast<int>(action) - static_cast<int>(ActionType::RAISE_25);
                raise_to = s.raise_to_from_fraction(RAISE_FRACTIONS[idx]);
            }

            const int64_t add = raise_to - s.invested_this_street[pi];
            if (add <= 0 || add > s.stacks[pi])
                throw std::runtime_error("raise resolution failed");

            const int64_t prev_bet = s.current_bet_chips();
            const int64_t raise_incr = raise_to - prev_bet;
            if (raise_incr >= std::max<int64_t>(s.last_raise_size, HUState::BIG_BLIND_CHIPS)) {
                // Full-size raise updates the min-raise baseline.
                s.last_raise_size = raise_incr;
            }
            // Else: short all-in shove — min-raise rule unchanged per standard NLHE.

            s.stacks[pi] -= add;
            s.invested_this_street[pi] = raise_to;
            s.pot_chips += add;
            if (s.stacks[pi] == 0) s.all_in[pi] = true;
            rec.bet_to_chips = raise_to;
            break;
        }
    }

    rec.pot_after_chips = s.pot_chips;
    rec.stack_after_chips = s.stacks[pi];
    rec.was_all_in = s.all_in[pi];
    s.history.push_back(rec);
    s.actions_this_street += 1;

    // Swap to_act and check for round close.
    s.to_act = other(s.to_act);
    maybe_close_street(s, eval);
}

void HUGame::resolve_showdown(HUState& s, const HandEvaluator& eval) {
    const std::array<Card, 7> h0 = {s.hole[0][0], s.hole[0][1],
                                    s.board[0], s.board[1], s.board[2], s.board[3], s.board[4]};
    const std::array<Card, 7> h1 = {s.hole[1][0], s.hole[1][1],
                                    s.board[0], s.board[1], s.board[2], s.board[3], s.board[4]};
    const uint16_t r0 = eval.evaluate7(h0.data());
    const uint16_t r1 = eval.evaluate7(h1.data());

    const int64_t c0 = s.invested_prior_streets[0] + s.invested_this_street[0];
    const int64_t c1 = s.invested_prior_streets[1] + s.invested_this_street[1];
    const int64_t matched = std::min(c0, c1);  // uncalled excess returns to over-committer

    if (r0 < r1) {                // player 0 wins
        s.payoff_chips[0] =  matched;
        s.payoff_chips[1] = -matched;
    } else if (r1 < r0) {         // player 1 wins
        s.payoff_chips[1] =  matched;
        s.payoff_chips[0] = -matched;
    } else {                       // chop: net transfer is 0 (each recovers own matched share)
        s.payoff_chips[0] = 0;
        s.payoff_chips[1] = 0;
    }
    s.pot_chips = 0;
}

}  // namespace pt
