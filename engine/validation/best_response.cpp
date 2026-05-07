#include "best_response.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pt::validation {

namespace {

// ── BR DFS context ─────────────────────────────────────────────────────────

/// Two evaluation modes for the same DFS:
///   * BR  — at the target's action node, take max-EV per hand. Models the
///           target playing best response against the opponent's fixed
///           JSON strategy.
///   * Eq  — at the target's action node, take the JSON-strategy-weighted
///           sum. Models the target playing the equilibrium strategy.
/// Both modes share opponent-action / chance / terminal handling.
enum class TargetPolicy { BR, Eq };

struct Ctx {
    Player                    target;        // whose V we compute
    Player                    opponent;      // = opponent_of(target)
    TargetPolicy              policy;
    const Scenario*           scenario;
    const HandEvaluator*      eval;
    const std::vector<Combo>* target_combos;
    const std::vector<Combo>* opp_combos;
    size_t                    n_target;
    size_t                    n_opp;
};

inline bool combos_share_card(const Combo& a, const Combo& b) {
    return a.cards[0] == b.cards[0] || a.cards[0] == b.cards[1] ||
           a.cards[1] == b.cards[0] || a.cards[1] == b.cards[1];
}

inline bool combo_overlaps_board(const Combo& c, const std::vector<Card>& board) {
    for (Card b : board) {
        if (c.cards[0] == b || c.cards[1] == b) return true;
    }
    return false;
}

inline bool combo_contains(const Combo& c, Card card) {
    return c.cards[0] == card || c.cards[1] == card;
}

// ── Showdown payoff ─────────────────────────────────────────────────────────

/// Showdown sign from BR-er's perspective: +1 if target hand wins,
/// -1 if loses, 0 if ties. Exhaustively averages over remaining board
/// cards if the terminal node's board has fewer than 5 cards
/// (e.g. all-in on the flop).
double showdown_sign(
    const Combo&             tc,
    const Combo&             oc,
    const std::vector<Card>& board,
    const HandEvaluator&     eval) {
    if (board.size() == 5) {
        std::array<Card, 7> tb = {tc.cards[0], tc.cards[1],
                                   board[0], board[1], board[2], board[3], board[4]};
        std::array<Card, 7> ob = {oc.cards[0], oc.cards[1],
                                   board[0], board[1], board[2], board[3], board[4]};
        uint16_t tr = eval.evaluate7(tb.data());
        uint16_t orank = eval.evaluate7(ob.data());
        // Lower rank = better hand in this evaluator.
        if (tr < orank) return  1.0;
        if (tr > orank) return -1.0;
        return 0.0;
    }
    // Need to enumerate remaining runouts. Build the dead-card set:
    uint64_t dead = 0;
    for (Card b : board) dead |= bit(b);
    dead |= bit(tc.cards[0]) | bit(tc.cards[1]);
    dead |= bit(oc.cards[0]) | bit(oc.cards[1]);
    if (board.size() == 4) {
        // Enumerate river only.
        std::array<Card, 7> tb = {tc.cards[0], tc.cards[1],
                                   board[0], board[1], board[2], board[3], 0};
        std::array<Card, 7> ob = {oc.cards[0], oc.cards[1],
                                   board[0], board[1], board[2], board[3], 0};
        double sum = 0;
        int n = 0;
        for (Card r = 0; r < NUM_CARDS; ++r) {
            if (dead & bit(r)) continue;
            tb[6] = r;
            ob[6] = r;
            uint16_t tr = eval.evaluate7(tb.data());
            uint16_t orank = eval.evaluate7(ob.data());
            sum += (tr < orank ? 1.0 : (tr > orank ? -1.0 : 0.0));
            ++n;
        }
        return n > 0 ? sum / n : 0.0;
    }
    if (board.size() == 3) {
        std::array<Card, 7> tb = {tc.cards[0], tc.cards[1],
                                   board[0], board[1], board[2], 0, 0};
        std::array<Card, 7> ob = {oc.cards[0], oc.cards[1],
                                   board[0], board[1], board[2], 0, 0};
        double sum = 0;
        int n = 0;
        for (Card t = 0; t < NUM_CARDS; ++t) {
            if (dead & bit(t)) continue;
            tb[5] = t;
            ob[5] = t;
            uint64_t dead2 = dead | bit(t);
            for (Card r = t + 1; r < NUM_CARDS; ++r) {
                if (dead2 & bit(r)) continue;
                tb[6] = r;
                ob[6] = r;
                uint16_t tr = eval.evaluate7(tb.data());
                uint16_t orank = eval.evaluate7(ob.data());
                sum += (tr < orank ? 1.0 : (tr > orank ? -1.0 : 0.0));
                ++n;
            }
        }
        return n > 0 ? sum / n : 0.0;
    }
    throw std::runtime_error("showdown_sign: bad board size " +
                             std::to_string(board.size()));
}

// ── Forward declaration for recursion ──────────────────────────────────────

std::vector<double> compute_v_at(
    const Node&                node,
    const std::vector<double>& opp_reach,
    Player                     parent_actor,
    const Ctx&                 ctx);

// ── Per-node handlers ──────────────────────────────────────────────────────

std::vector<double> handle_terminal(
    const Node&                node,
    const std::vector<double>& opp_reach,
    Player                     parent_actor,
    const Ctx&                 ctx) {
    std::vector<double> V(ctx.n_target, 0.0);
    const bool is_fold =
        !node.line.empty() && node.line.back() == "Fold";
    // Per-terminal chip accounting:
    //   bet_oop = effective_stack - stacks[0] (chips OOP put in this game)
    //   bet_ip  = effective_stack - stacks[1]
    // At a fold the unmatched bet is refunded → only the loser's matched
    // contribution is at risk. At a showdown both players contributed the
    // same. In both cases the loser's net contribution is
    //   loser_contribution = starting_pot/2 + min(bet_oop, bet_ip).
    // Winner +loser_contribution, loser −loser_contribution.
    const int bet_oop = ctx.scenario->effective_stack - node.stacks[0];
    const int bet_ip  = ctx.scenario->effective_stack - node.stacks[1];
    const int min_bet = bet_oop < bet_ip ? bet_oop : bet_ip;
    const double half_pot =
        ctx.scenario->starting_pot / 2.0 + static_cast<double>(min_bet);

    if (is_fold) {
        // The folder is whoever took the Fold action — that's `parent_actor`.
        const bool target_lost = (parent_actor == ctx.target);
        const double payoff = target_lost ? -half_pot : +half_pot;
        for (size_t h = 0; h < ctx.n_target; ++h) {
            const Combo& tc = (*ctx.target_combos)[h];
            if (combo_overlaps_board(tc, node.board)) continue;
            double total_reach = 0;
            for (size_t hq = 0; hq < ctx.n_opp; ++hq) {
                if (opp_reach[hq] <= 0) continue;
                const Combo& oc = (*ctx.opp_combos)[hq];
                if (combos_share_card(tc, oc)) continue;
                if (combo_overlaps_board(oc, node.board)) continue;
                total_reach += opp_reach[hq];
            }
            V[h] = payoff * total_reach;
        }
        return V;
    }
    // Showdown.
    for (size_t h = 0; h < ctx.n_target; ++h) {
        const Combo& tc = (*ctx.target_combos)[h];
        if (combo_overlaps_board(tc, node.board)) continue;
        double v = 0;
        for (size_t hq = 0; hq < ctx.n_opp; ++hq) {
            if (opp_reach[hq] <= 0) continue;
            const Combo& oc = (*ctx.opp_combos)[hq];
            if (combos_share_card(tc, oc)) continue;
            if (combo_overlaps_board(oc, node.board)) continue;
            const double sign = showdown_sign(tc, oc, node.board, *ctx.eval);
            v += opp_reach[hq] * sign * half_pot;
        }
        V[h] = v;
    }
    return V;
}

std::vector<double> handle_chance(
    const Node&                node,
    const std::vector<double>& opp_reach,
    const Ctx&                 ctx) {
    std::vector<double> V(ctx.n_target, 0.0);
    std::vector<int>    n_valid(ctx.n_target, 0);
    for (const auto& edge : node.children) {
        Card dealt = static_cast<Card>(edge.action_idx);
        // Filter opp combos that contain the dealt card.
        std::vector<double> child_reach(opp_reach);
        for (size_t hq = 0; hq < ctx.n_opp; ++hq) {
            if (combo_contains((*ctx.opp_combos)[hq], dealt)) {
                child_reach[hq] = 0;
            }
        }
        const Node& child = ctx.scenario->node_at(edge.node_id);
        auto child_v = compute_v_at(child, child_reach, Player::None, ctx);
        for (size_t h = 0; h < ctx.n_target; ++h) {
            // Skip target combos containing the dealt card — those reach 0
            // through this branch.
            if (combo_contains((*ctx.target_combos)[h], dealt)) continue;
            V[h] += child_v[h];
            ++n_valid[h];
        }
    }
    for (size_t h = 0; h < ctx.n_target; ++h) {
        if (n_valid[h] > 0) V[h] /= n_valid[h];
    }
    return V;
}

std::vector<double> handle_target_action(
    const Node&                node,
    const std::vector<double>& opp_reach,
    const Ctx&                 ctx) {
    std::vector<std::vector<double>> child_vs;
    child_vs.reserve(node.children.size());
    for (const auto& edge : node.children) {
        const Node& child = ctx.scenario->node_at(edge.node_id);
        // Target's choice doesn't change opp_reach.
        child_vs.push_back(
            compute_v_at(child, opp_reach, ctx.target, ctx));
    }
    std::vector<double> V(ctx.n_target, 0.0);
    if (ctx.policy == TargetPolicy::BR) {
        // Per-hand max-EV: each hand can choose its own best action.
        for (size_t h = 0; h < ctx.n_target; ++h) {
            double best = -std::numeric_limits<double>::infinity();
            for (const auto& cv : child_vs) {
                if (cv[h] > best) best = cv[h];
            }
            V[h] = best;
        }
    } else {
        // Equilibrium policy: weight by node.strategy[h][a].
        for (size_t h = 0; h < ctx.n_target; ++h) {
            double v = 0;
            for (size_t a = 0; a < child_vs.size(); ++a) {
                v += node.strategy[h][a] * child_vs[a][h];
            }
            V[h] = v;
        }
    }
    return V;
}

std::vector<double> handle_opponent_action(
    const Node&                node,
    const std::vector<double>& opp_reach,
    const Ctx&                 ctx) {
    // For each action a, descend with opp_reach scaled by opp's per-combo
    // strategy probability for action a. The downstream V (computed against
    // the scaled reach) already accounts for "opp took this action with
    // probability strategy[h_o][a]"; we just sum across actions, no
    // outer weighting.
    std::vector<double> V(ctx.n_target, 0.0);
    for (size_t a = 0; a < node.actions.size(); ++a) {
        std::vector<double> new_reach(opp_reach.size());
        for (size_t hq = 0; hq < opp_reach.size(); ++hq) {
            new_reach[hq] = opp_reach[hq] * node.strategy[hq][a];
        }
        const Node& child = ctx.scenario->node_at(node.children[a].node_id);
        auto child_v =
            compute_v_at(child, new_reach, ctx.opponent, ctx);
        for (size_t h = 0; h < ctx.n_target; ++h) {
            V[h] += child_v[h];
        }
    }
    return V;
}

// ── DFS dispatcher ─────────────────────────────────────────────────────────

std::vector<double> compute_v_at(
    const Node&                node,
    const std::vector<double>& opp_reach,
    Player                     parent_actor,
    const Ctx&                 ctx) {
    switch (node.kind) {
        case NodeKind::Terminal:
            return handle_terminal(node, opp_reach, parent_actor, ctx);
        case NodeKind::Chance:
            return handle_chance(node, opp_reach, ctx);
        case NodeKind::ChancePending:
            throw std::runtime_error(
                "compute_v_at: chance_pending node " + node.id +
                " — re-solve with `--depth river` for valid BR.");
        case NodeKind::Action:
            if (node.player == ctx.target) {
                return handle_target_action(node, opp_reach, ctx);
            }
            return handle_opponent_action(node, opp_reach, ctx);
    }
    throw std::runtime_error("compute_v_at: unreachable");
}

// ── Per-player BR/Eq driver ────────────────────────────────────────────────

struct Aggregated {
    std::vector<double> per_hand;
    double              aggregate;
};

Aggregated compute_for(
    Player               target,
    TargetPolicy         policy,
    const Scenario&      scenario,
    const HandEvaluator& eval) {
    Ctx ctx{
        .target        = target,
        .opponent      = opponent_of(target),
        .policy        = policy,
        .scenario      = &scenario,
        .eval          = &eval,
        .target_combos = &scenario.combos_for(target),
        .opp_combos    = &scenario.combos_for(opponent_of(target)),
        .n_target      = scenario.combos_for(target).size(),
        .n_opp         = scenario.combos_for(opponent_of(target)).size(),
    };

    // Initial reach for opp at root: take the appropriate weights vector
    // from the root node. The root's `weights` is for player-to-act, so
    // depending on who that is we pick weights or weights_opp.
    const Node& root = scenario.root();
    std::vector<double> opp_reach(ctx.n_opp);
    if (root.player == ctx.opponent) {
        for (size_t i = 0; i < ctx.n_opp; ++i) opp_reach[i] = root.weights[i];
    } else {
        for (size_t i = 0; i < ctx.n_opp; ++i) opp_reach[i] = root.weights_opp[i];
    }

    auto v_cf = compute_v_at(root, opp_reach, Player::None, ctx);

    // Convert from cf-value to chip-value units. cf_value[h] sums over opp
    // combos weighted by their reach; chip-value divides by that sum
    // (per-h to account for blocker effects: combos sharing a card with h
    // are excluded from both numerator and denominator).
    //
    // Without this normalization, V_OOP scales with sum(IP reach) and
    // V_IP scales with sum(OOP reach) — different factors when ranges
    // are asymmetric, which corrupts the (V_BR − V_eq) chip totals.
    std::vector<double> v_chip(ctx.n_target, 0.0);
    for (size_t h = 0; h < ctx.n_target; ++h) {
        const Combo& tc = (*ctx.target_combos)[h];
        double denom = 0;
        for (size_t hq = 0; hq < ctx.n_opp; ++hq) {
            if (opp_reach[hq] <= 0) continue;
            const Combo& oc = (*ctx.opp_combos)[hq];
            if (combos_share_card(tc, oc)) continue;
            denom += opp_reach[hq];
        }
        v_chip[h] = denom > 0 ? v_cf[h] / denom : 0.0;
    }

    // Range-weighted aggregate over target's root reach.
    std::vector<double> target_reach(ctx.n_target);
    if (root.player == ctx.target) {
        for (size_t i = 0; i < ctx.n_target; ++i) target_reach[i] = root.weights[i];
    } else {
        for (size_t i = 0; i < ctx.n_target; ++i)
            target_reach[i] = root.weights_opp[i];
    }
    double total = 0;
    double weighted = 0;
    for (size_t i = 0; i < ctx.n_target; ++i) {
        total    += target_reach[i];
        weighted += target_reach[i] * v_chip[i];
    }
    return Aggregated{
        std::move(v_chip),
        total > 0 ? weighted / total : 0.0,
    };
}

}  // namespace

BRResult compute_best_response(
    const Scenario&      s,
    const HandEvaluator& eval) {
    const auto oop_br = compute_for(Player::OOP, TargetPolicy::BR, s, eval);
    const auto ip_br  = compute_for(Player::IP,  TargetPolicy::BR, s, eval);
    const auto oop_eq = compute_for(Player::OOP, TargetPolicy::Eq, s, eval);
    const auto ip_eq  = compute_for(Player::IP,  TargetPolicy::Eq, s, eval);

    BRResult r;
    r.oop_br_values     = oop_br.per_hand;
    r.ip_br_values      = ip_br.per_hand;
    r.oop_br_aggregate  = oop_br.aggregate;
    r.ip_br_aggregate   = ip_br.aggregate;
    r.oop_eq_aggregate  = oop_eq.aggregate;
    r.ip_eq_aggregate   = ip_eq.aggregate;
    r.oop_br_gain       = r.oop_br_aggregate - r.oop_eq_aggregate;
    r.ip_br_gain        = r.ip_br_aggregate  - r.ip_eq_aggregate;
    r.br_exploitability = r.oop_br_gain + r.ip_br_gain;
    return r;
}

}  // namespace pt::validation
