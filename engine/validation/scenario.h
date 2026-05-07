#pragma once
//
// In-memory representation of a pt-solver tree-mode JSON dump.
//
// Mirrors the schema produced by `pt-solver --mode tree` (validation/src/main.rs).
// The validator loads a scenario, then walks `nodes` via parent/child id refs
// to compute a best response.
//
// All cards are stored using the engine's `pt::Card` uint8 (rank*4+suit).
// Combo strings are kept as the original 4-char form ("AhKh") for debugging
// + readable output; an additional [card, card] pair is parsed alongside.

#include "card.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace pt::validation {

struct ChildEdge {
    int         action_idx;
    std::string action;     // human-readable: "Check", "Bet(30)", "Th"
    std::string node_id;    // child's id in `Scenario::nodes`
};

enum class NodeKind {
    Action,
    Chance,           // walked chance node — has children for each card
    ChancePending,    // unwalked chance — children empty (file dump truncated)
    Terminal,
};

enum class Player : uint8_t {
    OOP = 0,
    IP  = 1,
    None,            // chance / terminal
};

inline Player opponent_of(Player p) {
    return p == Player::OOP ? Player::IP : Player::OOP;
}

struct Node {
    std::string                    id;
    NodeKind                       kind;
    std::vector<Card>              board;       // 3, 4, or 5 cards
    int                            pot;
    std::array<int, 2>             stacks;      // [oop, ip]
    Player                         player;      // None at chance/terminal
    std::vector<std::string>       actions;
    std::vector<ChildEdge>         children;
    /// Human-readable action sequence from root (e.g.
    /// `["Check","Bet(30)","Fold"]`). Only `back()` is needed by the BR
    /// validator (to distinguish fold-terminal from showdown-terminal).
    std::vector<std::string>       line;
    // Per-hand × per-action arrays. Only populated at action nodes.
    // strategy[h] sums to ~1 (within float tolerance). ev[h][a] in chips.
    std::vector<std::vector<float>> strategy;
    std::vector<std::vector<float>> ev;
    // Reach probabilities per hand for the player-to-act and the opponent.
    std::vector<float>             weights;     // for `player`
    std::vector<float>             weights_opp;
};

/// One combo for a player. `cards` contains the two hole cards as engine
/// Card bytes; `original` is the 4-char canonical string from the JSON.
struct Combo {
    std::array<Card, 2> cards;
    std::string         original;
};

struct Scenario {
    double                                exploitability;
    int                                   starting_pot;
    int                                   effective_stack;
    std::array<Card, 3>                   flop;
    std::optional<Card>                   turn;
    std::optional<Card>                   river;
    // Per-player combo lists, in the same order as the strategy/ev
    // arrays inside each Node.
    std::vector<Combo>                    oop_combos;
    std::vector<Combo>                    ip_combos;
    // id → node, ready for traversal. Empty id "" is the root.
    std::unordered_map<std::string, Node> nodes;
    std::string                           root_id;

    const std::vector<Combo>& combos_for(Player p) const {
        return p == Player::OOP ? oop_combos : ip_combos;
    }
    const Node& node_at(const std::string& id) const {
        return nodes.at(id);
    }
    const Node& root() const { return nodes.at(root_id); }
};

// Loads a scenario from a JSON file produced by `pt-solver --mode tree`.
// Throws std::runtime_error on parse error or unsupported schema version.
Scenario load_scenario(const std::string& path);

}  // namespace pt::validation
