#include "scenario.h"

#include "third_party/json.hpp"

#include <fstream>
#include <stdexcept>

namespace pt::validation {

using nlohmann::json;

namespace {

NodeKind parse_kind(const std::string& s) {
    if (s == "action")         return NodeKind::Action;
    if (s == "chance")         return NodeKind::Chance;
    if (s == "chance_pending") return NodeKind::ChancePending;
    if (s == "terminal")       return NodeKind::Terminal;
    throw std::runtime_error("scenario.cpp: unknown node kind: " + s);
}

Player parse_player(const json& j) {
    if (j.is_null()) return Player::None;
    const auto s = j.get<std::string>();
    if (s == "oop") return Player::OOP;
    if (s == "ip")  return Player::IP;
    throw std::runtime_error("scenario.cpp: unknown player: " + s);
}

Combo parse_combo(const std::string& s) {
    // "AhKs" — two 2-char tokens.
    if (s.size() != 4) {
        throw std::runtime_error("scenario.cpp: bad combo length: " + s);
    }
    Card c0 = parse_card(std::string_view{s.data(),     2});
    Card c1 = parse_card(std::string_view{s.data() + 2, 2});
    if (c0 == NO_CARD || c1 == NO_CARD) {
        throw std::runtime_error("scenario.cpp: bad combo: " + s);
    }
    return Combo{{c0, c1}, s};
}

std::array<Card, 3> parse_flop(const json& arr) {
    if (!arr.is_array() || arr.size() != 3) {
        throw std::runtime_error("scenario.cpp: flop must be 3 cards");
    }
    return {
        parse_card(arr[0].get<std::string>()),
        parse_card(arr[1].get<std::string>()),
        parse_card(arr[2].get<std::string>()),
    };
}

std::vector<Card> parse_board(const json& arr) {
    std::vector<Card> out;
    out.reserve(arr.size());
    for (const auto& s : arr) {
        out.push_back(parse_card(s.get<std::string>()));
    }
    return out;
}

std::vector<std::vector<float>> parse_matrix(const json& j) {
    std::vector<std::vector<float>> out;
    if (j.is_null()) return out;
    out.reserve(j.size());
    for (const auto& row : j) {
        std::vector<float> r;
        r.reserve(row.size());
        for (const auto& v : row) r.push_back(v.get<float>());
        out.push_back(std::move(r));
    }
    return out;
}

std::vector<float> parse_vec(const json& j) {
    std::vector<float> out;
    if (j.is_null()) return out;
    out.reserve(j.size());
    for (const auto& v : j) out.push_back(v.get<float>());
    return out;
}

}  // namespace

Scenario load_scenario(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("scenario.cpp: cannot open " + path);
    }
    json j;
    f >> j;

    if (j.value("version", 0) != 1) {
        throw std::runtime_error(
            "scenario.cpp: unsupported schema version: "
            + std::to_string(j.value("version", -1)));
    }

    Scenario s;
    s.exploitability   = j.at("exploitability").get<double>();
    s.starting_pot     = j.at("starting_pot").get<int>();
    s.effective_stack  = j.at("effective_stack").get<int>();
    s.flop             = parse_flop(j.at("flop"));
    if (auto& t = j.at("turn"); !t.is_null()) {
        s.turn = parse_card(t.get<std::string>());
    }
    if (auto& r = j.at("river"); !r.is_null()) {
        s.river = parse_card(r.get<std::string>());
    }

    const auto& priv = j.at("private_cards");
    s.oop_combos.reserve(priv.at("oop").size());
    for (const auto& c : priv.at("oop")) s.oop_combos.push_back(parse_combo(c));
    s.ip_combos.reserve(priv.at("ip").size());
    for (const auto& c : priv.at("ip"))  s.ip_combos.push_back(parse_combo(c));

    const auto& nodes = j.at("nodes");
    s.nodes.reserve(nodes.size());
    bool found_root = false;
    for (const auto& n : nodes) {
        Node node;
        node.id     = n.at("id").get<std::string>();
        node.kind   = parse_kind(n.at("kind").get<std::string>());
        node.board  = parse_board(n.at("board"));
        node.pot    = n.at("pot").get<int>();
        node.stacks = {n.at("stacks")[0].get<int>(), n.at("stacks")[1].get<int>()};
        node.player = n.contains("player") ? parse_player(n.at("player")) : Player::None;
        if (n.contains("actions") && !n.at("actions").is_null()) {
            for (const auto& a : n.at("actions")) {
                node.actions.push_back(a.get<std::string>());
            }
        }
        if (n.contains("children")) {
            for (const auto& c : n.at("children")) {
                node.children.push_back(ChildEdge{
                    c.at("action_idx").get<int>(),
                    c.at("action").get<std::string>(),
                    c.at("node_id").get<std::string>(),
                });
            }
        }
        if (n.contains("line") && !n.at("line").is_null()) {
            for (const auto& a : n.at("line")) {
                node.line.push_back(a.get<std::string>());
            }
        }
        if (n.contains("strategy")) node.strategy   = parse_matrix(n.at("strategy"));
        if (n.contains("ev"))       node.ev         = parse_matrix(n.at("ev"));
        if (n.contains("weights"))  node.weights    = parse_vec(n.at("weights"));
        if (n.contains("weights_opp")) node.weights_opp = parse_vec(n.at("weights_opp"));

        if (node.id.empty()) found_root = true;
        s.nodes.emplace(node.id, std::move(node));
    }
    if (!found_root) {
        throw std::runtime_error("scenario.cpp: no root node (id \"\")");
    }
    s.root_id = "";
    return s;
}

}  // namespace pt::validation
