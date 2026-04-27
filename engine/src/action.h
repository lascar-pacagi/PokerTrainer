#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace pt {

// Action discretization — see docs/STATE_ENCODING.md.
enum class ActionType : uint8_t {
    FOLD        = 0,
    CHECK_CALL  = 1,
    RAISE_25    = 2,   // block / tiny c-bet sizing
    RAISE_33    = 3,
    RAISE_50    = 4,
    RAISE_75    = 5,
    RAISE_100   = 6,
    RAISE_150   = 7,
    RAISE_200   = 8,
    RAISE_300   = 9,
    ALL_IN      = 10,
    N_ACTIONS   = 11,
};

inline constexpr int NUM_ACTIONS = static_cast<int>(ActionType::N_ACTIONS);

// Raise fractions of the pot (post-call). Order matches ActionType indices
// starting at RAISE_25 (RAISE_FRACTIONS[0] == RAISE_25's fraction).
inline constexpr std::array<double, 8> RAISE_FRACTIONS = {
    0.25, 0.33, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00,
};

inline constexpr bool is_raise(ActionType a) {
    return a >= ActionType::RAISE_25 && a <= ActionType::ALL_IN;
}

inline constexpr bool is_fold(ActionType a) {
    return a == ActionType::FOLD;
}

// A chosen action resolved to an absolute bet-to amount (in chips).
struct Action {
    ActionType type;
    int64_t bet_to_chips;   // absolute chip target; 0 for fold/check
};

std::string_view action_name(ActionType a);

}  // namespace pt
