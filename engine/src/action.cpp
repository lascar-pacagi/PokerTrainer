#include "action.h"

namespace pt {

std::string_view action_name(ActionType a) {
    switch (a) {
        case ActionType::FOLD:       return "fold";
        case ActionType::CHECK_CALL: return "check_call";
        case ActionType::RAISE_25:   return "raise_25";
        case ActionType::RAISE_33:   return "raise_33";
        case ActionType::RAISE_50:   return "raise_50";
        case ActionType::RAISE_75:   return "raise_75";
        case ActionType::RAISE_100:  return "raise_100";
        case ActionType::RAISE_150:  return "raise_150";
        case ActionType::RAISE_200:  return "raise_200";
        case ActionType::RAISE_300:  return "raise_300";
        case ActionType::ALL_IN:     return "all_in";
        default:                     return "?";
    }
}

}  // namespace pt
