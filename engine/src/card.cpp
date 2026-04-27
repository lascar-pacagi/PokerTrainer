#include "card.h"

namespace pt {

Card parse_card(std::string_view s) {
    if (s.size() != 2) return NO_CARD;
    auto rank_pos = RANK_CHARS.find(s[0]);
    auto suit_pos = SUIT_CHARS.find(s[1]);
    if (rank_pos == std::string_view::npos || suit_pos == std::string_view::npos)
        return NO_CARD;
    return make_card(static_cast<int>(rank_pos), static_cast<int>(suit_pos));
}

std::string card_to_string(Card c) {
    if (c >= NUM_CARDS) return "??";
    std::string out(2, '?');
    out[0] = RANK_CHARS[rank_of(c)];
    out[1] = SUIT_CHARS[suit_of(c)];
    return out;
}

}  // namespace pt
