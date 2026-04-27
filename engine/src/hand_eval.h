#pragma once

#include "card.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace pt {

// 7-card poker hand evaluator. Returns a rank in [1, 7462] where LOWER is
// better (1 = royal flush, 7462 = 7-high nothing). See GTO/hand_eval.cpp for
// the underlying perfect-hash construction.
class HandEvaluator {
public:
    // Load tables from disk. If the file doesn't exist, generate and save.
    // Throws std::runtime_error on I/O failure during generation.
    static HandEvaluator load_or_generate(std::string_view tables_path);

    // Evaluate 7 cards (2 hole + 5 board). `cards` must have length 7.
    uint16_t evaluate7(const Card* cards) const;

    // Evaluate 5 cards directly.
    uint16_t evaluate5(const Card* cards) const;

    // Category label for a rank ("Straight Flush", "Full House", etc.).
    static std::string_view rank_category(uint16_t rank);

    HandEvaluator();
    HandEvaluator(HandEvaluator&&) noexcept;
    HandEvaluator& operator=(HandEvaluator&&) noexcept;
    HandEvaluator(const HandEvaluator&)            = delete;
    HandEvaluator& operator=(const HandEvaluator&) = delete;
    ~HandEvaluator();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pt
