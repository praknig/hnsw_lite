/**
 * @file common.h
 * @brief Shared types and constants used by every Layer 1 file.
 *
 * Contains no classes. Defines:
 *  - NodeId: the internal number of a vector (0, 1, 2, ...).
 *  - kEmpty: the value that marks an unused neighbor slot.
 *  - kAlign / kFloatsPerLine: the 64-byte cache line and how many floats fit in it.
 *  - round_up(): rounds a size up to a multiple (used for padding).
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace vecdb {

// Our internal vector number: 0, 1, 2, ... (32 bits = up to ~4 billion vectors).
using NodeId = std::uint32_t;

// Marks an unused neighbor slot ("end of list").
inline constexpr NodeId kEmpty = std::numeric_limits<NodeId>::max();

// One cache line (the "tray" the CPU fetches) and how many floats fit in it.
inline constexpr std::size_t kAlign = 64;
inline constexpr std::size_t kFloatsPerLine = kAlign / sizeof(float);  // 16

// Rounds n up to the next multiple of m. Example: round_up(100, 16) == 112.
constexpr std::size_t round_up(std::size_t n, std::size_t m) {
    return (n + m - 1) / m * m;
}

// A count from a floating-point value, clamped to [0, limit] before converting:
// converting a double beyond size_t's range is undefined behavior (fuzzing found a
// beam width of 2e41 from a huge but finite planner multiplier). NaN gives 0.
inline std::size_t clamped_count(double x, std::size_t limit) noexcept {
    if (!(x > 0.0)) return 0;
    if (x >= static_cast<double>(limit)) return limit;
    return static_cast<std::size_t>(x);
}

// The largest vector dimension (16.7 million), for constructors and loaders alike:
// beyond it, sizes computed from the dimension could overflow (round_up of a
// dimension near 2**64 wraps around to a tiny size).
inline constexpr std::size_t kMaxDim = std::size_t(1) << 24;

// The dimension, checked before anything is computed or allocated from it.
inline std::size_t checked_dim(std::size_t dim) {
    if (dim == 0) throw std::invalid_argument("dim must be > 0");
    if (dim > kMaxDim) throw std::invalid_argument("dim must be at most 16,777,216");
    return dim;
}

// Makes sure `v` can take one more push_back without allocating, growing its
// capacity geometrically. Calling this first makes the push_back itself unable
// to throw, which lets callers change several containers all-or-nothing.
template <class Vector>
void reserve_one_more(Vector& v) {
    if (v.size() == v.capacity()) v.reserve(std::max<std::size_t>(16, v.capacity() * 2));
}

}  // namespace vecdb