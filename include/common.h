#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

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

}  // namespace vecdb