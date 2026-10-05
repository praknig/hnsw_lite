#pragma once
#include <cassert>
#include <type_traits>
#include <vector>

#include "aligned_block.h"

namespace vecdb {

/**
 * @brief Bump allocator that hands out small pieces of large memory blocks.
 *
 * How it works:
 *  - Holds a list of AlignedBlocks and a counter of bytes used in the last one.
 *  - allocate() rounds the counter up to the requested alignment, returns that
 *    address and moves the counter forward by the requested size.
 *  - When the last block is full, a new block is added. A request larger than
 *    one block gets its own block of exactly that size.
 *
 * Rules:
 *  - Pieces are never freed one by one; all memory is freed when the Arena is destroyed.
 *  - Returned addresses never change.
 *  - Only plain data (no destructors) may be stored; allocate_array() checks this.
 *  - Alignment must be a power of two no larger than 64.
 *  - Not thread-safe.
 *
 * Used by: GraphStorage, for upper-level neighbor lists.
 */
class Arena {
public:
    explicit Arena(std::size_t block_bytes = 1 << 20) : block_bytes_(block_bytes) {}

    // Returns `bytes` of memory starting at a multiple of `align`.
    void* allocate(std::size_t bytes, std::size_t align = alignof(std::max_align_t)) {
        assert(align <= kAlign && (align & (align - 1)) == 0);  // GCOVR_EXCL_BR_LINE: failing aborts
        std::size_t start = round_up(used_, align);
        if (blocks_.empty() || start + bytes > blocks_.back().size()) {
            // Current block is full: grab a new one (bigger if the request is huge).
            blocks_.emplace_back(bytes > block_bytes_ ? bytes : block_bytes_);
            start = 0;
        }
        used_ = start + bytes;
        return blocks_.back().data() + start;
    }

    // Typed helper. Only plain data is allowed, because the arena never
    // runs destructors (a std::vector stored here would leak).
    template <class T>
    T* allocate_array(std::size_t n) {
        static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>,
                      "Arena can only hold plain data");
        return static_cast<T*>(allocate(n * sizeof(T), alignof(T)));
    }

    std::size_t block_count() const { return blocks_.size(); }

private:
    std::size_t block_bytes_;
    std::size_t used_ = 0;              // bookmark inside the current (last) block
    std::vector<AlignedBlock> blocks_;  // moving these never moves their memory
};

}  // namespace vecdb