#pragma once
#include <cassert>
#include <type_traits>
#include <vector>

#include "aligned_block.h"

namespace vecdb {

    // "Ream of paper" allocator: grabs big blocks, hands out small pieces by
    // moving a bookmark forward. Nothing is freed individually; all blocks are
    // freed together when the Arena is destroyed. Not thread-safe (yet).
    class Arena {
    public:
        explicit Arena(std::size_t block_bytes = 1 << 20) : block_bytes_(block_bytes) {}

        // Returns `bytes` of memory starting at a multiple of `align`.
        void* allocate(std::size_t bytes, std::size_t align = alignof(std::max_align_t)) {
            assert(align <= kAlign && (align & (align - 1)) == 0);
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