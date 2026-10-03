#pragma once
#include <cstring>
#include <new>
#include <utility>

#include "common.h"

namespace vecdb {

    /**
     * @brief Owns one block of memory that starts on a 64-byte boundary.
     *
     * Responsibilities:
     *  - Requests `bytes` of memory (rounded up to a multiple of 64) from the system.
     *  - Fills every byte with `fill` (0 by default).
     *  - Frees the memory automatically in the destructor (RAII).
     *
     * Rules:
     *  - Not copyable: two copies would free the same memory twice.
     *  - Movable: ownership passes to the new object; the old one becomes empty.
     *  - The memory address never changes while the block is alive.
     *
     * Used by: VectorStore, GraphStorage and Arena. No other file calls new/delete.
     */
    class AlignedBlock {
    public:
        explicit AlignedBlock(std::size_t bytes, unsigned char fill = 0)
            : size_(round_up(bytes, kAlign)) {
            data_ = static_cast<std::byte*>(::operator new(size_, std::align_val_t{kAlign}));
            std::memset(data_, fill, size_);
        }
        ~AlignedBlock() { release(); }

        // Copying is forbidden: two owners would both free the same memory.
        AlignedBlock(const AlignedBlock&) = delete;
        AlignedBlock& operator=(const AlignedBlock&) = delete;

        // Moving is allowed: ownership is handed over, the old object becomes empty.
        AlignedBlock(AlignedBlock&& o) noexcept
            : data_(std::exchange(o.data_, nullptr)), size_(std::exchange(o.size_, 0)) {}
        AlignedBlock& operator=(AlignedBlock&& o) noexcept {
            if (this != &o) {
                release();
                data_ = std::exchange(o.data_, nullptr);
                size_ = std::exchange(o.size_, 0);
            }
            return *this;
        }

        std::byte* data() const { return data_; }
        std::size_t size() const { return size_; }

    private:
        void release() {
            if (data_) ::operator delete(data_, std::align_val_t{kAlign});
        }
        std::byte* data_ = nullptr;
        std::size_t size_ = 0;
    };

}  // namespace vecdb