#pragma once
#include <cstring>
#include <new>
#include <utility>

#include "common.h"

namespace vecdb {

    // Owns one block of memory that starts on a 64-byte boundary and is filled
    // with a byte value (zero by default). Frees itself automatically (RAII).
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