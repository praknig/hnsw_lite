#pragma once
#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

#include "aligned_block.h"

namespace vecdb {

/**
 * @brief Stores all vectors in 64-byte-aligned, zero-padded rows.
 *
 * Layout:
 *  - Each vector gets the next number (0, 1, 2, ...).
 *  - Each row holds `dim` values followed by zeros, up to `stride` values,
 *    where stride = dim rounded up to a multiple of 16. Every row therefore
 *    starts on a 64-byte boundary. The padding is always zero.
 *  - Rows live in fixed-size blocks of 2^shelf_bits rows. Blocks are added
 *    when needed and never moved, so returned spans stay valid.
 *
 * Finding vector N (arithmetic only):
 *  - block = N >> shelf_bits, row = N & mask
 *  - address = block start + row * stride
 *
 * Rules:
 *  - add() only accepts vectors of exactly `dim` values.
 *  - get() returns a read-only std::span of `dim` values; get_padded() includes the padding.
 *  - Not thread-safe.
 */
class VectorStore {
public:
    explicit VectorStore(std::size_t dim, unsigned shelf_bits = 16)
        : dim_(dim), stride_(round_up(dim, kFloatsPerLine)),
          shelf_bits_(shelf_bits), mask_((std::size_t{1} << shelf_bits) - 1) {
        if (dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    // Copies a vector in and returns its number (0, 1, 2, ...).
    NodeId add(std::span<const float> v) {
        if (v.size() != dim_) throw std::invalid_argument("vector has wrong dimension");
        if (count_ >= kEmpty) throw std::length_error("too many vectors");
        if ((count_ & mask_) == 0)  // first row of a new shelf
            shelves_.emplace_back((mask_ + 1) * stride_ * sizeof(float));
        std::copy(v.begin(), v.end(), row(count_));  // padding stays zero
        return static_cast<NodeId>(count_++);
    }

    // A read-only window onto vector `id` (no copy). Length = dim.
    std::span<const float> get(NodeId id) const { return {row(check(id)), dim_}; }

    // Same, but including the zero padding. Length = stride.
    std::span<const float> get_padded(NodeId id) const { return {row(check(id)), stride_}; }

    std::size_t size() const { return count_; }
    std::size_t dim() const { return dim_; }
    std::size_t stride() const { return stride_; }

private:
    // Address math is done in size_t (64-bit) to avoid overflow.
    float* row(std::size_t id) const {
        auto* base = reinterpret_cast<float*>(shelves_[id >> shelf_bits_].data());
        return base + (id & mask_) * stride_;
    }
    std::size_t check(NodeId id) const {
        if (id >= count_) throw std::out_of_range("unknown vector id");
        return id;
    }

    std::size_t dim_, stride_;
    unsigned shelf_bits_;
    std::size_t mask_;
    std::size_t count_ = 0;
    std::vector<AlignedBlock> shelves_;
};

}  // namespace vecdb