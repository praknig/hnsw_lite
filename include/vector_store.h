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
 *  - By default a block holds up to 65,536 rows but at most 8 MiB, so a small
 *    index of long vectors does not allocate hundreds of megabytes up front.
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
    /// Pass as shelf_bits to choose the block size automatically.
    static constexpr unsigned kAutoShelfBits = ~0u;

    /// Creates a store for vectors of `dim` floats. `shelf_bits` sets the rows
    /// per block (2^shelf_bits); by default it is chosen from the vector size.
    explicit VectorStore(std::size_t dim, unsigned shelf_bits = kAutoShelfBits)
        : dim_(dim), stride_(round_up(dim, kFloatsPerLine)),
          shelf_bits_(shelf_bits == kAutoShelfBits ? auto_shelf_bits(stride_) : shelf_bits),
          mask_((std::size_t{1} << shelf_bits_) - 1) {
        if (dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    /// Rows per block: up to 65,536, but at most 8 MiB per block.
    static unsigned auto_shelf_bits(std::size_t stride) {
        unsigned bits = 16;
        while (bits > 0 && (std::size_t{1} << bits) * stride * sizeof(float) > (std::size_t{8} << 20)) --bits;
        return bits;
    }

    // Copies a vector in and returns its number (0, 1, 2, ...).
    // All-or-nothing: if it throws, nothing changed.
    NodeId add(std::span<const float> v) {
        if (v.size() != dim_) throw std::invalid_argument("vector has wrong dimension");
        if (count_ >= kEmpty) throw std::length_error("too many vectors");
        if ((count_ >> shelf_bits_) >= shelves_.size())  // row lies past the last shelf
            shelves_.emplace_back((mask_ + 1) * stride_ * sizeof(float));
        std::copy(v.begin(), v.end(), row(count_));  // padding stays zero
        return static_cast<NodeId>(count_++);
    }

    // A read-only window onto vector `id` (no copy). Length = dim.
    std::span<const float> get(NodeId id) const { return {row(check(id)), dim_}; }

    // Same, but including the zero padding. Length = stride.
    std::span<const float> get_padded(NodeId id) const { return {row(check(id)), stride_}; }

    std::size_t size() const { return count_; }

    /// Undoes the most recent add(). Used to roll back a failed insert. The row
    /// is simply reused by the next add(); its padding is still zero.
    void undo_last_add() noexcept { --count_; }

    /// Removes the last row (used by swap-with-last removal). Its memory is
    /// reused by the next add(), without allocating; its padding stays zero.
    void pop_back() noexcept { --count_; }

    /// Replaces the values of row `id` (slot reuse). Padding stays zero and the
    /// address does not change. Throws on a wrong dimension or unknown id,
    /// before changing anything.
    void overwrite(NodeId id, std::span<const float> v) {
        if (v.size() != dim_) throw std::invalid_argument("vector has wrong dimension");
        std::copy(v.begin(), v.end(), row(check(id)));
    }

    /// Copies row `from` over row `to`, padding included (used by swap-with-last
    /// removal). Throws std::out_of_range for an unknown row, before changing anything.
    void move_row(NodeId from, NodeId to) {
        const float* src = row(check(from));
        std::copy_n(src, stride_, row(check(to)));
    }
    std::size_t rows_per_shelf() const { return mask_ + 1; }
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