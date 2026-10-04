/**
 * @file prepared_vector.h
 * @brief Converts a user's vector into the form Layer 2 kernels require.
 */
#pragma once
#include <algorithm>
#include <span>
#include <stdexcept>

#include "aligned_block.h"
#include "distance.h"

namespace vecdb {

/**
 * @brief A 64-byte-aligned, zero-padded copy of one vector, normalized for cosine.
 *
 * Why it exists:
 *  - Layer 2 kernels need a padded length (a multiple of 16) and zero padding.
 *    Vectors in VectorStore already follow this; vectors from users do not.
 *  - Inserts and queries both go through this one class, so cosine
 *    normalization can never be applied to one and forgotten in the other.
 *
 * How it works:
 *  - Owns one AlignedBlock of `stride` floats, zeroed when created.
 *  - prepare() writes only the first `dim` values, so the padding stays zero
 *    across any number of calls.
 *
 * Not thread-safe: each thread or search should use its own object.
 */
class PreparedVector {
public:
    /// Creates a buffer for vectors of `dim` floats.
    explicit PreparedVector(std::size_t dim)
        : dim_(dim), stride_(round_up(dim, kFloatsPerLine)),
          block_(round_up(dim, kFloatsPerLine) * sizeof(float)) {
        if (dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    /// Copies `v` in and normalizes it if `metric` is Cosine.
    /// Throws std::invalid_argument if `v` does not have `dim` values.
    void prepare(std::span<const float> v, Metric metric) {
        if (v.size() != dim_) throw std::invalid_argument("vector has wrong dimension");
        float* dst = mutable_data();
        std::copy(v.begin(), v.end(), dst);
        if (metric == Metric::Cosine) normalize({dst, dim_});
    }

    /// Start of the padded row; pass this and stride() to a DistanceFn.
    const float* data() const { return reinterpret_cast<const float*>(block_.data()); }

    /// The real values only (length dim), e.g. for VectorStore::add.
    std::span<const float> values() const { return {data(), dim_}; }

    /// The full row including zero padding (length stride).
    std::span<const float> padded() const { return {data(), stride_}; }

    std::size_t dim() const { return dim_; }
    std::size_t stride() const { return stride_; }

private:
    float* mutable_data() { return reinterpret_cast<float*>(block_.data()); }

    std::size_t dim_, stride_;
    AlignedBlock block_;
};

}  // namespace vecdb