/**
 * @file prepared_vector.h
 * @brief Converts a user's vector into the form Layer 2 kernels require.
 */
#pragma once
#include <algorithm>
#include <cmath>
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
 *  - NaN and infinity are rejected: one such value would corrupt every
 *    distance computed against this vector.
 *
 * Not thread-safe: each thread or search should use its own object.
 */
class PreparedVector {
public:
    /// Creates a buffer for vectors of `dim` floats.
    explicit PreparedVector(std::size_t dim)
        : dim_(checked_dim(dim)), stride_(round_up(checked_dim(dim), kFloatsPerLine)),
          block_(round_up(checked_dim(dim), kFloatsPerLine) * sizeof(float)) {}  // checked before the allocation

    /// Copies `v` in and normalizes it if `metric` is Cosine.
    /// Throws std::invalid_argument if `v` does not have `dim` values or contains
    /// NaN or infinity. On error the previous contents are left unchanged.
    void prepare(std::span<const float> v, Metric metric) {
        if (v.size() != dim_) throw std::invalid_argument("vector has wrong dimension");
        if (!std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); }))
            throw std::invalid_argument("vector contains NaN or infinity");
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

/// Prepares an input vector of `input_dim` values for an index that may use
/// only its first out.dim() values (Matryoshka truncation): checks the whole
/// vector (its size, and that every value is finite, including the part that is
/// cut off), then prepares the first out.dim() values (normalizing those for
/// cosine). Without truncation it is exactly out.prepare(v, metric).
inline void prepare_truncated(PreparedVector& out, std::span<const float> v, std::size_t input_dim, Metric metric) {
    if (out.dim() == input_dim) {
        out.prepare(v, metric);
        return;
    }
    if (v.size() != input_dim) throw std::invalid_argument("vector has wrong dimension");
    for (float x : v)
        if (!std::isfinite(x)) throw std::invalid_argument("vector contains NaN or infinity");
    out.prepare(v.first(out.dim()), metric);
}

}  // namespace vecdb