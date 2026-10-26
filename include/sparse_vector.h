/**
 * @file sparse_vector.h
 * @brief SparseVector on its own: the type log records carry (wal_ops.h), without the
 * whole sparse index, so that sparse.h can include the write-ahead log machinery.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vecdb {

/// Terms and their weights. Order does not matter; terms must be distinct and
/// weights finite. Zero weights are dropped when stored.
struct SparseVector {
    std::vector<std::uint32_t> indices;
    std::vector<float> values;
    std::size_t size() const { return indices.size(); }
    bool empty() const { return indices.empty(); }
    friend bool operator==(const SparseVector&, const SparseVector&) = default;
};

}  // namespace vecdb
