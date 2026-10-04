/**
 * @file flat_index.cpp
 * @brief Implementation of FlatIndex (see flat_index.h for the design).
 */
#include "flat_index.h"

#include <algorithm>

namespace vecdb {

FlatIndex::FlatIndex(std::size_t dim, Metric metric)
    : metric_(metric), distance_(get_distance(metric)), vectors_(dim), scratch_(dim) {}

void FlatIndex::add(std::uint64_t id, std::span<const float> vector) {
    scratch_.prepare(vector, metric_);  // 1. check size, copy, normalize (throws first)
    ids_.add(id);                       // 2. new internal number (all-or-nothing)
    try {
        vectors_.add(scratch_.values());  // 3. same number in VectorStore
    } catch (...) {
        ids_.undo_last_add(id);  // out of memory: undo step 2 so nothing changed
        throw;
    }
    ++live_;
}

bool FlatIndex::remove(std::uint64_t id) {
    const auto node = ids_.find(id);
    if (!node) return false;
    // Swap-with-last: move the last vector into the freed slot, then shrink by one.
    // Every step below is non-allocating and cannot throw.
    const auto last = static_cast<NodeId>(vectors_.size() - 1);
    ids_.release(id);
    if (*node != last) {
        vectors_.move_row(last, *node);
        ids_.move_slot(last, *node);
    }
    ids_.pop_back_slot();
    vectors_.pop_back();
    --live_;
    return true;
}

bool FlatIndex::contains(std::uint64_t id) const { return ids_.find(id).has_value(); }

std::vector<SearchResult> FlatIndex::search(std::span<const float> query, std::size_t k) const {
    PreparedVector q(dim());
    q.prepare(query, metric_);  // validates even when the index is empty
    if (k == 0 || live_ == 0) return {};
    k = std::min(k, live_);  // asking for more than exists returns everything

    // Compare the query with every vector, keeping the k closest.
    TopK top(k);
    const std::size_t count = vectors_.size(), stride = vectors_.stride();
    for (NodeId node = 0; node < count; ++node)
        top.push({ordered_distance(distance_(q.data(), vectors_.get_padded(node).data(), stride)), node});

    // Translate internal numbers to the user's IDs.
    std::vector<SearchResult> results;
    results.reserve(k);
    for (const Candidate& c : top.take_sorted()) results.push_back({ids_.external(c.id), c.distance});
    return results;
}

}  // namespace vecdb