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
        ids_.add(id);                       // 2. new internal number (throws on duplicate)
        vectors_.add(scratch_.values());    // 3. same number in VectorStore
        ++live_;
    }

    bool FlatIndex::remove(std::uint64_t id) {
        auto node = ids_.find(id);
        if (!node || ids_.is_deleted(*node)) return false;
        ids_.mark_deleted(*node);  // tombstone: memory stays, search skips it
        --live_;
        return true;
    }

    bool FlatIndex::contains(std::uint64_t id) const {
        auto node = ids_.find(id);
        return node && !ids_.is_deleted(*node);
    }

    std::vector<SearchResult> FlatIndex::search(std::span<const float> query, std::size_t k) const {
        PreparedVector q(dim());
        q.prepare(query, metric_);  // validates even when the index is empty
        if (k == 0 || live_ == 0) return {};
        k = std::min(k, live_);  // asking for more than exists returns everything

        // Compare the query with every live vector, keeping the k closest.
        TopK top(k);
        const std::size_t count = vectors_.size(), stride = vectors_.stride();
        for (NodeId node = 0; node < count; ++node) {
            if (ids_.is_deleted(node)) continue;  // one byte read, cheaper than a distance
            top.push({distance_(q.data(), vectors_.get_padded(node).data(), stride), node});
        }

        // Translate internal numbers to the user's IDs.
        std::vector<SearchResult> results;
        for (const Candidate& c : top.take_sorted())
            results.push_back({ids_.external(c.id), c.distance});
        return results;
    }

}  // namespace vecdb