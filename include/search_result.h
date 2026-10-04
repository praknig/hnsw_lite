/**
 * @file search_result.h
 * @brief Result types and the top-k heap shared by FlatIndex and HnswIndex.
 */
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "common.h"

namespace vecdb {

/**
 * @brief One search result as the user sees it.
 *
 * `id` is the user's own ID (not the internal NodeId). `distance` follows the
 * Layer 2 rule: smaller means closer.
 */
struct SearchResult {
    std::uint64_t id;
    float distance;
};

/**
 * @brief Internal search result: an internal NodeId and its distance.
 *
 * Layer 3 works with NodeIds everywhere (they index Layer 1's arrays directly)
 * and converts to user IDs only when returning results.
 *
 * Ordered by distance, then by id, so equal distances always sort the same way.
 */
struct Candidate {
    float distance;
    NodeId id;

    /// True if `a` is closer than `b` (ties broken by the smaller id).
    friend bool operator<(const Candidate& a, const Candidate& b) {
        return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
    }
    /// Reverse order, used to build min-heaps.
    friend bool operator>(const Candidate& a, const Candidate& b) { return b < a; }
};

/**
 * @brief Keeps the k best (closest) candidates seen so far.
 *
 * How it works:
 *  - A max-heap stored in a std::vector: front() is always the WORST kept candidate.
 *  - push() adds a candidate while there is room; once full, a new candidate
 *    replaces the worst one only if it is better. Each push costs O(log k).
 *  - take_sorted() empties the heap and returns the candidates closest first.
 *
 * Used by FlatIndex for the final results and by HnswIndex for its beam search.
 */
class TopK {
public:
    /// Creates an empty heap that keeps at most `k` candidates.
    explicit TopK(std::size_t k) : k_(k) { heap_.reserve(k); }

    /// Offers a candidate. Returns true if it was kept.
    bool push(Candidate c) {
        if (k_ == 0) return false;
        if (heap_.size() < k_) {
            heap_.push_back(c);
            std::push_heap(heap_.begin(), heap_.end());
            return true;
        }
        if (!(c < heap_.front())) return false;  // not better than the worst kept
        std::pop_heap(heap_.begin(), heap_.end());  // moves the worst to the back
        heap_.back() = c;
        std::push_heap(heap_.begin(), heap_.end());
        return true;
    }

    /// True when the heap holds k candidates.
    bool full() const { return heap_.size() >= k_; }

    /// Number of candidates currently kept.
    std::size_t size() const { return heap_.size(); }

    /// Distance of the worst kept candidate, or infinity if the heap is empty.
    float worst_distance() const {
        return heap_.empty() ? std::numeric_limits<float>::infinity() : heap_.front().distance;
    }

    /// Removes all candidates and returns them sorted closest first.
    std::vector<Candidate> take_sorted() {
        std::vector<Candidate> out = std::move(heap_);
        heap_.clear();
        std::sort_heap(out.begin(), out.end());
        return out;
    }

private:
    std::size_t k_;
    std::vector<Candidate> heap_;
};

}  // namespace vecdb