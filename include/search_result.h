/**
 * @file search_result.h
 * @brief Result types and the top-k heap shared by FlatIndex and HnswIndex.
 */
#pragma once
#include <string>
#include <stdexcept>
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "common.h"

namespace vecdb {

/**
 * @brief Turns a NaN distance into +infinity, so it sorts last instead of
 *        breaking the heap order.
 *
 * Indexes reject NaN and infinite input, but two vectors with huge finite
 * values (around 1e19 or more) can still overflow inside an inner product:
 * +inf and -inf partial sums add up to NaN. Such a pair is treated as "as far
 * apart as possible".
 */
inline float ordered_distance(float d) {
    return d != d ? std::numeric_limits<float>::infinity() : d;  // only NaN differs from itself
}

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
 * @brief Counts searches in progress, so an index can refuse to change during
 *        one: from a predicate (which may search, but not change, the index) or
 *        from another thread (not allowed). Detection from another thread is
 *        best effort, since the two may simply not overlap in time.
 */
class SearchCounter {
public:
    /// Marks a search as running until destroyed.
    class Scope {
    public:
        explicit Scope(const SearchCounter& c) noexcept : c_(c) { c_.count_.fetch_add(1, std::memory_order_acq_rel); }
        ~Scope() { c_.count_.fetch_sub(1, std::memory_order_acq_rel); }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        const SearchCounter& c_;
    };

    Scope enter() const noexcept { return Scope(*this); }

    /// Throws std::logic_error if a search is running.
    void check_no_search(const char* operation) const {
        if (count_.load(std::memory_order_acquire) != 0)
            throw std::logic_error(std::string(operation) +
                                   " called while a search is running (from a predicate or another thread); "
                                   "change the index only when no search is in progress");
    }

private:
    mutable std::atomic<int> count_{0};
};

/// A distance as an index reports it: cosine distances are clamped at 0
/// (rounding can make 1 - dot slightly negative for nearly identical
/// directions, which is meaningless), then NaN is mapped as ordered_distance()
/// does. The raw kernels from get_distance() are left unchanged.
inline float index_distance(float raw, bool cosine) {
    if (cosine && raw < 0.0f) raw = 0.0f;
    return ordered_distance(raw);
}

/// "No limit" for the number of results of a range search.
inline constexpr std::size_t kNoLimit = std::numeric_limits<std::size_t>::max();

/// Range search with L2 compares squared distances: converts an ordinary
/// (Euclidean) radius into the squared radius search_range() expects.
inline float l2_radius(float euclidean_radius) { return euclidean_radius * euclidean_radius; }

/**
 * @brief What compact() did: how many vectors it kept and how many slots of
 *        removed vectors it reclaimed.
 */
struct CompactStats {
    std::size_t kept;
    std::size_t reclaimed;
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
    /// Memory is reserved for at most 1024 up front, so a huge k is safe.
    explicit TopK(std::size_t k) : k_(k) { heap_.reserve(std::min<std::size_t>(k, 1024)); }

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
    }  // GCOVR_EXCL_LINE: gcov counts this brace separately (return value optimization)

private:
    std::size_t k_;
    std::vector<Candidate> heap_;
};

}  // namespace vecdb