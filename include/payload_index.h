/**
* @file payload_index.h
 * @brief Exact per-value counts for indexed metadata fields (the payload index).
 */
#pragma once
#include <cstdint>
#include <vector>

namespace vecdb {

    /**
     * @brief How many live vectors hold each value of one indexed field.
     *
     * Used by the query planner: for a filter like `category == "news"` on an
     * indexed field, the number of matching vectors is known exactly instead of
     * being estimated by sampling.
     *
     * Values are dictionary IDs (keyword and tag fields) or 0/1 (boolean fields).
     * ensure() may allocate; add() and remove() never do, so callers size the
     * counts first and then update them inside non-throwing commit steps.
     *
     * This first version keeps counts only. Posting lists (the matching vectors
     * themselves, so exact search could skip a column scan) are a later step.
     */
    class ValueCounts {
    public:
        /// Makes room for value IDs 0 .. values-1. May allocate.
        void ensure(std::size_t values) {
            if (counts_.size() < values) counts_.resize(values, 0);
        }

        /// Sets every count to zero for `values` value IDs. May allocate.
        void reset(std::size_t values) { counts_.assign(values, 0); }

        /// One more vector holds `id`. `id` must be below the ensured size.
        void add(std::uint32_t id) noexcept { ++counts_[id]; }

        /// Sets the count of `id` directly (used when renumbering the dictionary).
        void set(std::uint32_t id, std::size_t n) noexcept { counts_[id] = n; }

        /// One vector fewer holds `id`.
        void remove(std::uint32_t id) noexcept { --counts_[id]; }

        /// Number of live vectors holding `id` (0 for an unknown ID).
        std::size_t count(std::uint32_t id) const noexcept { return id < counts_.size() ? counts_[id] : 0; }

    private:
        std::vector<std::size_t> counts_;
    };

}  // namespace vecdb