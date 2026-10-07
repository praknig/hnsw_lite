/**
 * @file visited_list.h
 * @brief Fast "have I already checked this node?" tracking for HNSW searches.
 */
#pragma once
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "common.h"

namespace vecdb {

/**
 * @brief Remembers which nodes one search has already visited.
 *
 * How it works:
 *  - One 32-bit mark per node, plus a current "search number" (epoch).
 *  - visit(N) writes the epoch into mark N; a node is visited if its mark
 *    equals the current epoch.
 *  - reset() starts a new search by adding 1 to the epoch, so the array never
 *    needs clearing. Only after ~4 billion searches, when the epoch would wrap
 *    to 0, is the array cleared once.
 *
 * A hash set would also work, but costs a hash and a memory allocation per
 * insert; this costs one read and one write.
 */
class VisitedList {
public:
    /// Starts a new search over `node_count` nodes, growing the array if needed.
    void reset(std::size_t node_count) {
        if (marks_.size() < node_count) marks_.resize(node_count, 0);
        if (++epoch_ == 0) {  // wrapped around: old marks could look current
            std::fill(marks_.begin(), marks_.end(), 0);
            epoch_ = 1;
        }
    }

    /// Marks `id` as visited. Returns true on the first visit in this search.
    bool visit(NodeId id) {
        if (marks_[id] == epoch_) return false;
        marks_[id] = epoch_;
        return true;
    }

    /// True if `id` was visited in this search.
    bool visited(NodeId id) const { return marks_[id] == epoch_; }

    /// Sets the epoch directly. Only for testing the wrap-around case.
    void set_epoch_for_testing(std::uint32_t epoch) { epoch_ = epoch; }

private:
    std::vector<std::uint32_t> marks_;
    std::uint32_t epoch_ = 0;
};

/**
 * @brief A thread-safe stock of VisitedLists that searches borrow and return.
 *
 * Why it exists: with a million nodes one VisitedList is 4 MB. Creating one
 * per search would cost more than the search itself, so lists are reused.
 *
 * How it works:
 *  - acquire() takes an idle list (or creates one if none is idle), resets it,
 *    and returns it inside a Handle.
 *  - The Handle gives the list back automatically when it goes out of scope,
 *    even if the search throws an exception. Giving it back never allocates
 *    (room is reserved when a list is created), so it can never fail.
 *  - A mutex protects the idle stack, so several searches can run at once.
 */
class VisitedListPool {
public:
    /**
     * @brief Borrowed VisitedList; returns it to the pool when destroyed (RAII).
     * Not copyable or movable: it lives exactly as long as one search.
     */
    class Handle {
    public:
        ~Handle() { pool_.release(std::move(list_)); }
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;

        VisitedList& operator*() { return *list_; }
        VisitedList* operator->() { return list_.get(); }

    private:
        friend class VisitedListPool;
        Handle(VisitedListPool& pool, std::unique_ptr<VisitedList> list)
            : pool_(pool), list_(std::move(list)) {}

        VisitedListPool& pool_;
        std::unique_ptr<VisitedList> list_;
    };

    /// Borrows a list and resets it for a search over `node_count` nodes.
    Handle acquire(std::size_t node_count) {
        std::unique_ptr<VisitedList> list;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!idle_.empty()) {
                list = std::move(idle_.back());
                idle_.pop_back();
            }
        }
        if (!list) {
            list = std::make_unique<VisitedList>();
            // Make room in the stock for every list ever created, so returning a
            // list (in the Handle's destructor) never needs to allocate.
            std::lock_guard<std::mutex> lock(mutex_);
            idle_.reserve(created_ + 1);
            ++created_;
        }
        list->reset(node_count);
        return Handle(*this, std::move(list));
    }

    /// Frees every idle list (for example after compact() shrank the index, so
    /// pooled lists sized for the old index would only waste memory).
    void clear() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        idle_.clear();
        created_ = 0;
    }

    /// Number of lists currently waiting in the pool.
    std::size_t idle_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return idle_.size();
    }

private:
    /// Puts a list back. Never allocates, because acquire() reserved room for
    /// every list it created. The try/catch is only a safety net.
    void release(std::unique_ptr<VisitedList> list) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            idle_.push_back(std::move(list));
            // GCOVR_EXCL_START: unreachable, capacity was reserved in acquire()
        } catch (...) {
        }
        // GCOVR_EXCL_STOP
    }

    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<VisitedList>> idle_;
    std::size_t created_ = 0;  // lists ever created; idle_ has room for all of them
};

}  // namespace vecdb