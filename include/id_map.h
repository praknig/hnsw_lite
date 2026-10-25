#pragma once
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <vector>

#include "common.h"

namespace vecdb {

/**
 * @brief Translates between user IDs and internal NodeIds, and tracks deletions.
 *
 * Data:
 *  - to_external_: position N holds the user ID of vector N.
 *  - to_internal_: hash map from user ID to N.
 *  - deleted_: one byte per vector, 1 if the vector is deleted.
 *
 * Rules:
 *  - Internal numbers are given out in order: 0, 1, 2, ...
 *  - A user ID can be added only once.
 *  - Deleting only sets a flag ("tombstone"); nothing is freed or renumbered.
 *  - Used on insert, delete and lookup only, never inside the search loop.
 */
class IdMap {
public:
    /// Registers a user ID and returns its internal number.
    /// All-or-nothing: if it throws (duplicate ID or out of memory), nothing changed.
    NodeId add(std::uint64_t external) {
        auto next = static_cast<NodeId>(to_external_.size());
        reserve_one_more(to_external_);  // may throw; nothing changed yet
        reserve_one_more(deleted_);
        reserve_entry();
        if (locate(external) != kNone) throw std::invalid_argument("external id already exists");
        to_external_.push_back(external);  // cannot throw: capacity reserved
        deleted_.push_back(0);
        insert(next);
        return next;
    }

    /// Undoes the most recent add(external). Used to roll back a failed insert.
    void undo_last_add(std::uint64_t external) noexcept {
        if (const std::size_t b = locate(external); b != kNone) erase_bucket(b);  // while its key is still stored
        to_external_.pop_back();
        deleted_.pop_back();
    }

    std::optional<NodeId> find(std::uint64_t external) const {
        const std::size_t b = locate(external);
        if (b == kNone) return std::nullopt;
        return buckets_[b];
    }

    std::uint64_t external(NodeId id) const { return to_external_.at(id); }

    // Deleting only sets a flag ("tombstone"); memory is not reclaimed.
    void mark_deleted(NodeId id) { deleted_.at(id) = 1; }
    bool is_deleted(NodeId id) const { return deleted_.at(id) != 0; }

    /// Real deletion: frees the user ID so it can be added again, and marks its
    /// slot as free (deleted) so the slot can be reused. Returns the freed slot,
    /// or nothing if the ID is unknown or already released. Never throws.
    std::optional<NodeId> release(std::uint64_t external) noexcept {
        const std::size_t b = locate(external);
        if (b == kNone) return std::nullopt;
        const NodeId slot = buckets_[b];
        erase_bucket(b);
        deleted_[slot] = 1;
        return slot;
    }

    /// True if `slot` exists, is deleted and no user ID maps to it (it was released).
    bool is_free(NodeId slot) const {
        if (slot >= to_external_.size() || !deleted_[slot]) return false;
        const std::size_t b = locate(to_external_[slot]);
        return b == kNone || buckets_[b] != slot;
    }

    /// Gives a free slot a new user ID (slot reuse). Throws std::logic_error if
    /// the slot is not free, std::invalid_argument if the ID is already in use.
    /// All-or-nothing: if it throws, nothing changed.
    void bind(NodeId slot, std::uint64_t external) {
        if (!is_free(slot)) throw std::logic_error("slot is not free");
        reserve_entry();  // may throw; nothing changed yet
        if (locate(external) != kNone) throw std::invalid_argument("external id already exists");
        to_external_[slot] = external;
        deleted_[slot] = 0;
        insert(slot);
    }

    /// Moves slot `from`'s entry into slot `to` (used by swap-with-last removal).
    /// `to` must have been released. Never throws.
    void move_slot(NodeId from, NodeId to) noexcept {
        const std::uint64_t external = to_external_[from];
        if (const std::size_t b = locate(external); b != kNone && buckets_[b] == from) buckets_[b] = to;
        to_external_[to] = external;
        deleted_[to] = deleted_[from];
    }

    /// Appends a live slot holding `external` without making it findable:
    /// find(external) keeps returning the slot it returned before. update()
    /// builds the new version of a vector this way, then switches with repoint().
    /// All-or-nothing.
    NodeId add_unmapped(std::uint64_t external) {
        reserve_one_more(to_external_);  // may throw; nothing changed yet
        reserve_one_more(deleted_);
        to_external_.push_back(external);  // cannot throw: capacity reserved
        deleted_.push_back(0);
        return static_cast<NodeId>(to_external_.size() - 1);
    }

    /// Makes free slot `slot` live and holding `external`, without making it
    /// findable. Throws std::logic_error if the slot is not free.
    void occupy_unmapped(NodeId slot, std::uint64_t external) {
        if (!is_free(slot)) throw std::logic_error("slot is not free");
        to_external_[slot] = external;
        deleted_[slot] = 0;
    }

    /// Makes `find(external)` return `slot`. `external` must be stored. Never throws.
    void repoint(std::uint64_t external, NodeId slot) noexcept { buckets_[locate(external)] = slot; }

    /// Frees a live slot that find() does not point to: the old version of an
    /// updated vector, or a new version being abandoned. Never throws.
    void retire(NodeId slot) noexcept { deleted_[slot] = 1; }

    /// Removes the last slot, which must already be released or moved. Never throws.
    void pop_back_slot() noexcept {
        to_external_.pop_back();
        deleted_.pop_back();
    }

    std::size_t size() const { return to_external_.size(); }

private:
    // user id -> slot: an open-addressing table (linear probing) of slot numbers
    // only. A key is not stored twice: the key of the slot in a bucket is
    // to_external_[slot]. About 6 to 8 bytes per id, where a node-based
    // std::unordered_map takes about 40. Lookups are off the search path (searches
    // read to_external_ directly). Removal shifts entries back instead of leaving
    // tombstones, so it never allocates and never throws.
    static constexpr NodeId kFree = std::numeric_limits<NodeId>::max();
    static constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();

    static std::size_t mix(std::uint64_t x) noexcept {  // SplitMix64's finish: spreads sequential ids
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ull;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebull;
        return static_cast<std::size_t>(x ^ (x >> 31));
    }
    std::size_t home(std::uint64_t key) const noexcept { return mix(key) & (buckets_.size() - 1); }

    /// The bucket holding `key`, or kNone.
    std::size_t locate(std::uint64_t key) const noexcept {
        if (buckets_.empty()) return kNone;
        const std::size_t mask = buckets_.size() - 1;
        for (std::size_t b = home(key);; b = (b + 1) & mask) {
            const NodeId v = buckets_[b];
            if (v == kFree) return kNone;
            if (to_external_[v] == key) return b;
        }
    }

    /// Room for one more entry under a load of 0.7. May throw (allocation); changes
    /// nothing the map's users can see.
    void reserve_entry() {
        if ((count_ + 1) * 10 <= buckets_.size() * 7) return;
        std::vector<NodeId> grown(std::max<std::size_t>(16, buckets_.size() * 2), kFree);
        for (const NodeId v : buckets_)
            if (v != kFree) place(grown, v);
        buckets_.swap(grown);
    }

    void place(std::vector<NodeId>& table, NodeId v) const noexcept {
        const std::size_t mask = table.size() - 1;
        std::size_t b = mix(to_external_[v]) & mask;
        while (table[b] != kFree) b = (b + 1) & mask;
        table[b] = v;
    }

    void insert(NodeId v) noexcept {  // room reserved by reserve_entry()
        place(buckets_, v);
        ++count_;
    }

    /// Empties bucket `hole`, moving later entries of the same probe run back so every
    /// remaining key stays reachable from its home bucket.
    void erase_bucket(std::size_t hole) noexcept {
        const std::size_t mask = buckets_.size() - 1;
        for (std::size_t j = (hole + 1) & mask;; j = (j + 1) & mask) {
            const NodeId v = buckets_[j];
            if (v == kFree) break;
            const std::size_t h = home(to_external_[v]);
            if (((j - h) & mask) >= ((j - hole) & mask)) {  // v's home is not between hole and j
                buckets_[hole] = v;
                hole = j;
            }
        }
        buckets_[hole] = kFree;
        --count_;
    }

    std::vector<std::uint64_t> to_external_;  // position N -> user id
    std::vector<NodeId> buckets_;             // user id -> N (see above)
    std::size_t count_ = 0;                   // ids in buckets_
    std::vector<std::uint8_t> deleted_;       // bytes, not vector<bool> (simpler, faster)
};

}  // namespace vecdb