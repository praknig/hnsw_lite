#pragma once
#include <optional>
#include <stdexcept>
#include <unordered_map>
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
        auto [it, inserted] = to_internal_.try_emplace(external, next);  // strong guarantee
        if (!inserted) throw std::invalid_argument("external id already exists");
        to_external_.push_back(external);  // cannot throw: capacity reserved
        deleted_.push_back(0);
        return next;
    }

    /// Undoes the most recent add(external). Used to roll back a failed insert.
    void undo_last_add(std::uint64_t external) noexcept {
        to_internal_.erase(external);
        to_external_.pop_back();
        deleted_.pop_back();
    }

    std::optional<NodeId> find(std::uint64_t external) const {
        auto it = to_internal_.find(external);
        if (it == to_internal_.end()) return std::nullopt;
        return it->second;
    }

    std::uint64_t external(NodeId id) const { return to_external_.at(id); }

    // Deleting only sets a flag ("tombstone"); memory is not reclaimed.
    void mark_deleted(NodeId id) { deleted_.at(id) = 1; }
    bool is_deleted(NodeId id) const { return deleted_.at(id) != 0; }

    /// Real deletion: frees the user ID so it can be added again, and marks its
    /// slot as free (deleted) so the slot can be reused. Returns the freed slot,
    /// or nothing if the ID is unknown or already released. Never throws.
    std::optional<NodeId> release(std::uint64_t external) noexcept {
        auto it = to_internal_.find(external);
        if (it == to_internal_.end()) return std::nullopt;
        const NodeId slot = it->second;
        to_internal_.erase(it);
        deleted_[slot] = 1;
        return slot;
    }

    /// True if `slot` exists, is deleted and no user ID maps to it (it was released).
    bool is_free(NodeId slot) const {
        if (slot >= to_external_.size() || !deleted_[slot]) return false;
        auto it = to_internal_.find(to_external_[slot]);
        return it == to_internal_.end() || it->second != slot;
    }

    /// Gives a free slot a new user ID (slot reuse). Throws std::logic_error if
    /// the slot is not free, std::invalid_argument if the ID is already in use.
    /// All-or-nothing: if it throws, nothing changed.
    void bind(NodeId slot, std::uint64_t external) {
        if (!is_free(slot)) throw std::logic_error("slot is not free");
        auto [it, inserted] = to_internal_.try_emplace(external, slot);  // strong guarantee
        if (!inserted) throw std::invalid_argument("external id already exists");
        to_external_[slot] = external;
        deleted_[slot] = 0;
    }

    /// Moves slot `from`'s entry into slot `to` (used by swap-with-last removal).
    /// `to` must have been released. Never throws.
    void move_slot(NodeId from, NodeId to) noexcept {
        const std::uint64_t external = to_external_[from];
        auto it = to_internal_.find(external);
        if (it != to_internal_.end() && it->second == from) it->second = to;
        to_external_[to] = external;
        deleted_[to] = deleted_[from];
    }

    /// Removes the last slot, which must already be released or moved. Never throws.
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
    void repoint(std::uint64_t external, NodeId slot) noexcept { to_internal_.find(external)->second = slot; }

    /// Frees a live slot that find() does not point to: the old version of an
    /// updated vector, or a new version being abandoned. Never throws.
    void retire(NodeId slot) noexcept { deleted_[slot] = 1; }

    void pop_back_slot() noexcept {
        to_external_.pop_back();
        deleted_.pop_back();
    }

    std::size_t size() const { return to_external_.size(); }

private:
    std::vector<std::uint64_t> to_external_;                // position N -> user id
    std::unordered_map<std::uint64_t, NodeId> to_internal_;  // user id -> N
    std::vector<std::uint8_t> deleted_;  // bytes, not vector<bool> (simpler, faster)
};

}  // namespace vecdb