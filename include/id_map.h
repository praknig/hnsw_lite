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

    std::size_t size() const { return to_external_.size(); }

private:
    std::vector<std::uint64_t> to_external_;                // position N -> user id
    std::unordered_map<std::uint64_t, NodeId> to_internal_;  // user id -> N
    std::vector<std::uint8_t> deleted_;  // bytes, not vector<bool> (simpler, faster)
};

}  // namespace vecdb