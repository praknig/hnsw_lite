#pragma once
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "common.h"

namespace vecdb {

    // Translates between the user's ids (any 64-bit number) and our dense
    // internal numbers (0, 1, 2, ...). Also remembers which vectors are deleted.
    // Used on insert/delete/lookup only, never inside the search loop.
    class IdMap {
    public:
        NodeId add(std::uint64_t external) {
            auto next = static_cast<NodeId>(to_external_.size());
            auto [it, inserted] = to_internal_.try_emplace(external, next);
            if (!inserted) throw std::invalid_argument("external id already exists");
            to_external_.push_back(external);
            deleted_.push_back(0);
            return next;
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