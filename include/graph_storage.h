#pragma once
#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

#include "arena.h"

namespace vecdb {

/**
 * @brief Stores the HNSW neighbor lists of every node on every level.
 *
 * Level 0 (every node):
 *  - A fixed list of M0 = 2*M slots per node, stored in blocks of
 *    2^shelf_bits nodes and found by arithmetic, like VectorStore.
 *  - With M = 16 a list is 32 * 4 = 128 bytes, exactly two cache lines.
 *
 * Levels 1 and above (only some nodes):
 *  - A node at level L gets one Arena allocation of L * M slots:
 *    one list of M slots for each level from 1 to L.
 *  - upper_[N] points to that allocation, or is nullptr for level-0 nodes.
 *
 * Empty slots:
 *  - Unused slots hold kEmpty; count() returns the slots before the first kEmpty.
 *  - Level-0 blocks are filled with byte 0xFF, which makes every slot kEmpty.
 *
 * Rules:
 *  - Nodes are added in order; node N here matches vector N in VectorStore.
 *  - links() throws if the node does not reach the requested level.
 *  - Levels are limited to 0..255.
 *  - Not thread-safe.
 */
class GraphStorage {
public:
    explicit GraphStorage(std::size_t M = 16, unsigned shelf_bits = 16)
        : M_(M), M0_(2 * M), shelf_bits_(shelf_bits),
          mask_((std::size_t{1} << shelf_bits) - 1) {
        if (M == 0) throw std::invalid_argument("M must be > 0");
        free_upper_.resize(256);
    }

    // Registers the next node (its id is size()) with the given top level.
    // All-or-nothing: if it throws, nothing visible changed.
    NodeId add_node(int level) {
        if (level < 0 || level > 255) throw std::invalid_argument("bad level");
        auto id = static_cast<NodeId>(levels_.size());
        // Everything that can throw happens first, so a failure changes nothing.
        if ((std::size_t(id) >> shelf_bits_) >= layer0_.size())  // new shelf: 0xFF bytes = kEmpty slots
            layer0_.emplace_back((mask_ + 1) * M0_ * sizeof(NodeId), 0xFF);
        reserve_one_more(levels_);
        reserve_one_more(upper_);
        reserve_one_more(upper_cap_);
        NodeId* upper = nullptr;
        std::uint8_t cap = 0;
        if (level > 0) {
            upper = take_upper(level, cap);
            std::fill_n(upper, std::size_t(level) * M_, kEmpty);
        }
        levels_.push_back(static_cast<std::uint8_t>(level));  // cannot throw: reserved
        upper_.push_back(upper);
        upper_cap_.push_back(cap);
        return id;
    }

    // Writable window onto node `id`'s slots at `level` (M0 or M slots).
    std::span<NodeId> links(NodeId id, int level) {
        if (id >= levels_.size() || level < 0 || level > levels_[id])
            throw std::out_of_range("node does not exist on this level");
        if (level == 0) {
            auto* base = reinterpret_cast<NodeId*>(layer0_[id >> shelf_bits_].data());
            return {base + (id & mask_) * M0_, M0_};
        }
        return {upper_[id] + std::size_t(level - 1) * M_, M_};
    }
    std::span<const NodeId> links(NodeId id, int level) const {
        return const_cast<GraphStorage*>(this)->links(id, level);
    }

    // Number of used slots = position of the first kEmpty.
    static std::size_t count(std::span<const NodeId> slots) {
        return std::size_t(std::find(slots.begin(), slots.end(), kEmpty) - slots.begin());
    }

    /// Replaces node `id`'s list on `level` with `neighbors`, then marks every
    /// remaining slot kEmpty. Throws if `neighbors` does not fit in the slots.
    /// `neighbors` must not point into the slots being replaced.
    void set_links(NodeId id, int level, std::span<const NodeId> neighbors) {
        std::span<NodeId> slots = links(id, level);
        if (neighbors.size() > slots.size()) throw std::length_error("too many neighbors");
        auto end = std::copy(neighbors.begin(), neighbors.end(), slots.begin());
        std::fill(end, slots.end(), kEmpty);
    }

    int level(NodeId id) const { return levels_.at(id); }

    /// Undoes the most recent add_node(), before any links were written. Used to
    /// roll back a failed insert; the node's level-0 slots are still all kEmpty.
    void undo_last_add() noexcept {
        levels_.pop_back();
        upper_.pop_back();  // its block (if any) is simply left unused
        upper_cap_.pop_back();
    }
    std::size_t size() const { return levels_.size(); }
    std::size_t M() const { return M_; }
    std::size_t M0() const { return M0_; }

    /// Clears node `id` and gives it a new top level (slot reuse). Every list
    /// becomes empty. If the level changes, the node's old upper-level block is
    /// recycled and it takes the smallest recycled block that fits (a new one
    /// from the arena only if none fits; none at all for level 0). Throws on an invalid level or node,
    /// or if memory runs out; in every case nothing has changed.
    void reset_node(NodeId id, int level) {
        if (level < 0 || level > 255) throw std::invalid_argument("bad level");
        if (id >= levels_.size()) throw std::out_of_range("unknown node");
        // Unless the level is unchanged, the old block is recycled and the node
        // takes the smallest recycled block that fits (none for level 0), so
        // blocks circulate and churn does not grow memory.
        NodeId* upper = upper_[id];
        std::uint8_t cap = upper_cap_[id];
        if (level != levels_[id]) {  // a node's block always fits its current level
            const std::uint8_t old_cap = cap;
            if (old_cap > 0) reserve_one_more(free_upper_[old_cap]);  // may throw; nothing changed
            upper = nullptr;
            cap = 0;
            if (level > 0) upper = take_upper(level, cap);             // may throw only before any change
            if (old_cap > 0) free_upper_[old_cap].push_back(upper_[id]);  // cannot throw: reserved
        }
        auto* base = reinterpret_cast<NodeId*>(layer0_[std::size_t(id) >> shelf_bits_].data());
        std::fill_n(base + (std::size_t(id) & mask_) * M0_, M0_, kEmpty);
        if (level > 0) std::fill_n(upper, std::size_t(level) * M_, kEmpty);
        upper_[id] = upper;
        upper_cap_[id] = cap;
        levels_[id] = static_cast<std::uint8_t>(level);
    }

    /// Upper-level link slots ever taken from the arena (grows only when no
    /// recycled block fits; used by tests to check memory stays bounded).
    std::size_t upper_slots_allocated() const { return upper_slots_allocated_; }
    /// Upper-level blocks waiting to be reused.
    std::size_t recycled_upper_blocks() const {
        std::size_t n = 0;
        for (const auto& list : free_upper_) n += list.size();
        return n;
    }

private:
    /// An upper-level block for at least `level` levels: the smallest recycled
    /// block that fits, otherwise a new one from the arena. Sets `cap` to the
    /// block's size in levels. May throw only when allocating, before any change.
    NodeId* take_upper(int level, std::uint8_t& cap) {
        for (std::size_t c = std::size_t(level); c < free_upper_.size(); ++c) {
            if (free_upper_[c].empty()) continue;
            NodeId* block = free_upper_[c].back();
            free_upper_[c].pop_back();
            cap = static_cast<std::uint8_t>(c);
            return block;
        }
        NodeId* block = arena_.allocate_array<NodeId>(std::size_t(level) * M_);
        upper_slots_allocated_ += std::size_t(level) * M_;
        cap = static_cast<std::uint8_t>(level);
        return block;
    }

    std::size_t M_, M0_;
    unsigned shelf_bits_;
    std::size_t mask_;
    std::vector<AlignedBlock> layer0_;  // shelves of level-0 pages
    std::vector<std::uint8_t> levels_;  // top level of each node
    std::vector<NodeId*> upper_;        // upper-level block, or nullptr
    std::vector<std::uint8_t> upper_cap_;                 // levels each node's block can hold
    std::vector<std::vector<NodeId*>> free_upper_;        // recycled blocks, by capacity in levels
    std::size_t upper_slots_allocated_ = 0;
    Arena arena_;                       // memory for upper-level blocks
};

}  // namespace vecdb