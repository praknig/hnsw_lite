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
        NodeId* upper = nullptr;
        if (level > 0) {
            upper = arena_.allocate_array<NodeId>(std::size_t(level) * M_);
            std::fill_n(upper, std::size_t(level) * M_, kEmpty);
        }
        levels_.push_back(static_cast<std::uint8_t>(level));  // cannot throw: reserved
        upper_.push_back(upper);
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
        upper_.pop_back();  // its arena block (if any) is simply left unused
    }
    std::size_t size() const { return levels_.size(); }
    std::size_t M() const { return M_; }
    std::size_t M0() const { return M0_; }

private:
    std::size_t M_, M0_;
    unsigned shelf_bits_;
    std::size_t mask_;
    std::vector<AlignedBlock> layer0_;  // shelves of level-0 pages
    std::vector<std::uint8_t> levels_;  // top level of each node
    std::vector<NodeId*> upper_;        // upper-level block, or nullptr
    Arena arena_;                       // memory for upper-level blocks
};

}  // namespace vecdb