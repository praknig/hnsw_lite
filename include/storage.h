#pragma once
#include <optional>
#include <span>
#include <stdexcept>
#include "graph_storage.h"
#include "id_map.h"
#include "vector_store.h"

namespace vecdb {

/**
 * @brief Single entry point to Layer 1; keeps IdMap, VectorStore and GraphStorage in sync.
 *
 * insert() does, in order:
 *  1. Checks the vector size and the level (0..255).
 *  2. Gets a new NodeId from IdMap (fails on a duplicate user ID).
 *  3. Stores the vector in VectorStore.
 *  4. Creates the node's neighbor slots in GraphStorage.
 *
 * Rules:
 *  - All input checks happen before any change, so invalid input changes nothing.
 *  - All-or-nothing: if any step runs out of memory, the earlier steps are
 *    undone, so a failed insert never leaves the three parts out of sync.
 *  - The same vector always has the same NodeId in all three parts.
 *  - The level is passed in for now; the HNSW layer will choose it later.
 *  - Not thread-safe.
 */
class Storage {
public:
    /// With `keep_originals` (cosine indexes), each slot also keeps the vector
    /// exactly as given, before normalization, so get_vector() can return it.
    /// `original_dim` (0: `dim`) is the size of the vectors as given, larger than
    /// `dim` when the index truncates them (Matryoshka).
    explicit Storage(std::size_t dim, std::size_t M = 16, bool keep_originals = false, std::size_t original_dim = 0)
        : vectors_(dim), graph_(M) {
        if (keep_originals) originals_.emplace(original_dim == 0 ? dim : original_dim);
    }

    // `level` will be chosen randomly by the HNSW layer later.
    /// `original` is the vector as given (kept only with keep_originals; empty
    /// means the same as `v`). Every insert is all-or-nothing.
    NodeId insert(std::uint64_t external, std::span<const float> v, int level, std::span<const float> original = {}) {
        check_insert(v, level, original);
        NodeId id = ids_.add(external);  // all-or-nothing (duplicate or out of memory)
        try {
            add_vector(v, original);
        } catch (...) {
            ids_.undo_last_add(external);
            throw;
        }
        try {
            graph_.add_node(level);
        } catch (...) {
            undo_last_vector();
            ids_.undo_last_add(external);
            throw;
        }
        return id;
    }

    /// Puts a new vector into a free (released) slot instead of appending.
    /// Same checks as insert(); also throws std::logic_error if `slot` is not free.
    /// All-or-nothing: if anything fails, the slot is still free and nothing changed.
    NodeId insert_into(NodeId slot, std::uint64_t external, std::span<const float> v, int level,
                       std::span<const float> original = {}) {
        check_insert(v, level, original);
        ids_.bind(slot, external);  // all-or-nothing (slot not free, duplicate, out of memory)
        try {
            graph_.reset_node(slot, level);  // all-or-nothing
        } catch (...) {
            ids_.release(external);  // the slot is free again
            throw;
        }
        overwrite_vector(slot, v, original);  // cannot fail: dimensions and slot already checked
        return slot;
    }

    /// Like insert(), but the slot holds `external` without being findable
    /// (see IdMap::add_unmapped). All-or-nothing.
    NodeId insert_unmapped(std::uint64_t external, std::span<const float> v, int level,
                           std::span<const float> original = {}) {
        check_insert(v, level, original);
        const NodeId id = ids_.add_unmapped(external);
        try {
            add_vector(v, original);
        } catch (...) {
            ids_.pop_back_slot();
            throw;
        }
        try {
            graph_.add_node(level);
        } catch (...) {
            undo_last_vector();
            ids_.pop_back_slot();
            throw;
        }
        return id;
    }

    /// Like insert_into(), but the slot holds `external` without being findable.
    /// All-or-nothing.
    NodeId insert_into_unmapped(NodeId slot, std::uint64_t external, std::span<const float> v, int level,
                                std::span<const float> original = {}) {
        check_insert(v, level, original);
        if (!ids_.is_free(slot)) throw std::logic_error("slot is not free");
        graph_.reset_node(slot, level);           // all-or-nothing
        ids_.occupy_unmapped(slot, external);     // cannot fail: checked above
        overwrite_vector(slot, v, original);      // cannot fail: dimensions and slot checked
        return slot;
    }

    const VectorStore& vectors() const { return vectors_; }

    /// The vectors as given, before normalization; null unless kept (cosine).
    const VectorStore* originals() const { return originals_ ? &*originals_ : nullptr; }

    /// The vector of `slot` as given: the original if kept, else the stored one.
    std::span<const float> original(NodeId slot) const { return originals_ ? originals_->get(slot) : vectors_.get(slot); }
    GraphStorage& graph() { return graph_; }
    const GraphStorage& graph() const { return graph_; }
    IdMap& ids() { return ids_; }
    const IdMap& ids() const { return ids_; }
    std::size_t size() const { return vectors_.size(); }

private:
    void check_insert(std::span<const float> v, int level, std::span<const float> original) const {
        if (v.size() != vectors_.dim()) throw std::invalid_argument("vector has wrong dimension");
        if (!original.empty() && original.size() != (originals_ ? originals_->dim() : vectors_.dim()))
            throw std::invalid_argument("original has wrong dimension");
        if (original.empty() && originals_ && originals_->dim() != vectors_.dim())
            throw std::logic_error("a truncating index needs the vector as given");
        if (level < 0 || level > 255) throw std::invalid_argument("level must be in 0..255");
    }
    void add_vector(std::span<const float> v, std::span<const float> original) {
        vectors_.add(v);
        if (!originals_) return;
        try {
            originals_->add(original.empty() ? v : original);
        } catch (...) {
            vectors_.undo_last_add();
            throw;
        }
    }
    void undo_last_vector() noexcept {
        vectors_.undo_last_add();
        if (originals_) originals_->undo_last_add();
    }
    void overwrite_vector(NodeId slot, std::span<const float> v, std::span<const float> original) {
        vectors_.overwrite(slot, v);
        if (originals_) originals_->overwrite(slot, original.empty() ? v : original);
    }

    VectorStore vectors_;
    std::optional<VectorStore> originals_;  // cosine: the vectors as given
    GraphStorage graph_;
    IdMap ids_;
};

}  // namespace vecdb