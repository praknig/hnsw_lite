#pragma once
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
    explicit Storage(std::size_t dim, std::size_t M = 16) : vectors_(dim), graph_(M) {}

    // `level` will be chosen randomly by the HNSW layer later.
    NodeId insert(std::uint64_t external, std::span<const float> v, int level) {
        if (v.size() != vectors_.dim()) throw std::invalid_argument("vector has wrong dimension");
        if (level < 0 || level > 255) throw std::invalid_argument("level must be in 0..255");
        NodeId id = ids_.add(external);  // all-or-nothing (duplicate or out of memory)
        try {
            vectors_.add(v);
        } catch (...) {
            ids_.undo_last_add(external);
            throw;
        }
        try {
            graph_.add_node(level);
        } catch (...) {
            vectors_.undo_last_add();
            ids_.undo_last_add(external);
            throw;
        }
        return id;
    }

    const VectorStore& vectors() const { return vectors_; }
    GraphStorage& graph() { return graph_; }
    const GraphStorage& graph() const { return graph_; }
    IdMap& ids() { return ids_; }
    const IdMap& ids() const { return ids_; }
    std::size_t size() const { return vectors_.size(); }

private:
    VectorStore vectors_;
    GraphStorage graph_;
    IdMap ids_;
};

}  // namespace vecdb