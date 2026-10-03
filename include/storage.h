#pragma once
#include "graph_storage.h"
#include "id_map.h"
#include "vector_store.h"

namespace vecdb {

    // The doorway to Layer 1. Keeps the three parts in sync so that a vector's
    // number is the same in IdMap, VectorStore and GraphStorage.
    class Storage {
    public:
        explicit Storage(std::size_t dim, std::size_t M = 16) : vectors_(dim), graph_(M) {}

        // `level` will be chosen randomly by the HNSW layer later.
        NodeId insert(std::uint64_t external, std::span<const float> v, int level) {
            if (v.size() != vectors_.dim()) throw std::invalid_argument("vector has wrong dimension");
            NodeId id = ids_.add(external);  // throws on duplicates before anything changes
            vectors_.add(v);
            graph_.add_node(level);
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