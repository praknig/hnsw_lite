/**
 * @file inspect.h
 * @brief What a file or a durable directory holds, read from its header (or its
 * DURABLE file) without loading it: for tools, and for choosing how to load it.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vecdb {

enum class StoredKind { Flat, Hnsw, Sparse, Collection, Disk };

/// "flat", "hnsw", "sparse", "collection" or "disk".
const char* stored_kind_name(StoredKind kind);

struct StoredInfo {
    StoredKind kind = StoredKind::Hnsw;
    bool durable = false;                   ///< a durable directory (durable.h), not a file
    std::uint32_t version = 0;              ///< file format version (a durable directory: its newest snapshot's)
    std::uint64_t bytes = 0;                ///< the file's size, or the directory's files together
    std::uint64_t identity = 0;             ///< durable directories: the identity its log segments carry
    std::vector<std::uint64_t> snapshots;   ///< durable directories: snapshot LSNs, ascending
    std::vector<std::uint64_t> segments;    ///< durable directories: first LSNs of the log segments, ascending
};

/// Identifies `path`: an index or collection file (save()), a disk index file, or a
/// durable directory. Reads only headers. Throws FormatError for anything else, and
/// std::runtime_error if `path` cannot be read.
StoredInfo identify(const std::string& path);

}  // namespace vecdb
