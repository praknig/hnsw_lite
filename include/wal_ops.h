/**
 * @file wal_ops.h
 * @brief The operations a write-ahead log records, and their encoding
 *        (durability, phase D2). Little-endian, bounds-checked when decoded.
 *
 * Every public change of an index is logged under its write lock before it is
 * applied; one that then fails is followed by an Abort record naming its LSN, and
 * replay skips it. Operations whose effect depends on the index's state are
 * logged as their effect: upsert as an add or an update, remove_where and
 * set_metadata_where as the IDs they matched.
 */
#pragma once
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "health.h"
#include "metadata.h"
#include "sparse.h"
#include "filter.h"

namespace vecdb::walop {

enum Type : std::uint8_t {
    kAdd = 1,             ///< id, vector as given, metadata
    kAddBatch = 2,        ///< ids, vectors, metadata (or none)
    kRemove = 3,          ///< one id (remove: repaired one at a time)
    kRemoveBatch = 8,     ///< ids (remove_batch, remove_where resolved: repaired as a batch)
    kUpdate = 4,          ///< id, vector, metadata (or none: kept)
    kSetMetadata = 5,     ///< id, metadata changes
    kSetMetadataIds = 6,  ///< ids, metadata changes (set_metadata_where, resolved)
    kAbort = 7,           ///< the LSN of a logged operation that failed
    kCompact = 10,
    kUpkeep = 11,         ///< UpkeepOptions
    kRebuild = 12,        ///< RebuildOptions
    kTrainCodec = 13,
    kTrainLists = 14,
    kCreatePayloadIndex = 15,  ///< field name
    kSetPlanner = 16,          ///< PlannerParams
    kSetAutoCompact = 17,      ///< fraction
    kReorder = 18,
    kDocument = 20,            ///< Collection: id, parts, vector, text, sparse vector, metadata (an upsert)
    kDocumentRemove = 21,      ///< Collection: ids
    kDocumentMetadata = 22,    ///< Collection: ids, metadata changes
    kCollectionSettings = 23,  ///< Collection: expiry enabled, default lifetime, string keys
};

/// A decoded operation; only the fields its type uses are set.
struct Op {
    Type type = kAdd;
    std::vector<std::uint64_t> ids;  ///< one for kAdd, kUpdate, kSetMetadata
    std::vector<float> vectors;      ///< ids.size() * dim values
    std::vector<Metadata> metadata;  ///< one per id, or empty (none given)
    std::uint64_t lsn = 0;           ///< kAbort
    UpkeepOptions upkeep;
    RebuildOptions rebuild;
    std::string field;
    PlannerParams planner;
    double fraction = 0;
    std::uint8_t parts = 0;  ///< kDocument: 1 vector, 2 text, 4 sparse vector
    std::string text;
    SparseVector sparse;
    std::uint8_t ttl_enabled = 0;     ///< kCollectionSettings
    std::int64_t ttl_default_ms = 0;
    std::uint8_t string_ids = 0;
};

std::vector<unsigned char> encode_add(std::uint64_t id, std::span<const float> vector, const Metadata& metadata);
std::vector<unsigned char> encode_add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors,
                                            std::span<const Metadata> metadata);
std::vector<unsigned char> encode_ids(std::span<const std::uint64_t> ids);
std::vector<unsigned char> encode_update(std::uint64_t id, std::span<const float> vector, const Metadata* metadata);
std::vector<unsigned char> encode_set_metadata(std::span<const std::uint64_t> ids, const Metadata& changes);
std::vector<unsigned char> encode_abort(std::uint64_t lsn);
std::vector<unsigned char> encode_upkeep(const UpkeepOptions& o);
std::vector<unsigned char> encode_rebuild(const RebuildOptions& o);
std::vector<unsigned char> encode_field(std::string_view field);
std::vector<unsigned char> encode_planner(const PlannerParams& p);
std::vector<unsigned char> encode_fraction(double fraction);
std::vector<unsigned char> encode_collection_settings(bool ttl_enabled, std::int64_t ttl_default_ms, bool string_ids);
std::vector<unsigned char> encode_document(std::uint64_t id, std::uint8_t parts, std::span<const float> vector, std::string_view text,
                                           const SparseVector& sparse, const Metadata& metadata);

/// Decodes a record. Throws FormatError for an unknown type or a payload that
/// does not match its type (a checksum guards the bytes; this guards the meaning).
Op decode(std::uint8_t type, std::span<const unsigned char> payload);

}  // namespace vecdb::walop