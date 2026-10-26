/**
 * @file facets.h
 * @brief Faceted counts: how many documents hold each value of a metadata field
 * ("120 in English, 40 in French"), over a whole index, the matches of a filter, or
 * a given set of documents such as the top hits of a search.
 *
 * One FacetCounter counts one field over the documents it is given (add()), so every
 * index type counts the same way. Results are deterministic: values by count, most
 * frequent first, then by value.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "metadata.h"

namespace vecdb {

/// One facet: which field to count, and how.
struct FacetRequest {
    std::string field;
    /// The most frequent values to report (every type); 0 reports them all. The
    /// documents holding the others are summed in FacetResult::other.
    std::size_t limit = 10;
    /// Integer and float fields: count range buckets between these boundaries
    /// (finite, strictly ascending) instead of values. n boundaries make n + 1 buckets:
    /// below the first, [b0, b1), ..., and from the last on.
    std::vector<double> boundaries;
    /// Also count the distinct values present (exact).
    bool distinct = false;

    /// Throws std::invalid_argument for an empty field name or invalid boundaries.
    void validate() const;
};

/// A value and how many documents hold it. Tag facets report each tag as a keyword.
struct FacetValue {
    Value value;
    std::size_t count = 0;
};

/// A numeric range [from, to): `from` empty below the first boundary, `to` empty from
/// the last boundary on.
struct FacetBucket {
    std::optional<double> from;
    std::optional<double> to;
    std::size_t count = 0;
};

struct FacetResult {
    std::string field;
    std::size_t total = 0;                ///< documents counted
    std::size_t missing = 0;              ///< of those, without the field
    std::vector<FacetValue> values;       ///< without boundaries: most frequent first, then by value
    std::size_t other = 0;                ///< summed counts of the values beyond `limit`
    std::vector<FacetBucket> buckets;     ///< with boundaries: one per range, the open ends included
    std::optional<std::size_t> distinct;  ///< when asked: distinct values present
};

/**
 * @brief Counts one field over documents given one at a time.
 *
 * The constructor resolves the field: unknown under a strict schema throws
 * std::invalid_argument (as filters do); unknown under a dynamic schema counts every
 * document as missing. Boundaries on a field that is not integer or float throw.
 */
class FacetCounter {
public:
    FacetCounter(const MetadataStore& store, const FacetRequest& request);

    /// Counts the document in `slot` (the caller passes live documents only).
    void add(NodeId slot);

    /// Counts `documents` documents at once from the store's exact per-value counts,
    /// for an indexed keyword, tag or boolean field over every live document (no
    /// filter): the same result as adding each document, without reading them.
    /// Returns false (nothing counted) when that does not apply.
    bool add_all_from_index(std::size_t documents);

    FacetResult result() const;

private:
    const MetadataStore& store_;
    FacetRequest request_;
    std::optional<std::size_t> field_;
    FieldType type_ = FieldType::Int;
    std::size_t total_ = 0, missing_ = 0;
    std::unordered_map<std::uint32_t, std::size_t> by_string_;  // keyword and tag dictionary IDs
    std::unordered_map<std::int64_t, std::size_t> by_int_;
    std::unordered_map<std::uint64_t, std::size_t> by_float_;   // bit patterns, -0 counted as 0
    std::size_t by_bool_[2] = {0, 0};
    std::vector<std::size_t> buckets_;

    void add_number(double x);
};

/// Counts every request over these documents (live slots), such as a filter's
/// matches or given IDs. Every request is checked before anything is counted.
std::vector<FacetResult> count_facets(const MetadataStore& store, const std::vector<FacetRequest>& requests,
                                      const std::vector<NodeId>& slots);

/// Counts every request over all `live` documents: an indexed keyword, tag or boolean
/// field from its per-value counts, without reading documents; the others from the
/// slots `all_slots()` returns, called at most once and only if a request needs them.
std::vector<FacetResult> count_facets(const MetadataStore& store, const std::vector<FacetRequest>& requests, std::size_t live,
                                      const std::function<std::vector<NodeId>()>& all_slots);

}  // namespace vecdb
