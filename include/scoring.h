/**
 * @file scoring.h
 * @brief Re-scoring search results by metadata (ScoreModifiers), shared by the
 * indexes, hybrid search and collections.
 */
#pragma once

#include <algorithm>
#include <optional>
#include <vector>

#include "distance.h"
#include "filter.h"
#include "search_result.h"

namespace vecdb {

/// A positive relevance from a distance, higher for closer: 1 / (1 + d) for L2,
/// 1 - d / 2 for cosine (in [0, 1]), and the logistic of the dot product for inner
/// product (distance is minus the dot product).
inline double relevance_from_distance(float distance, Metric metric) {
    if (metric == Metric::Cosine) return std::clamp(1.0 - double(distance) / 2.0, 0.0, 1.0);
    if (metric == Metric::InnerProduct) return 1.0 / (1.0 + std::exp(double(distance)));
    return 1.0 / (1.0 + std::max(0.0, double(distance)));
}

/// Scores `found` by `m` (base: relevance(distance); metadata: metadata_of(id)),
/// orders by score (ties keep the search's order) and keeps the best k.
template <class MetadataOf, class Relevance>
std::vector<SearchResult> rescored(std::vector<SearchResult> found, std::size_t k, const ScoreModifiers& m,
                                   MetadataOf&& metadata_of, Relevance&& relevance) {
    for (SearchResult& r : found) {
        const std::optional<Metadata> md = metadata_of(r.id);
        r.score = static_cast<float>(m.apply(relevance(r.distance), md ? *md : Metadata{}));
    }
    std::stable_sort(found.begin(), found.end(), [](const SearchResult& a, const SearchResult& b) { return a.score > b.score; });
    if (found.size() > k) found.resize(k);
    return found;
}

/// Maximal marginal relevance: the order in which to keep up to k of the
/// candidates (indexes into `relevance`, which is sorted best first), each next
/// one maximizing lambda x relevance/max - (1 - lambda) x its highest cosine
/// similarity to those already kept. Candidates without a vector count as
/// dissimilar to everything. lambda 1 keeps the ranking.
inline std::vector<std::size_t> mmr_order(const std::vector<double>& relevance, const std::vector<std::optional<std::vector<float>>>& vectors,
                                          std::size_t k, double lambda) {
    const std::size_t n = relevance.size();
    double top = 0;
    for (const double r : relevance) top = std::max(top, r);
    auto cosine = [&](std::size_t a, std::size_t b) {
        if (!vectors[a] || !vectors[b]) return 0.0;
        double dot = 0, na = 0, nb = 0;
        for (std::size_t d = 0; d < vectors[a]->size(); ++d) {
            dot += double((*vectors[a])[d]) * (*vectors[b])[d];
            na += double((*vectors[a])[d]) * (*vectors[a])[d];
            nb += double((*vectors[b])[d]) * (*vectors[b])[d];
        }
        return na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : 0.0;
    };
    std::vector<std::size_t> chosen;
    std::vector<double> closest(n, -1.0);  // each candidate's highest similarity to the chosen ones
    std::vector<bool> taken(n, false);
    while (chosen.size() < std::min(k, n)) {
        std::size_t best = n;
        double best_value = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (taken[i]) continue;
            const double rel = top > 0 ? relevance[i] / top : 0.0;
            const double value = lambda * rel - (1 - lambda) * (chosen.empty() ? 0.0 : closest[i]);
            if (best == n || value > best_value) {  // ties keep the ranking's order
                best = i;
                best_value = value;
            }
        }
        taken[best] = true;
        chosen.push_back(best);
        for (std::size_t i = 0; i < n; ++i)
            if (!taken[i]) closest[i] = std::max(closest[i], cosine(i, best));
    }
    return chosen;
}

}  // namespace vecdb