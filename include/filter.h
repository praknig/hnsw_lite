/**
 * @file filter.h
 * @brief Metadata filters, their compiled form, search options and the query
 *        planner's interface.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "id_map.h"
#include "metadata.h"

namespace vecdb {

enum class CompareOp { Eq, Ne, Lt, Le, Gt, Ge };

/**
 * @brief A condition on metadata, such as
 *        Filter::eq("category", "news") && Filter::ge("year", 2020).
 *
 * Semantics:
 *  - Comparisons never match a vector that lacks the field, including `ne`
 *    (as in SQL). `!` negates the whole condition, so !Filter::eq(...) does
 *    match vectors without the field. Filter::exists(field) tests presence.
 *  - Integers and floats compare exactly with each other.
 *  - Keywords support eq, ne and in; booleans eq, ne and in; tag sets
 *    has_tag, has_any_tag and has_all_tags (has_all_tags with no tags matches
 *    every vector that has the field).
 *  - between(lo, hi) is inclusive; lo > hi matches nothing; an empty `in` list
 *    matches nothing.
 *
 * Filters may be nested at most kMaxDepth (256) levels deep.
 *
 * A Filter is a cheap, shareable value. It is checked against an index's
 * schema each time it is used. NaN values throw std::invalid_argument here.
 */
class Filter {
public:
    /// One node of the expression tree.
    struct Node {
        enum class Kind { All, Compare, Between, In, HasTag, HasAnyTag, HasAllTags, Exists, And, Or, Not };
        Kind kind = Kind::All;
        std::string field;
        CompareOp op = CompareOp::Eq;
        Value a, b;
        std::vector<Value> values;
        std::vector<std::string> tags;
        std::vector<std::shared_ptr<const Node>> children;
        int depth = 1;  // levels in this subtree
    };

    /// Deepest filter allowed. Filters are evaluated recursively, so an
    /// unlimited depth (for example, chaining && in a loop 100,000 times) would
    /// overflow the stack. Combine many conditions with in() or has_any_tag(),
    /// or as a balanced tree, instead.
    static constexpr int kMaxDepth = 256;

    /// Matches every live vector (the same as no filter).
    Filter() : node_(std::make_shared<Node>()) {}
    static Filter all() { return Filter(); }

    template <class T> static Filter eq(std::string field, T&& v) { return compare(std::move(field), CompareOp::Eq, Value::from(std::forward<T>(v))); }
    template <class T> static Filter ne(std::string field, T&& v) { return compare(std::move(field), CompareOp::Ne, Value::from(std::forward<T>(v))); }
    template <class T> static Filter lt(std::string field, T&& v) { return compare(std::move(field), CompareOp::Lt, Value::from(std::forward<T>(v))); }
    template <class T> static Filter le(std::string field, T&& v) { return compare(std::move(field), CompareOp::Le, Value::from(std::forward<T>(v))); }
    template <class T> static Filter gt(std::string field, T&& v) { return compare(std::move(field), CompareOp::Gt, Value::from(std::forward<T>(v))); }
    template <class T> static Filter ge(std::string field, T&& v) { return compare(std::move(field), CompareOp::Ge, Value::from(std::forward<T>(v))); }

    /// lo <= field <= hi.
    template <class T, class U>
    static Filter between(std::string field, T&& lo, U&& hi) {
        auto n = std::make_shared<Node>();
        n->kind = Node::Kind::Between;
        n->field = std::move(field);
        n->a = checked(Value::from(std::forward<T>(lo)));
        n->b = checked(Value::from(std::forward<U>(hi)));
        return Filter(std::move(n));
    }

    /// field equals any of the values.
    template <class T>
    static Filter in(std::string field, const std::vector<T>& values) {
        auto n = std::make_shared<Node>();
        n->kind = Node::Kind::In;
        n->field = std::move(field);
        for (const T& v : values) n->values.push_back(checked(Value::from(v)));
        return Filter(std::move(n));
    }
    template <class T>
    static Filter in(std::string field, std::initializer_list<T> values) {
        return in(std::move(field), std::vector<T>(values));
    }

    static Filter has_tag(std::string field, std::string tag) { return tags_node(Node::Kind::HasTag, std::move(field), {std::move(tag)}); }
    static Filter has_any_tag(std::string field, std::vector<std::string> tags) { return tags_node(Node::Kind::HasAnyTag, std::move(field), std::move(tags)); }
    static Filter has_all_tags(std::string field, std::vector<std::string> tags) { return tags_node(Node::Kind::HasAllTags, std::move(field), std::move(tags)); }

    /// The vector has a value for `field`.
    static Filter exists(std::string field) {
        auto n = std::make_shared<Node>();
        n->kind = Node::Kind::Exists;
        n->field = std::move(field);
        return Filter(std::move(n));
    }

    friend Filter operator&&(const Filter& x, const Filter& y) { return combine(Node::Kind::And, x, y); }
    friend Filter operator||(const Filter& x, const Filter& y) { return combine(Node::Kind::Or, x, y); }
    friend Filter operator!(const Filter& x) {
        auto n = std::make_shared<Node>();
        n->kind = Node::Kind::Not;
        n->depth = checked_depth(1 + x.node_->depth);
        n->children.push_back(x.node_);
        return Filter(std::move(n));
    }

    const Node& root() const { return *node_; }
    /// True for the default "match everything" filter.
    bool is_all() const { return node_->kind == Node::Kind::All; }

private:
    explicit Filter(std::shared_ptr<const Node> n) : node_(std::move(n)) {}

    static int checked_depth(int depth) {
        if (depth > kMaxDepth)
            throw std::invalid_argument("filter is nested more than " + std::to_string(kMaxDepth) + " levels deep");
        return depth;
    }
    static Value checked(Value v) {
        if (v.type == FieldType::Float && std::isnan(v.f)) throw std::invalid_argument("filter value is NaN");
        return v;
    }
    static Filter compare(std::string field, CompareOp op, Value v) {
        auto n = std::make_shared<Node>();
        n->kind = Node::Kind::Compare;
        n->field = std::move(field);
        n->op = op;
        n->a = checked(std::move(v));
        return Filter(std::move(n));
    }
    static Filter tags_node(Node::Kind kind, std::string field, std::vector<std::string> tags) {
        auto n = std::make_shared<Node>();
        n->kind = kind;
        n->field = std::move(field);
        n->tags = std::move(tags);
        return Filter(std::move(n));
    }
    static Filter combine(Node::Kind kind, const Filter& x, const Filter& y) {
        auto n = std::make_shared<Node>();
        n->kind = kind;
        n->depth = checked_depth(1 + std::max(x.node_->depth, y.node_->depth));
        n->children = {x.node_, y.node_};
        return Filter(std::move(n));
    }

    std::shared_ptr<const Node> node_;
};

/**
 * @brief A Filter checked against one index's schema and turned into a flat,
 *        fast form: field names become column numbers, strings become
 *        dictionary IDs. matches() reads a few columns and never allocates.
 *
 * Throws std::invalid_argument for a type mismatch, or for an unknown field in
 * strict mode (in dynamic mode an unknown field simply matches nothing).
 */
class CompiledFilter {
public:
    CompiledFilter(const Filter& filter, const MetadataStore& store);

    /// True if slot `slot` satisfies the filter.
    bool matches(NodeId slot) const { return eval(root_, slot); }

    /// Exact number of live matching vectors, when it can be read from the
    /// payload index or presence counts; otherwise nothing.
    std::optional<std::size_t> exact_count() const;

    /// True if the filter can never match (decided while compiling).
    bool always_false() const { return nodes_[static_cast<std::size_t>(root_)].kind == K::False; }

private:
    enum class K { True, False, Num, NumBetween, NumIn, Key, KeyIn, Bool, Tag, AnyTag, AllTags, Exists, And, Or, Not };
    struct CNode {
        K kind = K::True;
        std::size_t field = 0;
        bool field_is_int = false;
        CompareOp op = CompareOp::Eq;
        bool lit_is_int = false, lit2_is_int = false;
        std::int64_t i = 0, i2 = 0;
        double f = 0.0, f2 = 0.0;
        bool b = false;
        std::uint32_t id = 0;
        std::vector<Value> nums;
        std::vector<std::uint32_t> ids;
        std::vector<int> children;
    };

    int build(const Filter::Node& n);
    int add(CNode c) {
        nodes_.push_back(std::move(c));
        return static_cast<int>(nodes_.size()) - 1;
    }
    bool eval(int n, NodeId slot) const;

    const MetadataStore* store_;
    std::vector<CNode> nodes_;
    int root_ = 0;
};

/// How a filtered search is carried out.
enum class Strategy { Auto, ForceExact, ForceGraph };

/// Why the planner chose what it chose (reported in SearchStats).
enum class PlanReason { NoFilter, Forced, MatchCount, MatchFraction, Graph, FlatIndex };

/**
 * @brief Settings of the query planner.
 *
 * A filtered HNSW search uses exact search over matching vectors when the
 * expected number of matches is at most `max_exact_matches`, OR their fraction
 * is at most `max_exact_fraction`. Otherwise it searches the graph, with `ef`
 * raised by 1 / selectivity, up to `max_ef_multiplier` times.
 * On very large indexes, lower `max_exact_fraction`: 1% of 100 million vectors
 * is a million exact distance computations.
 */
struct PlannerParams {
    std::size_t max_exact_matches = 2000;
    double max_exact_fraction = 0.01;
    std::size_t sample_size = 256;
    double max_ef_multiplier = 32.0;
};

/// Throws std::invalid_argument for settings outside their valid range.
void validate(const PlannerParams& p);

/// What a search did, for tests, benchmarks and tuning.
struct SearchStats {
    Strategy strategy = Strategy::Auto;  // ForceExact or ForceGraph: what actually ran
    PlanReason reason = PlanReason::NoFilter;
    double selectivity = 1.0;            // estimated fraction of live vectors matching
    std::size_t estimated_matches = 0;
    bool exact_count = false;            // estimate came from the payload index
    std::size_t ef = 0;                  // beam width used by a graph search
    std::size_t nodes_visited = 0;
    std::size_t distance_computations = 0;
};

/**
 * @brief Optional settings for one search call.
 *
 *  - filter: a metadata condition; only matching vectors are returned.
 *  - predicate: any condition on the user ID, for data outside the index
 *    (permissions, for example). Combined with `filter`, both must pass; the
 *    filter is checked first. It may be called from several threads at once in
 *    batch search, so it must be safe to call concurrently.
 *  - strategy: Auto lets the planner choose; ForceExact / ForceGraph override it.
 *  - planner: settings for this query only (otherwise the index's settings).
 *  - stats: if set, filled with what the search did (ignored by batch search).
 *  - range_expand_outside: range search only; how many hops a search may take
 *    through vectors outside the radius to reach more of the region.
 */
struct SearchOptions {
    std::optional<Filter> filter;
    std::function<bool(std::uint64_t)> predicate;
    Strategy strategy = Strategy::Auto;
    std::optional<PlannerParams> planner;
    SearchStats* stats = nullptr;
    std::size_t range_expand_outside = 1;
};

/**
 * @brief "May this slot appear in results?": live, passes the compiled filter
 *        and passes the predicate (checked in that order, cheapest first).
 */
struct Eligibility {
    const IdMap* ids = nullptr;
    const CompiledFilter* filter = nullptr;
    const std::function<bool(std::uint64_t)>* predicate = nullptr;
    NodeId exclude = kEmpty;  // a node to leave out (the old version during an update)

    bool operator()(NodeId n) const {
        if (n == exclude || ids->is_deleted(n)) return false;
        if (filter && !filter->matches(n)) return false;
        if (predicate && !(*predicate)(ids->external(n))) return false;
        return true;
    }
    bool filtered() const { return filter != nullptr || predicate != nullptr; }
};

/// The planner's decision for one filtered search.
struct PlanDecision {
    Strategy strategy = Strategy::ForceGraph;
    PlanReason reason = PlanReason::Graph;
    double selectivity = 1.0;
    std::size_t estimated_matches = 0;
    bool exact_count = false;
    std::size_t ef = 0;
};

/**
 * @brief Chooses exact or graph search for a filtered query.
 *
 * Estimates selectivity exactly from the payload index when possible (no
 * predicate and a supported filter), otherwise by testing a deterministic
 * sample of live slots (every live slot if there are fewer than sample_size).
 *
 * @param capacity  number of slots, live or removed
 * @param live      number of live vectors
 */
PlanDecision plan_search(std::size_t capacity, std::size_t live, const Eligibility& eligible,
                         std::optional<std::size_t> exact_matches, std::size_t ef, Strategy forced,
                         const PlannerParams& params);

/// Checks a batch of queries stored back to back (`dim` floats each) before any
/// work: throws std::invalid_argument if the size is not a multiple of `dim`
/// or any value is NaN or infinite. Returns the number of queries.
inline std::size_t validate_batch(std::span<const float> queries, std::size_t dim) {
    if (queries.size() % dim != 0) throw std::invalid_argument("query buffer size is not a multiple of the dimension");
    for (std::size_t i = 0; i < queries.size(); ++i)
        if (!std::isfinite(queries[i]))
            throw std::invalid_argument("query " + std::to_string(i / dim) + " contains NaN or infinity");
    return queries.size() / dim;
}

}  // namespace vecdb