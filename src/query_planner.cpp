/**
 * @file query_planner.cpp
 * @brief Compiled filters and the query planner (see filter.h for the design).
 */
#include <algorithm>
#include <cmath>
#include <random>

#include "filter.h"

namespace vecdb {
namespace {

constexpr std::uint32_t kNoString = std::numeric_limits<std::uint32_t>::max();

/// Exact three-way comparison of an integer with a double (no precision loss,
/// even beyond 2^53): -1 if a < b, 0 if equal, 1 if a > b.
int compare_int_double(std::int64_t a, double b) {
    if (b >= 9223372036854775808.0) return -1;   // b >= 2^63, beyond every int64
    if (b < -9223372036854775808.0) return 1;    // b < -2^63
    const double t = std::trunc(b);
    const auto bi = static_cast<std::int64_t>(t);
    if (a < bi) return -1;
    if (a > bi) return 1;
    const double frac = b - t;
    return frac > 0 ? -1 : (frac < 0 ? 1 : 0);
}

/// Three-way comparison of two numbers, each an integer or a double.
int compare_num(bool a_int, std::int64_t ai, double ad, bool b_int, std::int64_t bi, double bd) {
    if (a_int && b_int) return (ai > bi) - (ai < bi);
    if (!a_int && !b_int) return (ad > bd) - (ad < bd);
    if (a_int) return compare_int_double(ai, bd);
    return -compare_int_double(bi, ad);
}

bool apply_op(CompareOp op, int c) {
    switch (op) {
        case CompareOp::Eq: return c == 0;
        case CompareOp::Ne: return c != 0;
        case CompareOp::Lt: return c < 0;
        case CompareOp::Le: return c <= 0;
        case CompareOp::Gt: return c > 0;
        default: return c >= 0;
    }
}

bool is_numeric(FieldType t) { return t == FieldType::Int || t == FieldType::Float; }

[[noreturn]] void type_error(const std::string& field, const std::string& what) {
    throw std::invalid_argument("filter on field '" + field + "': " + what);
}

}  // namespace

CompiledFilter::CompiledFilter(const Filter& filter, const MetadataStore& store) : store_(&store) {
    root_ = build(filter.root());
}

int CompiledFilter::build(const Filter::Node& n) {
    using NK = Filter::Node::Kind;
    CNode c;
    if (n.kind == NK::All) return add(c);  // True

    if (n.kind == NK::And || n.kind == NK::Or || n.kind == NK::Not) {
        std::vector<int> kids;
        for (const auto& child : n.children) kids.push_back(build(*child));
        auto kind_of = [&](int k) { return nodes_[static_cast<std::size_t>(k)].kind; };
        if (n.kind == NK::Not) {  // fold constants
            if (kind_of(kids[0]) == K::True) { c.kind = K::False; return add(c); }
            if (kind_of(kids[0]) == K::False) { c.kind = K::True; return add(c); }
            c.kind = K::Not;
        } else if (n.kind == NK::And) {
            if (kind_of(kids[0]) == K::False || kind_of(kids[1]) == K::False) { c.kind = K::False; return add(c); }
            c.kind = K::And;
        } else {
            if (kind_of(kids[0]) == K::False && kind_of(kids[1]) == K::False) { c.kind = K::False; return add(c); }
            c.kind = K::Or;
        }
        c.children = std::move(kids);
        return add(std::move(c));
    }

    // A condition on one field.
    const auto f = store_->find_field(n.field);
    if (!f) {
        if (store_->mode() == SchemaMode::Strict) type_error(n.field, "unknown field");
        c.kind = K::False;  // dynamic schema: the field may not exist yet
        return add(c);
    }
    const FieldType ft = store_->field_type(*f);
    c.field = *f;
    c.field_is_int = ft == FieldType::Int;

    switch (n.kind) {
        case NK::Exists:
            c.kind = K::Exists;
            return add(c);

        case NK::Compare: {
            const Value& v = n.a;
            if (is_numeric(ft)) {
                if (!is_numeric(v.type)) type_error(n.field, "numeric field compared with a non-number");
                c.kind = K::Num;
                c.op = n.op;
                c.lit_is_int = v.type == FieldType::Int;
                c.i = v.i;
                c.f = v.f;
                return add(c);
            }
            if (n.op != CompareOp::Eq && n.op != CompareOp::Ne)
                type_error(n.field, std::string(field_type_name(ft)) + " fields support only eq and ne");
            if (ft == FieldType::Keyword) {
                if (v.type != FieldType::Keyword) type_error(n.field, "keyword field compared with a non-string");
                c.kind = K::Key;
                c.op = n.op;
                c.id = store_->find_string(v.s).value_or(kNoString);  // unknown: never equal
                return add(c);
            }
            if (ft == FieldType::Bool) {
                if (v.type != FieldType::Bool) type_error(n.field, "boolean field compared with a non-boolean");
                c.kind = K::Bool;
                c.op = n.op;
                c.b = v.b;
                return add(c);
            }
            type_error(n.field, "tag sets support has_tag, has_any_tag and has_all_tags");
        }

        case NK::Between: {
            if (!is_numeric(ft) || !is_numeric(n.a.type) || !is_numeric(n.b.type))
                type_error(n.field, "between needs a numeric field and numeric bounds");
            const bool lo_int = n.a.type == FieldType::Int, hi_int = n.b.type == FieldType::Int;
            if (compare_num(lo_int, n.a.i, n.a.f, hi_int, n.b.i, n.b.f) > 0) {
                c.kind = K::False;  // lo > hi: empty range
                return add(c);
            }
            c.kind = K::NumBetween;
            c.lit_is_int = lo_int;
            c.i = n.a.i;
            c.f = n.a.f;
            c.lit2_is_int = hi_int;
            c.i2 = n.b.i;
            c.f2 = n.b.f;
            return add(c);
        }

        case NK::In: {
            if (n.values.empty()) {
                c.kind = K::False;
                return add(c);
            }
            if (is_numeric(ft)) {
                for (const Value& v : n.values)
                    if (!is_numeric(v.type)) type_error(n.field, "numeric field compared with a non-number");
                c.kind = K::NumIn;
                c.nums = n.values;
                return add(std::move(c));
            }
            if (ft == FieldType::Keyword) {
                for (const Value& v : n.values) {
                    if (v.type != FieldType::Keyword) type_error(n.field, "keyword field compared with a non-string");
                    if (auto id = store_->find_string(v.s))
                        if (std::find(c.ids.begin(), c.ids.end(), *id) == c.ids.end()) c.ids.push_back(*id);
                }
                std::sort(c.ids.begin(), c.ids.end());  // binary search per vector
                c.kind = c.ids.empty() ? K::False : K::KeyIn;
                return add(std::move(c));
            }
            if (ft == FieldType::Bool) {
                bool has_true = false, has_false = false;
                for (const Value& v : n.values) {
                    if (v.type != FieldType::Bool) type_error(n.field, "boolean field compared with a non-boolean");
                    (v.b ? has_true : has_false) = true;
                }
                if (has_true && has_false) {
                    c.kind = K::Exists;
                } else {
                    c.kind = K::Bool;
                    c.op = CompareOp::Eq;
                    c.b = has_true;
                }
                return add(c);
            }
            type_error(n.field, "use has_any_tag for tag sets");
        }

        case NK::HasTag:
        case NK::HasAnyTag:
        case NK::HasAllTags: {
            if (ft != FieldType::Tags) type_error(n.field, "tag conditions need a tag-set field");
            if (n.kind == NK::HasAllTags && n.tags.empty()) {
                c.kind = K::Exists;  // vacuously true for every vector that has the field
                return add(c);
            }
            bool unknown = false;
            for (const std::string& t : n.tags) {
                if (auto id = store_->find_string(t)) {
                    if (std::find(c.ids.begin(), c.ids.end(), *id) == c.ids.end()) c.ids.push_back(*id);
                } else {
                    unknown = true;
                }
            }
            if (n.kind == NK::HasAnyTag) c.kind = c.ids.empty() ? K::False : K::AnyTag;
            else if (unknown || c.ids.empty()) c.kind = K::False;  // a needed tag no vector has
            else c.kind = n.kind == NK::HasTag ? K::Tag : K::AllTags;
            return add(std::move(c));
        }

        // GCOVR_EXCL_START: unreachable, every node kind is handled above
        default:
            c.kind = K::True;
            return add(c);
            // GCOVR_EXCL_STOP
    }
}

bool CompiledFilter::eval(int index, NodeId slot) const {
    const CNode& c = nodes_[static_cast<std::size_t>(index)];
    const MetadataStore& s = *store_;
    switch (c.kind) {
        case K::True: return true;
        case K::False: return false;
        case K::And: return eval(c.children[0], slot) && eval(c.children[1], slot);
        case K::Or: return eval(c.children[0], slot) || eval(c.children[1], slot);
        case K::Not: return !eval(c.children[0], slot);
        default: break;
    }
    if (!s.has(c.field, slot)) return false;  // comparisons never match a missing field
    switch (c.kind) {
        case K::Exists: return true;
        case K::Num: {
            const std::int64_t fi = c.field_is_int ? s.int_at(c.field, slot) : 0;
            const double fd = c.field_is_int ? 0.0 : s.float_at(c.field, slot);
            return apply_op(c.op, compare_num(c.field_is_int, fi, fd, c.lit_is_int, c.i, c.f));
        }
        case K::NumBetween: {
            const std::int64_t fi = c.field_is_int ? s.int_at(c.field, slot) : 0;
            const double fd = c.field_is_int ? 0.0 : s.float_at(c.field, slot);
            return compare_num(c.field_is_int, fi, fd, c.lit_is_int, c.i, c.f) >= 0 &&
                   compare_num(c.field_is_int, fi, fd, c.lit2_is_int, c.i2, c.f2) <= 0;
        }
        case K::NumIn: {
            const std::int64_t fi = c.field_is_int ? s.int_at(c.field, slot) : 0;
            const double fd = c.field_is_int ? 0.0 : s.float_at(c.field, slot);
            for (const Value& v : c.nums)
                if (compare_num(c.field_is_int, fi, fd, v.type == FieldType::Int, v.i, v.f) == 0) return true;
            return false;
        }
        case K::Key: {
            const bool equal = s.keyword_at(c.field, slot) == c.id;
            return c.op == CompareOp::Eq ? equal : !equal;
        }
        case K::KeyIn: return std::binary_search(c.ids.begin(), c.ids.end(), s.keyword_at(c.field, slot));
        case K::Bool: {
            const bool equal = s.bool_at(c.field, slot) == c.b;
            return c.op == CompareOp::Eq ? equal : !equal;
        }
        case K::Tag:
        case K::AnyTag: {
            const auto& tags = s.tags_at(c.field, slot);
            for (std::uint32_t id : c.ids)
                if (std::find(tags.begin(), tags.end(), id) != tags.end()) return true;
            return false;
        }
        case K::AllTags: {
            const auto& tags = s.tags_at(c.field, slot);
            for (std::uint32_t id : c.ids)
                if (std::find(tags.begin(), tags.end(), id) == tags.end()) return false;
            return true;
        }
        default: return false;  // GCOVR_EXCL_LINE: unreachable, every kind is handled above
    }
}

std::optional<std::size_t> CompiledFilter::exact_count() const {
    const CNode& c = nodes_[static_cast<std::size_t>(root_)];
    const MetadataStore& s = *store_;
    switch (c.kind) {
        case K::False: return 0;
        case K::Exists: return s.present_count(c.field);
        case K::Key:
            if (c.op != CompareOp::Eq) return std::nullopt;
            return s.value_count(c.field, c.id);
        case K::KeyIn: {
            std::size_t total = 0;
            for (std::uint32_t id : c.ids) {
                auto n = s.value_count(c.field, id);
                if (!n) return std::nullopt;
                total += *n;
            }
            return total;
        }
        case K::Bool: return s.value_count(c.field, (c.op == CompareOp::Eq) == c.b ? 1u : 0u);
        case K::Tag: return s.value_count(c.field, c.ids[0]);
        default: return std::nullopt;
    }
}

void validate(const PlannerParams& p) {
    if (!(p.max_exact_fraction >= 0.0 && p.max_exact_fraction <= 1.0))
        throw std::invalid_argument("PlannerParams::max_exact_fraction must be in [0, 1]");
    if (p.sample_size < 1) throw std::invalid_argument("PlannerParams::sample_size must be at least 1");
    if (!(p.max_ef_multiplier >= 1.0) || !std::isfinite(p.max_ef_multiplier))
        throw std::invalid_argument("PlannerParams::max_ef_multiplier must be a finite number >= 1");
}

PlanDecision plan_search(std::size_t capacity, std::size_t live, const Eligibility& eligible,
                         std::optional<std::size_t> exact_matches, std::size_t ef, Strategy forced,
                         const PlannerParams& params) {
    PlanDecision d;
    if (live == 0) {
        d.strategy = Strategy::ForceExact;
        d.reason = PlanReason::MatchCount;
        d.selectivity = 0.0;
        d.exact_count = true;
        return d;
    }

    // 1. Estimate how many live vectors match.
    if (exact_matches) {
        d.estimated_matches = *exact_matches;
        d.exact_count = true;
    } else if (live <= params.sample_size) {  // small: test every live slot
        std::size_t hits = 0;
        for (NodeId n = 0; n < capacity; ++n) hits += eligible(n);
        d.estimated_matches = hits;
        d.exact_count = true;
    } else {  // deterministic sample, so the same query always plans the same way
        std::mt19937_64 rng(0x5EEDu);
        std::size_t samples = 0, hits = 0;
        for (std::size_t attempts = 0; samples < params.sample_size && attempts < params.sample_size * 64; ++attempts) {
            const auto n = static_cast<NodeId>(rng() % capacity);
            if (eligible.ids->is_deleted(n)) continue;
            ++samples;
            hits += eligible(n);
        }
        const double fraction = samples ? double(hits) / double(samples) : 0.0;
        d.estimated_matches = static_cast<std::size_t>(std::llround(fraction * double(live)));
    }
    d.selectivity = std::min(1.0, double(d.estimated_matches) / double(live));

    // 2. Choose a strategy.
    if (forced != Strategy::Auto) {
        d.strategy = forced;
        d.reason = PlanReason::Forced;
    } else if (d.estimated_matches <= params.max_exact_matches) {
        d.strategy = Strategy::ForceExact;
        d.reason = PlanReason::MatchCount;
    } else if (d.selectivity <= params.max_exact_fraction) {
        d.strategy = Strategy::ForceExact;
        d.reason = PlanReason::MatchFraction;
    } else {
        d.strategy = Strategy::ForceGraph;
        d.reason = PlanReason::Graph;
    }

    // 3. A graph search visits about 1 / selectivity nodes per useful one: widen the beam.
    const double cap = double(ef) * params.max_ef_multiplier;
    const double wanted = d.selectivity > 0.0 ? std::ceil(double(ef) / d.selectivity) : cap;
    d.ef = std::min(capacity, std::max(ef, static_cast<std::size_t>(std::min(wanted, cap))));
    return d;
}

}  // namespace vecdb