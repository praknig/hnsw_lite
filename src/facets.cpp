#include "facets.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace vecdb {

void FacetRequest::validate() const {
    if (field.empty()) throw std::invalid_argument("FacetRequest: field must not be empty");
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
        if (!std::isfinite(boundaries[i])) throw std::invalid_argument("FacetRequest: boundaries must be finite");
        if (i > 0 && !(boundaries[i] > boundaries[i - 1]))
            throw std::invalid_argument("FacetRequest: boundaries must be strictly ascending");
    }
}

FacetCounter::FacetCounter(const MetadataStore& store, const FacetRequest& request) : store_(store), request_(request) {
    request_.validate();
    field_ = store.find_field(request_.field);
    if (!field_) {
        if (store.mode() == SchemaMode::Strict)
            throw std::invalid_argument("facet: unknown field '" + request_.field + "' (strict schema)");
    } else {
        type_ = store.field_type(*field_);
    }
    if (!request_.boundaries.empty()) {
        if (field_ && type_ != FieldType::Int && type_ != FieldType::Float)
            throw std::invalid_argument("facet: range boundaries need an integer or float field, and '" + request_.field +
                                        "' is a " + field_type_name(type_) + " field");
        buckets_.assign(request_.boundaries.size() + 1, 0);
    }
}

void FacetCounter::add_number(double x) {
    if (buckets_.empty()) return;
    // The first boundary above x: x falls in the bucket just below it.
    const auto it = std::upper_bound(request_.boundaries.begin(), request_.boundaries.end(), x);
    ++buckets_[static_cast<std::size_t>(it - request_.boundaries.begin())];
}

void FacetCounter::add(NodeId slot) {
    ++total_;
    if (!field_ || !store_.has(*field_, slot)) {
        ++missing_;
        return;
    }
    const std::size_t f = *field_;
    const bool values = request_.boundaries.empty() || request_.distinct;  // buckets alone need no value counts
    switch (type_) {
        case FieldType::Int: {
            const std::int64_t v = store_.int_at(f, slot);
            add_number(static_cast<double>(v));
            if (values) ++by_int_[v];
            break;
        }
        case FieldType::Float: {
            double v = store_.float_at(f, slot);
            add_number(v);
            if (v == 0.0) v = 0.0;  // -0 and 0 are one value
            if (values) ++by_float_[std::bit_cast<std::uint64_t>(v)];
            break;
        }
        case FieldType::Bool: ++by_bool_[store_.bool_at(f, slot) ? 1 : 0]; break;
        case FieldType::Keyword: ++by_string_[store_.keyword_at(f, slot)]; break;
        case FieldType::Tags:
            for (const std::uint32_t id : store_.tags_at(f, slot)) ++by_string_[id];  // stored without repeats: once per document
            break;
    }
}

bool FacetCounter::add_all_from_index(std::size_t documents) {
    if (!field_) {  // an unknown field (dynamic schema): every document lacks it
        total_ += documents;
        missing_ += documents;
        return true;
    }
    const std::size_t f = *field_;
    // Only keyword, tag and boolean fields can be indexed, and only numeric ones take
    // boundaries: an indexed field is counted here whatever the request.
    if (!store_.is_indexed(f)) return false;
    total_ += documents;
    missing_ += documents - store_.present_count(f);
    if (type_ == FieldType::Bool) {
        for (std::uint32_t v = 0; v < 2; ++v) by_bool_[v] += store_.value_count(f, v).value_or(0);
    } else {
        for (std::uint32_t id = 0; id < store_.dictionary_size(); ++id)
            if (const std::size_t c = store_.value_count(f, id).value_or(0); c > 0) by_string_[id] += c;
    }
    return true;
}

FacetResult FacetCounter::result() const {
    FacetResult r;
    r.field = request_.field;
    r.total = total_;
    r.missing = missing_;
    if (!request_.boundaries.empty()) {
        const auto& b = request_.boundaries;
        for (std::size_t i = 0; i < buckets_.size(); ++i) {
            FacetBucket bucket;
            if (i > 0) bucket.from = b[i - 1];
            if (i < b.size()) bucket.to = b[i];
            bucket.count = buckets_[i];
            r.buckets.push_back(bucket);
        }
    }
    // Every value with its count, ordered by count (most first), then by value.
    std::vector<FacetValue> all;
    switch (type_) {
        case FieldType::Int:
            for (const auto& [v, c] : by_int_) all.push_back({Value::of_int(v), c});
            break;
        case FieldType::Float:
            for (const auto& [bits, c] : by_float_) all.push_back({Value::of_float(std::bit_cast<double>(bits)), c});
            break;
        case FieldType::Bool:
            for (std::uint32_t v = 0; v < 2; ++v)
                if (by_bool_[v]) all.push_back({Value::of_bool(v == 1), by_bool_[v]});
            break;
        case FieldType::Keyword:
        case FieldType::Tags:
            for (const auto& [id, c] : by_string_) all.push_back({Value::of_keyword(store_.string_at(id)), c});
            break;
    }
    if (request_.distinct) r.distinct = all.size();
    if (!request_.boundaries.empty()) return r;  // ranges replace the value list
    std::sort(all.begin(), all.end(), [](const FacetValue& a, const FacetValue& b) {
        if (a.count != b.count) return a.count > b.count;
        switch (a.value.type) {
            case FieldType::Int: return a.value.i < b.value.i;
            case FieldType::Float: return a.value.f < b.value.f;
            case FieldType::Bool: return a.value.b < b.value.b;
            default: return a.value.s < b.value.s;
        }
    });
    const std::size_t keep = request_.limit == 0 ? all.size() : std::min(request_.limit, all.size());
    for (std::size_t i = keep; i < all.size(); ++i) r.other += all[i].count;
    all.resize(keep);
    r.values = std::move(all);
    return r;
}

namespace {
std::vector<FacetCounter> counters_for(const MetadataStore& store, const std::vector<FacetRequest>& requests) {
    std::vector<FacetCounter> counters;
    counters.reserve(requests.size());
    for (const FacetRequest& r : requests) counters.emplace_back(store, r);  // throws before counting anything
    return counters;
}
std::vector<FacetResult> results_of(const std::vector<FacetCounter>& counters) {
    std::vector<FacetResult> out;
    out.reserve(counters.size());
    for (const FacetCounter& c : counters) out.push_back(c.result());
    return out;
}
}  // namespace

std::vector<FacetResult> count_facets(const MetadataStore& store, const std::vector<FacetRequest>& requests,
                                      const std::vector<NodeId>& slots) {
    std::vector<FacetCounter> counters = counters_for(store, requests);
    for (const NodeId slot : slots)
        for (FacetCounter& c : counters) c.add(slot);
    return results_of(counters);
}

std::vector<FacetResult> count_facets(const MetadataStore& store, const std::vector<FacetRequest>& requests, std::size_t live,
                                      const std::function<std::vector<NodeId>()>& all_slots) {
    std::vector<FacetCounter> counters = counters_for(store, requests);
    std::vector<FacetCounter*> read;
    for (FacetCounter& c : counters)
        if (!c.add_all_from_index(live)) read.push_back(&c);
    if (!read.empty()) {
        const std::vector<NodeId> slots = all_slots();
        for (const NodeId slot : slots)
            for (FacetCounter* c : read) c->add(slot);
    }
    return results_of(counters);
}

}  // namespace vecdb
