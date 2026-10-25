/**
 * @file bindings.cpp
 * @brief The Python extension module hnsw_lite_lib._core (pybind11).
 *
 * Conventions:
 *  - Vectors are NumPy arrays (any float dtype and layout: converted to
 *    contiguous float32) or anything NumPy can convert, such as lists.
 *  - IDs are unsigned 64-bit integers; batch results come back as NumPy arrays
 *    of shape (n, k), padded with ID 2**64 - 1 and distance +inf where fewer
 *    than k results exist.
 *  - Metadata is a dict: int, float, bool, str, or a list/tuple/set of str
 *    (tags); None clears a field in set_metadata.
 *  - Sparse vectors are dicts {term: weight} or pairs (indices, values).
 *  - Python objects are converted first; the library then runs without the GIL,
 *    so other Python threads can search at the same time.
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <limits>

#include "collection.h"
#include "flat_index.h"
#include "hnsw_index.h"
#include "serialize.h"
#include "sparse.h"

namespace py = pybind11;
using namespace vecdb;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
constexpr std::uint64_t kMissing = std::numeric_limits<std::uint64_t>::max();

// ---------------------------------------------------------------- conversions

/// A 1-D vector from any array-like.
std::vector<float> to_vector(const py::handle& obj, const char* what) {
    FloatArray a = FloatArray::ensure(obj);
    if (!a) throw py::type_error(std::string(what) + ": expected a sequence of numbers");
    if (a.ndim() != 1) throw py::value_error(std::string(what) + ": expected a 1-D array, got " + std::to_string(a.ndim()) + " dimensions");
    return std::vector<float>(a.data(), a.data() + a.size());
}

/// A 2-D matrix (n x dim) from any array-like; a 1-D one is one row.
FloatArray to_matrix(const py::handle& obj, const char* what) {
    FloatArray a = FloatArray::ensure(obj);
    if (!a) throw py::type_error(std::string(what) + ": expected a 2-D array of numbers");
    if (a.ndim() == 1) a = a.reshape({py::ssize_t(1), a.shape(0)});
    if (a.ndim() != 2) throw py::value_error(std::string(what) + ": expected a 2-D array (n, dim)");
    return a;
}

/// IDs: non-negative integers. Checked before any conversion: NumPy's cast to an
/// unsigned type would wrap -3 around to 2**64 - 3 without a word.
std::vector<std::uint64_t> to_ids(const py::handle& obj) {
    if (py::isinstance<py::bool_>(obj)) throw py::type_error("ids: expected integers, got a bool");
    if (py::isinstance<py::int_>(obj)) {
        if (obj.cast<py::int_>() < py::int_(0)) throw py::value_error("ids must be non-negative");
        return {obj.cast<std::uint64_t>()};
    }
    py::array a = py::array::ensure(obj);
    if (!a) throw py::type_error("ids: expected integers");
    const char kind = a.dtype().kind();
    if (a.size() > 0 && kind != 'i' && kind != 'u') throw py::type_error("ids: expected integers");
    if (a.ndim() != 1) throw py::value_error("ids: expected a 1-D sequence");
    if (kind == 'i') {
        const auto s = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(a);
        for (py::ssize_t i = 0; i < s.size(); ++i)
            if (s.data()[i] < 0) throw py::value_error("ids must be non-negative");
    }
    const auto u = py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>::ensure(a);
    return std::vector<std::uint64_t>(u.data(), u.data() + u.size());
}

bool is_tags(const py::handle& v) { return py::isinstance<py::list>(v) || py::isinstance<py::tuple>(v) || py::isinstance<py::set>(v) || py::isinstance<py::frozenset>(v); }

Metadata to_metadata(const py::handle& obj) {
    Metadata m;
    if (obj.is_none()) return m;
    if (!py::isinstance<py::dict>(obj)) throw py::type_error("metadata: expected a dict");
    for (auto item : py::reinterpret_borrow<py::dict>(obj)) {
        const std::string name = py::str(item.first);
        const py::handle v = item.second;
        if (v.is_none()) m.unset(name);
        else if (py::isinstance<py::bool_>(v)) m.set(name, v.cast<bool>());  // before int: bool is an int in Python
        else if (py::isinstance<py::int_>(v)) m.set(name, v.cast<std::int64_t>());
        else if (py::isinstance<py::float_>(v)) m.set(name, v.cast<double>());
        else if (py::isinstance<py::str>(v)) m.set(name, v.cast<std::string>());
        else if (is_tags(v)) {
            std::vector<std::string> tags;
            for (auto t : v) {
                if (!py::isinstance<py::str>(t)) throw py::type_error("metadata '" + name + "': tags must be strings");
                tags.push_back(t.cast<std::string>());
            }
            m.set_tags(name, std::move(tags));
        } else if (py::hasattr(v, "item")) {  // NumPy scalars: their Python equivalent
            Metadata one = to_metadata(py::dict(py::arg(name.c_str()) = v.attr("item")()));
            m.set(name, *one.get(name));
        } else {
            throw py::type_error("metadata '" + name + "': expected int, float, bool, str or a list of str, got " +
                                 std::string(py::str(py::type::of(v).attr("__name__"))));
        }
    }
    return m;
}

py::object value_to_py(const Value& v) {
    switch (v.type) {
        case FieldType::Int: return py::int_(v.i);
        case FieldType::Float: return py::float_(v.f);
        case FieldType::Bool: return py::bool_(v.b);
        case FieldType::Keyword: return py::str(v.s);
        case FieldType::Tags: return py::cast(v.tags);
    }
    return py::none();  // GCOVR_EXCL_LINE
}

py::dict from_metadata(const Metadata& m) {
    py::dict d;
    for (const auto& [name, value] : m.fields())
        if (value) d[py::str(name)] = value_to_py(*value);
    return d;
}

/// A sparse term: an integer in [0, 2**32), checked before conversion (a cast
/// would wrap -1 around to 4294967295 without a word).
std::uint32_t to_term(const py::handle& t) {
    py::object x = py::reinterpret_borrow<py::object>(t);
    if (!py::isinstance<py::int_>(x) && py::hasattr(x, "item")) x = x.attr("item")();  // NumPy integers
    if (!py::isinstance<py::int_>(x) || py::isinstance<py::bool_>(x)) throw py::type_error("sparse: terms must be integers");
    if (x.cast<py::int_>() < py::int_(0) || x.cast<py::int_>() > py::int_(0xFFFFFFFFull))
        throw py::value_error("sparse: terms must be in [0, 2**32)");
    return x.cast<std::uint32_t>();
}

SparseVector to_sparse(const py::handle& obj) {
    SparseVector v;
    if (obj.is_none()) return v;
    if (py::isinstance<py::dict>(obj)) {
        for (auto item : py::reinterpret_borrow<py::dict>(obj)) {
            v.indices.push_back(to_term(item.first));
            v.values.push_back(item.second.cast<float>());
        }
        return v;
    }
    if (py::isinstance<py::tuple>(obj) && py::len(obj) == 2) {
        const auto t = py::reinterpret_borrow<py::tuple>(obj);
        py::array idx = py::array::ensure(t[0]);
        if (!idx || idx.ndim() != 1) throw py::type_error("sparse: indices must be a 1-D sequence of integers");
        const char kind = idx.dtype().kind();
        if (idx.size() > 0 && kind != 'i' && kind != 'u') throw py::type_error("sparse: terms must be integers");
        auto add = [&](auto term, bool negative) {
            if (negative || term > 0xFFFFFFFFu) throw py::value_error("sparse: terms must be in [0, 2**32)");
            v.indices.push_back(static_cast<std::uint32_t>(term));
        };
        if (kind == 'u') {
            const auto u = py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>::ensure(idx);
            for (py::ssize_t i = 0; i < u.size(); ++i) add(u.data()[i], false);
        } else {
            const auto w = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(idx);
            for (py::ssize_t i = 0; i < w.size(); ++i) add(static_cast<std::uint64_t>(w.data()[i]), w.data()[i] < 0);
        }
        v.values = to_vector(t[1], "sparse values");
        return v;
    }
    throw py::type_error("sparse: expected a dict {term: weight} or a pair (indices, values)");
}

py::dict from_sparse(const SparseVector& v) {
    py::dict d;
    for (std::size_t i = 0; i < v.size(); ++i) d[py::int_(v.indices[i])] = py::float_(v.values[i]);
    return d;
}

Schema to_schema(const py::handle& fields, bool strict) {
    std::vector<FieldSpec> specs;
    if (!fields.is_none())
        for (auto item : py::reinterpret_borrow<py::dict>(fields)) specs.push_back({py::str(item.first), item.second.cast<FieldType>()});
    return strict ? Schema::strict(std::move(specs)) : Schema::dynamic(std::move(specs));
}

// A filter value of the right C++ type.
template <class Make>
Filter with_value(const py::handle& v, Make&& make) {
    if (py::isinstance<py::bool_>(v)) return make(v.cast<bool>());
    if (py::isinstance<py::int_>(v)) return make(v.cast<std::int64_t>());
    if (py::isinstance<py::float_>(v)) return make(v.cast<double>());
    if (py::isinstance<py::str>(v)) return make(v.cast<std::string>());
    if (py::hasattr(v, "item")) return with_value(v.attr("item")(), make);  // NumPy scalars
    throw py::type_error("filter value: expected int, float, bool or str");
}

Filter filter_in(const std::string& field, const py::sequence& raw) {
    // NumPy scalars (np.bool_, np.int64, ...) as the Python values they stand for.
    py::list values;
    for (auto v : raw) {
        const bool plain = py::isinstance<py::bool_>(v) || py::isinstance<py::int_>(v) || py::isinstance<py::float_>(v) || py::isinstance<py::str>(v);
        values.append(!plain && py::hasattr(v, "item") ? v.attr("item")() : py::reinterpret_borrow<py::object>(v));
    }
    bool all_str = true, all_int = true, all_bool = true;
    for (auto v : values) {
        all_str = all_str && py::isinstance<py::str>(v);
        all_bool = all_bool && py::isinstance<py::bool_>(v);
        all_int = all_int && py::isinstance<py::int_>(v) && !py::isinstance<py::bool_>(v);
    }
    if (py::len(values) == 0) return Filter::in(field, std::vector<std::string>{});  // matches nothing
    if (all_str) return Filter::in(field, values.cast<std::vector<std::string>>());
    if (all_int) return Filter::in(field, values.cast<std::vector<std::int64_t>>());
    if (all_bool) {
        std::optional<Filter> f;
        for (auto v : values) f = f ? (*f || Filter::eq(field, v.cast<bool>())) : Filter::eq(field, v.cast<bool>());
        return *f;
    }
    return Filter::in(field, values.cast<std::vector<double>>());  // numbers (ints and floats mixed)
}

/// (ids, distances) as NumPy arrays from one result list.
py::tuple results_1d(const std::vector<SearchResult>& r) {
    py::array_t<std::uint64_t> ids(static_cast<py::ssize_t>(r.size()));
    py::array_t<float> dist(static_cast<py::ssize_t>(r.size()));
    auto i = ids.mutable_unchecked<1>();
    auto d = dist.mutable_unchecked<1>();
    for (std::size_t j = 0; j < r.size(); ++j) {
        i(static_cast<py::ssize_t>(j)) = r[j].id;
        d(static_cast<py::ssize_t>(j)) = r[j].distance;
    }
    return py::make_tuple(ids, dist);
}

/// (ids, distances) of shape (n, k), padded with 2**64 - 1 and +inf.
py::tuple results_2d(const std::vector<std::vector<SearchResult>>& r, std::size_t k) {
    const auto n = static_cast<py::ssize_t>(r.size()), kk = static_cast<py::ssize_t>(k);
    py::array_t<std::uint64_t> ids({n, kk});
    py::array_t<float> dist({n, kk});
    auto i = ids.mutable_unchecked<2>();
    auto d = dist.mutable_unchecked<2>();
    for (py::ssize_t q = 0; q < n; ++q)
        for (py::ssize_t j = 0; j < kk; ++j) {
            const bool have = static_cast<std::size_t>(j) < r[static_cast<std::size_t>(q)].size();
            i(q, j) = have ? r[static_cast<std::size_t>(q)][static_cast<std::size_t>(j)].id : kMissing;
            d(q, j) = have ? r[static_cast<std::size_t>(q)][static_cast<std::size_t>(j)].distance : std::numeric_limits<float>::infinity();
        }
    return py::make_tuple(ids, dist);
}

SearchOptions search_options(const std::optional<Filter>& filter, Strategy strategy) {
    SearchOptions o;
    o.filter = filter;
    o.strategy = strategy;
    return o;
}

/// ScoreModifiers from the canonical dict the Python layer builds (hnsw_lite_lib._modifiers).
std::optional<ScoreModifiers> to_modifiers(const py::handle& obj) {
    if (obj.is_none()) return std::nullopt;
    const auto d = py::reinterpret_borrow<py::dict>(obj);
    ScoreModifiers m;
    if (d.contains("decays"))
        for (auto x : d["decays"]) {
            const auto e = py::reinterpret_borrow<py::dict>(x);
            RecencyDecay r;
            r.field = e["field"].cast<std::string>();
            r.origin = e["origin"].cast<double>();
            r.half_life = e["half_life"].cast<double>();
            r.shape = e["shape"].cast<std::string>() == "linear" ? DecayShape::Linear : DecayShape::Exponential;
            r.missing = e["missing"].cast<double>();
            r.floor = e["floor"].cast<double>();
            m.decays.push_back(r);
        }
    if (d.contains("multipliers"))
        for (auto x : d["multipliers"]) {
            const auto e = py::reinterpret_borrow<py::dict>(x);
            m.multipliers.push_back({e["field"].cast<std::string>(), e["missing"].cast<double>(), e["min"].cast<double>(), e["max"].cast<double>()});
        }
    if (d.contains("boosts"))
        for (auto x : d["boosts"]) {
            const auto e = py::reinterpret_borrow<py::dict>(x);
            m.boosts.push_back({e["when"].cast<Filter>(), e["bonus"].cast<double>()});
        }
    if (d.contains("pool")) m.pool = d["pool"].cast<std::size_t>();
    return m;
}

/// (ids, distances, scores) as NumPy arrays: a search with modifiers.
py::tuple scored_1d(const std::vector<SearchResult>& r) {
    py::tuple t = results_1d(r);
    py::array_t<float> scores(static_cast<py::ssize_t>(r.size()));
    auto s = scores.mutable_unchecked<1>();
    for (std::size_t j = 0; j < r.size(); ++j) s(static_cast<py::ssize_t>(j)) = r[j].score;
    return py::make_tuple(t[0], t[1], scores);
}

py::dict health_dict(const HealthReport& h) {
    py::dict d;
    d["live"] = h.live;
    d["slots"] = h.slots;
    d["free_slots"] = h.free_slots;
    d["mean_degree"] = h.mean_degree;
    d["min_degree"] = h.min_degree;
    d["underfull"] = h.underfull;
    d["no_incoming"] = h.no_incoming;
    d["unreachable"] = h.unreachable;
    d["links_to_removed"] = h.links_to_removed;
    d["stale_links"] = h.stale_links;
    d["recall"] = h.recall ? py::object(py::float_(*h.recall)) : py::none();
    d["codec_error"] = h.codec_error ? py::object(py::float_(*h.codec_error)) : py::none();
    d["codec_error_retrained"] = h.codec_error_retrained ? py::object(py::float_(*h.codec_error_retrained)) : py::none();
    d["removals"] = h.removals;
    d["updates"] = h.updates;
    d["churn"] = h.churn();
    return d;
}

py::dict maintenance_dict(const MaintenanceReport& r) {
    py::dict d;
    d["elapsed_us"] = r.elapsed.count();
    d["health"] = r.health ? py::object(health_dict(*r.health)) : py::none();
    d["upkept"] = r.upkept;
    d["rebuilt"] = r.rebuilt;
    d["checkpointed"] = r.checkpointed;
    d["checkpoint_lsn"] = r.checkpoint_lsn;
    d["budget_exhausted"] = r.budget_exhausted;
    d["notes"] = r.notes;
    d["error"] = r.error;
    return d;
}

py::object opt(const std::optional<std::size_t>& v) { return v ? py::object(py::int_(*v)) : py::none(); }
py::object optf(const std::optional<float>& v) { return v ? py::object(py::float_(*v)) : py::none(); }

/// The index methods HnswIndex and FlatIndex share.
template <class Index, class Class>
void bind_records(Class& c) {
    c.def("__len__", &Index::size)
        .def("__contains__", &Index::contains, py::arg("id"))
        .def_property_readonly("dim", &Index::dim)
        .def_property_readonly("metric", &Index::metric)
        .def("_add", [](Index& x, const py::handle& ids, const py::handle& vectors, const py::handle& metadata, std::size_t threads) {
            const std::vector<std::uint64_t> id = to_ids(ids);
            const FloatArray m = to_matrix(vectors, "vectors");
            if (static_cast<std::size_t>(m.shape(0)) != id.size())
                throw py::value_error("add: " + std::to_string(id.size()) + " ids for " + std::to_string(m.shape(0)) + " vectors");
            if (static_cast<std::size_t>(m.shape(1)) != x.dim())
                throw py::value_error("add: vectors have " + std::to_string(m.shape(1)) + " dimensions, the index " + std::to_string(x.dim()));
            const std::span<const float> flat(m.data(), static_cast<std::size_t>(m.size()));
            if (metadata.is_none()) {
                py::gil_scoped_release free;
                x.add_batch(id, flat, threads);
            } else {
                std::vector<Metadata> md;
                for (auto item : metadata) md.push_back(to_metadata(item));
                if (md.size() != id.size()) throw py::value_error("add: metadata must have one entry per id");
                py::gil_scoped_release free;
                x.add_batch(id, flat, md, threads);
            }
        }, py::arg("ids"), py::arg("vectors"), py::arg("metadata") = py::none(), py::arg("threads") = 0)
        .def("upsert", [](Index& x, std::uint64_t id, const py::handle& vector, const py::handle& metadata) {
            const std::vector<float> v = to_vector(vector, "vector");
            if (metadata.is_none()) {
                py::gil_scoped_release free;
                return x.upsert(id, v);
            }
            const Metadata m = to_metadata(metadata);
            py::gil_scoped_release free;
            return x.upsert(id, v, m);
        }, py::arg("id"), py::arg("vector"), py::arg("metadata") = py::none(), "Adds or replaces (keeping metadata unless given); True if added.")
        .def("update", [](Index& x, std::uint64_t id, const py::handle& vector, const py::handle& metadata) {
            const std::vector<float> v = to_vector(vector, "vector");
            if (metadata.is_none()) {
                py::gil_scoped_release free;
                return x.update(id, v);
            }
            const Metadata m = to_metadata(metadata);
            py::gil_scoped_release free;
            return x.update(id, v, m);
        }, py::arg("id"), py::arg("vector"), py::arg("metadata") = py::none())
        .def("remove", [](Index& x, const py::handle& ids) {
            const std::vector<std::uint64_t> id = to_ids(ids);
            py::gil_scoped_release free;
            return x.remove_batch(id);
        }, py::arg("ids"), "Removes one ID or many; returns how many were stored.")
        .def("remove_where", [](Index& x, const Filter& f) { py::gil_scoped_release free; return x.remove_where(f); }, py::arg("filter"))
        .def("count_where", [](const Index& x, const Filter& f) { py::gil_scoped_release free; return x.count_where(f); }, py::arg("filter"))
        .def("set_metadata", [](Index& x, std::uint64_t id, const py::handle& md) { const Metadata m = to_metadata(md); return x.set_metadata(id, m); },
             py::arg("id"), py::arg("metadata"))
        .def("set_metadata_where", [](Index& x, const Filter& f, const py::handle& md) { const Metadata m = to_metadata(md); return x.set_metadata_where(f, m); },
             py::arg("filter"), py::arg("changes"))
        .def("get_vector", [](const Index& x, std::uint64_t id) -> py::object {
            const auto v = x.get_vector(id);
            if (!v) return py::none();
            return py::array_t<float>(static_cast<py::ssize_t>(v->size()), v->data());
        }, py::arg("id"))
        .def("get_metadata", [](const Index& x, std::uint64_t id) -> py::object {
            const auto m = x.get_metadata(id);
            return m ? py::object(from_metadata(*m)) : py::none();
        }, py::arg("id"))
        .def("create_payload_index", &Index::create_payload_index, py::arg("field"))
        .def("health", [](const Index& x, std::size_t recall_sample) {
            HealthOptions o;
            o.recall_sample = recall_sample;
            HealthReport h;
            {
                py::gil_scoped_release free;
                h = x.health(o);
            }
            return health_dict(h);
        }, py::arg("recall_sample") = 100)
        .def("train_codec", [](Index& x) { py::gil_scoped_release free; x.train_codec(); })
        .def("save", [](const Index& x, const std::string& path) { py::gil_scoped_release free; x.save(path); }, py::arg("path"))
        .def_static("load", [](const std::string& path) { py::gil_scoped_release free; return Index::load(path); }, py::arg("path"))
        .def_static("open_durable", [](const std::string& dir, std::size_t dim, Metric metric, const DurableOptions& options) {
            py::gil_scoped_release free;
            if constexpr (std::is_same_v<Index, HnswIndex>) return Index::open_durable(dir, dim, metric, {}, Schema::dynamic(), options);
            else return Index::open_durable(dir, dim, metric, Schema::dynamic(), {}, {}, options);
        }, py::arg("dir"), py::arg("dim"), py::arg("metric"), py::arg("options") = DurableOptions{})
        .def("checkpoint", [](Index& x) { py::gil_scoped_release free; return x.checkpoint(); })
        .def_property_readonly("durable", &Index::durable);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "hnsw-lite: vector search with metadata filters, hybrid search and durability (C++ core)";
    m.attr("MISSING_ID") = py::int_(kMissing);
    m.def("parse_date", [](const std::string& iso) { return parse_date(iso); }, py::arg("iso"), "Milliseconds since 1970 (UTC) of an ISO 8601 date.");
    m.def("format_date", &format_date, py::arg("ms"), "An ISO 8601 date (UTC) from milliseconds since 1970.");
    m.def("now_ms", &now_ms, "The current time in milliseconds since 1970 (UTC).");

    py::register_exception<FormatError>(m, "FormatError", PyExc_RuntimeError);
    py::register_exception<Cancelled>(m, "Cancelled", PyExc_RuntimeError);

    py::enum_<Metric>(m, "Metric").value("L2", Metric::L2).value("INNER_PRODUCT", Metric::InnerProduct).value("COSINE", Metric::Cosine);
    py::enum_<FieldType>(m, "FieldType").value("INT", FieldType::Int).value("FLOAT", FieldType::Float).value("BOOL", FieldType::Bool)
        .value("KEYWORD", FieldType::Keyword).value("TAGS", FieldType::Tags);
    py::enum_<CodecKind>(m, "Codec").value("NONE", CodecKind::None).value("SQ8", CodecKind::SQ8).value("BINARY", CodecKind::Binary).value("PQ", CodecKind::PQ)
        .value("F16", CodecKind::F16)
        .value("BF16", CodecKind::BF16);
    py::enum_<Strategy>(m, "Strategy").value("AUTO", Strategy::Auto).value("EXACT", Strategy::ForceExact).value("GRAPH", Strategy::ForceGraph)
        .value("ACORN", Strategy::ForceAcorn);
    py::enum_<Preset>(m, "Preset").value("BALANCED", Preset::Balanced).value("HIGH_RECALL", Preset::HighRecall).value("LOW_MEMORY", Preset::LowMemory);
    py::enum_<Fusion>(m, "Fusion").value("RRF", Fusion::Rrf).value("WEIGHTED", Fusion::Weighted);
    py::enum_<Durability>(m, "Durability").value("BUFFERED", Durability::Buffered).value("GROUPED", Durability::Grouped).value("SYNCED", Durability::Synced);
    py::enum_<Maintenance>(m, "Maintenance").value("MANUAL", Maintenance::Manual).value("BACKGROUND", Maintenance::Background);

    // ------------------------------------------------------------ filters
    py::class_<Filter>(m, "Filter", "A condition on metadata. Combine with &, | and ~.")
        .def_static("all", &Filter::all)
        .def_static("eq", [](const std::string& f, const py::handle& v) { return with_value(v, [&](auto x) { return Filter::eq(f, x); }); })
        .def_static("ne", [](const std::string& f, const py::handle& v) { return with_value(v, [&](auto x) { return Filter::ne(f, x); }); })
        .def_static("lt", [](const std::string& f, const py::handle& v) { return with_value(v, [&](auto x) { return Filter::lt(f, x); }); })
        .def_static("le", [](const std::string& f, const py::handle& v) { return with_value(v, [&](auto x) { return Filter::le(f, x); }); })
        .def_static("gt", [](const std::string& f, const py::handle& v) { return with_value(v, [&](auto x) { return Filter::gt(f, x); }); })
        .def_static("ge", [](const std::string& f, const py::handle& v) { return with_value(v, [&](auto x) { return Filter::ge(f, x); }); })
        .def_static("between", [](const std::string& f, const py::handle& lo, const py::handle& hi) {
            auto plain = [](const py::handle& v) -> py::object {  // NumPy scalars become Python numbers
                if (!py::isinstance<py::int_>(v) && !py::isinstance<py::float_>(v) && py::hasattr(v, "item")) return v.attr("item")();
                return py::reinterpret_borrow<py::object>(v);
            };
            const py::object a = plain(lo), b = plain(hi);
            auto number = [](const py::object& v) { return !py::isinstance<py::bool_>(v) && (py::isinstance<py::int_>(v) || py::isinstance<py::float_>(v)); };
            if (!number(a) || !number(b)) throw py::type_error("between: expected two numbers");
            if (py::isinstance<py::int_>(a) && py::isinstance<py::int_>(b)) return Filter::between(f, a.cast<std::int64_t>(), b.cast<std::int64_t>());
            return Filter::between(f, a.cast<double>(), b.cast<double>());
        }, py::arg("field"), py::arg("low"), py::arg("high"))
        .def_static("in_", &filter_in, py::arg("field"), py::arg("values"))
        .def_static("has_tag", &Filter::has_tag)
        .def_static("has_any_tag", &Filter::has_any_tag)
        .def_static("has_all_tags", &Filter::has_all_tags)
        .def_static("exists", &Filter::exists)
        .def("is_all", &Filter::is_all)
        .def("__and__", [](const Filter& a, const Filter& b) { return a && b; })
        .def("__or__", [](const Filter& a, const Filter& b) { return a || b; })
        .def("__invert__", [](const Filter& a) { return !a; })
        .def("__repr__", [](const Filter& f) { return f.is_all() ? std::string("Filter.all()") : std::string("<hnsw_lite_lib.Filter>"); });

    // ------------------------------------------------------------ parameters
    py::class_<QuantizationParams>(m, "QuantizationParams")
        .def(py::init<>())
        .def_readwrite("codec", &QuantizationParams::codec)
        .def_readwrite("train_after", &QuantizationParams::train_after)
        .def_readwrite("rerank", &QuantizationParams::rerank)
        .def_readwrite("rerank_factor", &QuantizationParams::rerank_factor)
        .def_readwrite("truncate_to", &QuantizationParams::truncate_to)
        .def_readwrite("keep_full_vectors", &QuantizationParams::keep_full_vectors)
        .def_readwrite("pq_subspaces", &QuantizationParams::pq_subspaces);
    py::class_<HnswParams>(m, "HnswParams")
        .def(py::init<>())
        .def_readwrite("M", &HnswParams::M)
        .def_readwrite("ef_construction", &HnswParams::ef_construction)
        .def_readwrite("seed", &HnswParams::seed)
        .def_readwrite("repair_on_remove", &HnswParams::repair_on_remove)
        .def_readwrite("quantization", &HnswParams::quantization);
    py::class_<IvfParams>(m, "IvfParams")
        .def(py::init<>())
        .def_readwrite("lists", &IvfParams::lists)
        .def_readwrite("probes", &IvfParams::probes)
        .def_readwrite("train_after", &IvfParams::train_after);
    py::class_<DurableOptions>(m, "DurableOptions")
        .def(py::init<>())
        .def_readwrite("keep_snapshots", &DurableOptions::keep_snapshots)
        .def_readwrite("durability", &DurableOptions::durability)
        .def_property("group_interval_ms", [](const DurableOptions& o) { return o.group_interval.count(); },
                      [](DurableOptions& o, long long ms) { o.group_interval = std::chrono::milliseconds(ms); });
    py::class_<Bm25Params>(m, "Bm25Params")
        .def(py::init<>())
        .def_readwrite("k1", &Bm25Params::k1)
        .def_readwrite("b", &Bm25Params::b)
        .def_readwrite("avg_doc_length", &Bm25Params::avg_doc_length);
    py::class_<MaintenancePolicy>(m, "MaintenancePolicy")
        .def(py::init<>())
        .def_readwrite("mode", &MaintenancePolicy::mode)
        .def_property("interval_ms", [](const MaintenancePolicy& p) { return p.interval.count(); },
                      [](MaintenancePolicy& p, long long ms) { p.interval = std::chrono::milliseconds(ms); })
        .def_property("budget_ms", [](const MaintenancePolicy& p) { return p.budget.count(); },
                      [](MaintenancePolicy& p, long long ms) { p.budget = std::chrono::milliseconds(ms); })
        .def_readwrite("checkpoint_bytes", &MaintenancePolicy::checkpoint_bytes)
        .def_readwrite("checkpoint_records", &MaintenancePolicy::checkpoint_records)
        .def_property("checkpoint_age_s", [](const MaintenancePolicy& p) { return p.checkpoint_age.count(); },
                      [](MaintenancePolicy& p, long long s) { p.checkpoint_age = std::chrono::seconds(s); })
        .def_readwrite("health_check_churn", &MaintenancePolicy::health_check_churn)
        .def_readwrite("upkeep_churn", &MaintenancePolicy::upkeep_churn)
        .def_readwrite("upkeep_nodes", &MaintenancePolicy::upkeep_nodes)
        .def_readwrite("rebuild_check_churn", &MaintenancePolicy::rebuild_check_churn)
        .def_readwrite("rebuild_drift", &MaintenancePolicy::rebuild_drift);

    // ------------------------------------------------------------ HnswIndex
    py::class_<HnswIndex> hnsw(m, "HnswIndex", "Approximate nearest-neighbor index (HNSW graph) with metadata.");
    hnsw.def(py::init([](std::size_t dim, Metric metric, const HnswParams& params, const py::handle& fields, bool strict) {
             return std::make_unique<HnswIndex>(dim, metric, params, to_schema(fields, strict));
         }), py::arg("dim"), py::arg("metric") = Metric::Cosine, py::arg("params") = HnswParams{}, py::arg("schema") = py::none(),
         py::arg("strict") = false)
        .def("_search", [](const HnswIndex& x, const py::handle& query, std::size_t k, std::size_t ef, std::optional<Filter> filter, Strategy strategy,
                           const py::handle& modifiers, std::optional<double> diversity) {
            const std::vector<float> q = to_vector(query, "query");
            SearchOptions o = search_options(filter, strategy);
            o.modifiers = to_modifiers(modifiers);
            o.diversity = diversity;
            std::vector<SearchResult> r;
            {
                py::gil_scoped_release free;
                r = x.search(q, k, ef, o);
            }
            return o.modifiers || o.diversity ? scored_1d(r) : results_1d(r);
        }, py::arg("query"), py::arg("k") = 10, py::arg("ef") = 64, py::arg("filter") = py::none(), py::arg("strategy") = Strategy::Auto,
           py::arg("modifiers") = py::none(), py::arg("diversity") = py::none(),
           "Returns (ids, distances): NumPy arrays, closest first.")
        .def("search_batch", [](const HnswIndex& x, const py::handle& queries, std::size_t k, std::size_t ef, std::optional<Filter> filter,
                                std::size_t threads) {
            const FloatArray q = to_matrix(queries, "queries");
            std::vector<std::vector<SearchResult>> r;
            {
                py::gil_scoped_release free;
                r = x.search_batch(std::span<const float>(q.data(), static_cast<std::size_t>(q.size())), k, ef, threads, search_options(filter, Strategy::Auto));
            }
            return results_2d(r, k);
        }, py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 64, py::arg("filter") = py::none(), py::arg("threads") = 0,
           "Returns (ids, distances) of shape (n, k), padded with MISSING_ID and inf.")
        .def("search_by_id", [](const HnswIndex& x, std::uint64_t id, std::size_t k, std::size_t ef, std::optional<Filter> filter) -> py::object {
            std::optional<std::vector<SearchResult>> r;
            {
                py::gil_scoped_release free;
                r = x.search_by_id(id, k, ef, search_options(filter, Strategy::Auto));
            }
            return r ? py::object(results_1d(*r)) : py::none();
        }, py::arg("id"), py::arg("k") = 10, py::arg("ef") = 64, py::arg("filter") = py::none())
        .def("upkeep", [](HnswIndex& x) {
            UpkeepStats st;
            {
                py::gil_scoped_release free;
                st = x.upkeep();
            }
            py::dict d;
            d["nodes_repaired"] = st.nodes_repaired;
            d["reconnected"] = st.reconnected;
            d["complete"] = st.complete;
            return d;
        })
        .def("compact", [](HnswIndex& x) { py::gil_scoped_release free; return x.compact().reclaimed; })
        .def("reorder", [](HnswIndex& x) { py::gil_scoped_release free; return x.reorder().kept; },
             "Opt-in: stores neighbors near each other for faster searches (same results). Returns the nodes kept.")
        .def("rebuild", [](HnswIndex& x) { py::gil_scoped_release free; return x.rebuild().kept; });
    bind_records<HnswIndex>(hnsw);

    // ------------------------------------------------------------ FlatIndex
    py::class_<FlatIndex> flat(m, "FlatIndex", "Exact search (optionally with an inverted file), with metadata.");
    flat.def(py::init([](std::size_t dim, Metric metric, const py::handle& fields, bool strict, const QuantizationParams& q, const IvfParams& ivf) {
             return std::make_unique<FlatIndex>(dim, metric, to_schema(fields, strict), q, ivf);
         }), py::arg("dim"), py::arg("metric") = Metric::Cosine, py::arg("schema") = py::none(), py::arg("strict") = false,
         py::arg("quantization") = QuantizationParams{}, py::arg("ivf") = IvfParams{})
        .def("_search", [](const FlatIndex& x, const py::handle& query, std::size_t k, std::optional<Filter> filter, const py::handle& modifiers,
                           std::optional<double> diversity) {
            const std::vector<float> q = to_vector(query, "query");
            SearchOptions o = search_options(filter, Strategy::Auto);
            o.modifiers = to_modifiers(modifiers);
            o.diversity = diversity;
            std::vector<SearchResult> r;
            {
                py::gil_scoped_release free;
                r = x.search(q, k, o);
            }
            return o.modifiers || o.diversity ? scored_1d(r) : results_1d(r);
        }, py::arg("query"), py::arg("k") = 10, py::arg("filter") = py::none(), py::arg("modifiers") = py::none(), py::arg("diversity") = py::none())
        .def("search_batch", [](const FlatIndex& x, const py::handle& queries, std::size_t k, std::optional<Filter> filter, std::size_t threads) {
            const FloatArray q = to_matrix(queries, "queries");
            std::vector<std::vector<SearchResult>> r;
            {
                py::gil_scoped_release free;
                r = x.search_batch(std::span<const float>(q.data(), static_cast<std::size_t>(q.size())), k, threads, search_options(filter, Strategy::Auto));
            }
            return results_2d(r, k);
        }, py::arg("queries"), py::arg("k") = 10, py::arg("filter") = py::none(), py::arg("threads") = 0)
        .def("train_lists", [](FlatIndex& x) { py::gil_scoped_release free; x.train_lists(); });
    bind_records<FlatIndex>(flat);

    // ------------------------------------------------------------ Collection
    py::class_<CollectionOptions>(m, "CollectionOptions")
        .def(py::init<>())
        .def_static("preset", &CollectionOptions::preset, py::arg("dim"), py::arg("preset") = Preset::Balanced, py::arg("expected_size") = 100000)
        .def_readwrite("dim", &CollectionOptions::dim)
        .def_readwrite("metric", &CollectionOptions::metric)
        .def_readwrite("hnsw", &CollectionOptions::hnsw)
        .def_readwrite("text", &CollectionOptions::text)
        .def_readwrite("bm25", &CollectionOptions::bm25)
        .def_readwrite("sparse", &CollectionOptions::sparse)
        .def_readwrite("store_text", &CollectionOptions::store_text)
        .def_readwrite("durability", &CollectionOptions::durability)
        .def_readwrite("maintenance", &CollectionOptions::maintenance)
        .def_readwrite("ef", &CollectionOptions::ef)
        .def_property("ttl_enabled", [](const CollectionOptions& o) { return o.ttl.enabled; }, [](CollectionOptions& o, bool v) { o.ttl.enabled = v; })
        .def_property("ttl_default_ms", [](const CollectionOptions& o) { return o.ttl.default_ms; }, [](CollectionOptions& o, std::int64_t v) { o.ttl.default_ms = v; })
        .def_readwrite("string_ids", &CollectionOptions::string_ids)
        .def("set_schema", [](CollectionOptions& o, const py::handle& fields, bool strict) { o.schema = to_schema(fields, strict); },
             py::arg("fields"), py::arg("strict") = false);

    auto document = [](const py::handle& vector, const py::handle& text, const py::handle& sparse, const py::handle& metadata,
                       const py::handle& expires_at = py::none()) {
        Document d;
        if (!expires_at.is_none()) d.expires_at = expires_at.cast<std::int64_t>();
        if (!vector.is_none()) d.vector = to_vector(vector, "vector");
        if (!text.is_none()) d.text = text.cast<std::string>();
        d.sparse = to_sparse(sparse);
        d.metadata = to_metadata(metadata);
        return d;
    };
    py::class_<Collection>(m, "Collection", "Documents (vector, text, sparse vector, metadata) under one ID space, with hybrid search.")
        .def(py::init<CollectionOptions>(), py::arg("options"))
        .def_static("open", [](const std::string& dir, const CollectionOptions& o) { py::gil_scoped_release free; return Collection::open(dir, o); },
                    py::arg("dir"), py::arg("options"))
        .def("__len__", &Collection::size)
        .def("__contains__", [](const Collection& c, const py::handle& id) {
            if (py::isinstance<py::str>(id)) return c.contains(std::string_view(id.cast<std::string>()));
            return c.contains(to_ids(id).at(0));
        }, py::arg("id"))
        .def("add", [document](Collection& c, const py::handle& id, const py::handle& vector, const py::handle& text, const py::handle& sparse,
                                  const py::handle& metadata, const py::handle& expires_at) {
            const Document d = document(vector, text, sparse, metadata, expires_at);
            if (py::isinstance<py::str>(id)) {
                const std::string key = id.cast<std::string>();
                py::gil_scoped_release free;
                c.add(std::string_view(key), d);
            } else {
                const std::uint64_t n = to_ids(id).at(0);
                py::gil_scoped_release free;
                c.add(n, d);
            }
        }, py::arg("id"), py::arg("vector") = py::none(), py::arg("text") = py::none(), py::arg("sparse") = py::none(), py::arg("metadata") = py::none(),
           py::arg("expires_at") = py::none())
        .def("upsert", [document](Collection& c, const py::handle& id, const py::handle& vector, const py::handle& text, const py::handle& sparse,
                                  const py::handle& metadata, const py::handle& expires_at) {
            const Document d = document(vector, text, sparse, metadata, expires_at);
            if (py::isinstance<py::str>(id)) {
                const std::string key = id.cast<std::string>();
                py::gil_scoped_release free;
                return c.upsert(std::string_view(key), d);
            } else {
                const std::uint64_t n = to_ids(id).at(0);
                py::gil_scoped_release free;
                return c.upsert(n, d);
            }
        }, py::arg("id"), py::arg("vector") = py::none(), py::arg("text") = py::none(), py::arg("sparse") = py::none(), py::arg("metadata") = py::none(),
           py::arg("expires_at") = py::none())
        .def("_upsert_many", [](Collection& c, const py::handle& ids, const py::handle& vectors, const py::handle& texts, const py::handle& sparse,
                                const py::handle& metadata) {
            // Converted first (under the GIL), then written without it.
            const std::vector<std::uint64_t> id = to_ids(ids);
            std::vector<Document> docs(id.size());
            if (!vectors.is_none()) {
                const FloatArray m = to_matrix(vectors, "vectors");
                if (static_cast<std::size_t>(m.shape(0)) != id.size()) throw py::value_error("upsert_many: one vector per id required");
                const std::size_t dim = static_cast<std::size_t>(m.shape(1));
                for (std::size_t i = 0; i < id.size(); ++i) docs[i].vector.assign(m.data() + i * dim, m.data() + (i + 1) * dim);
            }
            auto each = [&](const py::handle& seq, const char* what, auto&& set) {
                if (seq.is_none()) return;
                std::size_t i = 0;
                for (auto item : seq) {
                    if (i >= id.size()) throw py::value_error(std::string("upsert_many: more ") + what + " than ids");
                    set(docs[i++], item);
                }
                if (i != id.size()) throw py::value_error(std::string("upsert_many: one entry of ") + what + " per id required");
            };
            each(texts, "texts", [](Document& d, const py::handle& t) { if (!t.is_none()) d.text = py::str(t); });
            each(sparse, "sparse vectors", [](Document& d, const py::handle& s) { d.sparse = to_sparse(s); });
            each(metadata, "metadata", [](Document& d, const py::handle& md) { d.metadata = to_metadata(md); });
            std::size_t added = 0;
            py::gil_scoped_release free;
            for (std::size_t i = 0; i < id.size(); ++i) added += c.upsert(id[i], docs[i]);
            return added;
        }, py::arg("ids"), py::arg("vectors") = py::none(), py::arg("texts") = py::none(), py::arg("sparse") = py::none(),
           py::arg("metadata") = py::none())
        .def("_upsert_many_keyed", [document](Collection& c, const std::vector<std::string>& keys, const py::handle& vectors, const py::handle& texts,
                                              const py::handle& sparse, const py::handle& metadata, const py::handle& expires) {
            // String keys and/or expiry times: one document at a time, converted first.
            std::vector<Document> docs(keys.size());
            const FloatArray m = vectors.is_none() ? FloatArray() : to_matrix(vectors, "vectors");
            for (std::size_t i = 0; i < keys.size(); ++i) {
                if (!vectors.is_none()) {
                    if (static_cast<std::size_t>(m.shape(0)) != keys.size()) throw py::value_error("upsert_many: one vector per id required");
                    const std::size_t dim = static_cast<std::size_t>(m.shape(1));
                    docs[i].vector.assign(m.data() + i * dim, m.data() + (i + 1) * dim);
                }
            }
            auto each = [&](const py::handle& seq, auto&& set) {
                if (seq.is_none()) return;
                std::size_t i = 0;
                for (auto item : seq) {
                    if (i >= keys.size()) throw py::value_error("upsert_many: more entries than ids");
                    set(docs[i++], item);
                }
                if (i != keys.size()) throw py::value_error("upsert_many: one entry per id required");
            };
            each(texts, [](Document& d, const py::handle& t) { if (!t.is_none()) d.text = py::str(t); });
            each(sparse, [](Document& d, const py::handle& s) { d.sparse = to_sparse(s); });
            each(metadata, [](Document& d, const py::handle& md) { d.metadata = to_metadata(md); });
            each(expires, [](Document& d, const py::handle& e) { if (!e.is_none()) d.expires_at = e.cast<std::int64_t>(); });
            std::size_t added = 0;
            py::gil_scoped_release free;
            for (std::size_t i = 0; i < keys.size(); ++i) added += c.upsert(std::string_view(keys[i]), docs[i]);
            return added;
        }, py::arg("keys"), py::arg("vectors") = py::none(), py::arg("texts") = py::none(), py::arg("sparse") = py::none(),
           py::arg("metadata") = py::none(), py::arg("expires_at") = py::none())
        .def("remove", [](Collection& c, const py::handle& ids) {
            const std::vector<std::uint64_t> id = to_ids(ids);
            py::gil_scoped_release free;
            return c.remove_batch(id);
        }, py::arg("ids"))
        .def("remove_expired", [](Collection& c) { py::gil_scoped_release free; return c.remove_expired(); })
        .def("_remove_key", [](Collection& c, const std::string& key) { py::gil_scoped_release free; return c.remove(std::string_view(key)); })
        .def("_contains_key", [](const Collection& c, const std::string& key) { return c.contains(std::string_view(key)); })
        .def_static("id_of", [](const std::string& key) { return Collection::id_of(key); }, py::arg("key"))
        .def("_set_metadata_key", [](Collection& c, const std::string& key, const py::handle& md) {
            const Metadata m = to_metadata(md);
            return c.set_metadata(std::string_view(key), m);
        }, py::arg("key"), py::arg("metadata"))
        .def("_set_runtime", [](Collection& c, bool ttl, std::int64_t default_ms, bool string_ids) {
            c.set_ttl(TtlOptions{ttl, default_ms});
            c.set_string_ids(string_ids);
        }, py::arg("ttl"), py::arg("default_ms"), py::arg("string_ids"))
        .def("remove_where", [](Collection& c, const Filter& f) { py::gil_scoped_release free; return c.remove_where(f); }, py::arg("filter"))
        .def("count_where", [](const Collection& c, const Filter& f) { py::gil_scoped_release free; return c.count_where(f); }, py::arg("filter"))
        .def("set_metadata", [](Collection& c, std::uint64_t id, const py::handle& md) { const Metadata m = to_metadata(md); return c.set_metadata(id, m); },
             py::arg("id"), py::arg("metadata"))
        .def("set_metadata_where", [](Collection& c, const Filter& f, const py::handle& md) { const Metadata m = to_metadata(md); return c.set_metadata_where(f, m); },
             py::arg("filter"), py::arg("changes"))
        .def("get", [](const Collection& c, const py::handle& id) -> py::object {
            std::optional<DocumentRecord> r;
            if (py::isinstance<py::str>(id)) {
                const std::string key = id.cast<std::string>();
                py::gil_scoped_release free;
                r = c.get(std::string_view(key));
            } else {
                const std::uint64_t n = to_ids(id).at(0);
                py::gil_scoped_release free;
                r = c.get(n);
            }
            if (!r) return py::none();
            py::dict d;
            d["id"] = r->id;
            d["vector"] = r->vector ? py::object(py::array_t<float>(static_cast<py::ssize_t>(r->vector->size()), r->vector->data())) : py::none();
            d["text"] = r->text;
            d["sparse"] = r->sparse ? py::object(from_sparse(*r->sparse)) : py::none();
            d["metadata"] = from_metadata(r->metadata);
            if (r->key) d["key"] = *r->key;
            if (r->expires_at) d["expires_at"] = *r->expires_at;
            return d;
        }, py::arg("id"))
        .def("_search", [](const Collection& c, const py::handle& vector, const py::handle& text, const py::handle& sparse, std::size_t k,
                           std::optional<Filter> filter, Fusion fusion, double rrf_k, double vector_weight, double text_weight, double sparse_weight,
                           std::size_t candidates, std::size_t ef, bool with_metadata, bool with_text, std::optional<double> timeout_s,
                           const py::handle& modifiers, std::optional<double> diversity) {
            CollectionQuery q;
            if (!vector.is_none()) q.vector = to_vector(vector, "vector");
            if (!text.is_none()) q.text = text.cast<std::string>();
            q.sparse = to_sparse(sparse);
            CollectionSearchOptions o;
            o.filter = filter;
            o.fusion = fusion;
            o.rrf_k = rrf_k;
            o.vector_weight = vector_weight;
            o.text_weight = text_weight;
            o.sparse_weight = sparse_weight;
            o.candidates = candidates;
            o.ef = ef;
            o.with_metadata = with_metadata;
            o.with_text = with_text;
            o.modifiers = to_modifiers(modifiers);
            o.diversity = diversity;
            if (timeout_s) o.deadline = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(*timeout_s));
            bool stopped = false;
            o.stopped_early = &stopped;
            std::vector<CollectionHit> hits;
            {
                py::gil_scoped_release free;
                hits = c.search(q, k, o);
            }
            py::list out;
            for (const CollectionHit& h : hits) {
                py::dict d;
                d["id"] = h.id;
                d["score"] = h.score;
                d["vector_rank"] = opt(h.vector_rank);
                d["text_rank"] = opt(h.text_rank);
                d["sparse_rank"] = opt(h.sparse_rank);
                d["vector_distance"] = optf(h.vector_distance);
                d["text_score"] = optf(h.text_score);
                d["sparse_score"] = optf(h.sparse_score);
                if (h.base_score) d["base_score"] = *h.base_score;
                if (h.key) d["key"] = *h.key;
                if (with_metadata) d["metadata"] = h.metadata ? py::object(from_metadata(*h.metadata)) : py::none();
                if (with_text) d["text"] = h.text ? py::object(py::str(*h.text)) : py::none();
                out.append(d);
            }
            return py::make_tuple(out, stopped);
        }, py::arg("vector"), py::arg("text"), py::arg("sparse"), py::arg("k"), py::arg("filter"), py::arg("fusion"), py::arg("rrf_k"),
           py::arg("vector_weight"), py::arg("text_weight"), py::arg("sparse_weight"), py::arg("candidates"), py::arg("ef"), py::arg("with_metadata"),
           py::arg("with_text"), py::arg("timeout"), py::arg("modifiers") = py::none(), py::arg("diversity") = py::none())
        .def("checkpoint", [](Collection& c) { py::gil_scoped_release free; return c.checkpoint(); })
        .def_property_readonly("durable", &Collection::durable)
        .def("maintain", [](Collection& c, long long budget_ms) {
            MaintenanceReport r;
            {
                py::gil_scoped_release free;
                r = c.maintain(std::chrono::milliseconds(budget_ms));
            }
            return maintenance_dict(r);
        }, py::arg("budget_ms") = 0)
        .def("set_maintenance", &Collection::set_maintenance, py::arg("policy"))
        .def("maintenance_status", [](const Collection& c) {
            const MaintenanceStatus st = c.maintenance_status();
            py::dict d;
            d["log_bytes"] = st.log_bytes;
            d["log_records"] = st.log_records;
            d["since_checkpoint_s"] = st.since_checkpoint.count();
            d["churn_since_upkeep"] = st.churn_since_upkeep;
            d["churn_since_rebuild"] = st.churn_since_rebuild;
            d["rounds"] = st.rounds;
            return d;
        })
        .def_property_readonly("options", &Collection::options)
        .def("save", [](const Collection& c, const std::string& path) { py::gil_scoped_release free; c.save(path); }, py::arg("path"))
        .def_static("load", [](const std::string& path) { py::gil_scoped_release free; return Collection::load(path); }, py::arg("path"));
}