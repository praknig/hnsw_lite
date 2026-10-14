/**
 * @file persistence.cpp
 * @brief Saving and loading indexes (file format version 1; see serialize.h).
 *
 * Sections, in order:
 *   Flat: CONF (dimension, metric), NODE (each vector with its user ID), META, END
 *   HNSW: CONF (dimension, metric, HNSW settings, planner settings),
 *         NODE (each slot: user ID, removed flag, level, vector, links per level),
 *         STAT (entry point, top level, free slots in order, random generator state),
 *         META, END
 *   META: schema mode, fields (name, type, indexed), string dictionary, then each
 *         field's rows (slot and value), in slot order.
 *
 * The graph is saved logically (levels and link lists) and rebuilt on load, so
 * the file does not depend on memory layout. Removed slots are saved too: other
 * nodes may still link through them, and the next inserts reuse them in the
 * saved order, so an index continues after loading exactly as if it had never
 * been saved. Payload-index counts and string reference counts are recomputed
 * on load rather than trusted from the file.
 */
#include <filesystem>
#include <fstream>
#include <sstream>

#include "flat_index.h"
#include "hnsw_index.h"
#include "serialize.h"

namespace vecdb {

using persist::IndexKind;
using persist::Reader;
using persist::Writer;

namespace {

constexpr std::uint32_t kConf = persist::tag("CONF");
constexpr std::uint32_t kNode = persist::tag("NODE");
constexpr std::uint32_t kStat = persist::tag("STAT");
constexpr std::uint32_t kMeta = persist::tag("META");
constexpr std::uint32_t kEnd = persist::tag("END ");

/// Largest dimension accepted from a file (16 million floats per vector).
constexpr std::uint64_t kMaxDim = std::uint64_t(1) << 24;

/// Writes to "<path>.tmp", then renames it over `path`; removes the temporary
/// file if anything fails, so an existing file at `path` is never damaged.
template <class Write>
void save_to_file(const std::string& path, Write&& write) {
    const std::string tmp = path + ".tmp";
    try {
        {
            std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
            if (!file) throw std::runtime_error("cannot open " + tmp + " for writing");
            const persist::StreamErrorsThrow errors(file);
            write(file);
            file.close();
            if (!file) throw std::runtime_error("writing " + tmp + " failed");  // GCOVR_EXCL_BR_LINE: defensive, StreamErrorsThrow throws first
        }
        std::filesystem::rename(tmp, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);
        throw;
    }
}

std::ifstream open_for_reading(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path + " for reading");
    return file;
}

/// Runs a load. Problems with the content become FormatError; running out of
/// memory passes through unchanged.
template <class Load>
auto guarded_load(Load&& load) {
    try {
        return load();
    } catch (const FormatError&) {
        throw;
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& e) {
        throw FormatError(std::string("invalid index content: ") + e.what());
    }
}

Metric read_metric(Reader& r) {
    const auto m = r.get<std::uint32_t>();
    if (m > static_cast<std::uint32_t>(Metric::Cosine)) throw FormatError("unknown metric");
    return static_cast<Metric>(m);
}

std::size_t read_dim(Reader& r) {
    const auto dim = r.get<std::uint64_t>();
    if (dim == 0 || dim > kMaxDim) throw FormatError("invalid dimension");
    return static_cast<std::size_t>(dim);
}

void write_end(std::ostream& out) {
    persist::write_section(out, kEnd, [](Writer&) {});
    out.flush();
    if (!out) throw std::runtime_error("writing the index failed");  // GCOVR_EXCL_BR_LINE: defensive, StreamErrorsThrow throws first
}

void read_end(Reader& r) {
    r.begin_section(kEnd, "END");
    r.end_section();
    r.expect_end_of_file();
}

}  // namespace

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

void MetadataStore::save(Writer& w) const {
    w.put<std::uint8_t>(mode_ == SchemaMode::Strict ? 1 : 0);
    w.put<std::uint32_t>(static_cast<std::uint32_t>(specs_.size()));
    for (std::size_t f = 0; f < specs_.size(); ++f) {
        w.put_string(specs_[f].name);
        w.put<std::uint8_t>(static_cast<std::uint8_t>(specs_[f].type));
        w.put<std::uint8_t>(cols_[f].indexed ? 1 : 0);
    }
    w.put<std::uint32_t>(static_cast<std::uint32_t>(strings_.size()));
    for (const std::string& s : strings_) w.put_string(s);
    for (const Column& c : cols_) {
        w.put<std::uint64_t>(c.present_count);
        for (NodeId s = 0; s < c.rows(); ++s) {
            if (!c.present[s]) continue;
            w.put<std::uint32_t>(s);
            switch (c.type) {
                case FieldType::Int: w.put<std::int64_t>(c.ints[s]); break;
                case FieldType::Float: w.put<double>(c.floats[s]); break;
                case FieldType::Bool: w.put<std::uint8_t>(c.bools[s]); break;
                case FieldType::Keyword: w.put<std::uint32_t>(c.keywords[s]); break;
                case FieldType::Tags:
                    w.put<std::uint32_t>(static_cast<std::uint32_t>(c.tags[s].size()));
                    for (std::uint32_t id : c.tags[s]) w.put<std::uint32_t>(id);
                    break;
            }
        }
    }
}

MetadataStore MetadataStore::load(Reader& r, std::size_t slots, const std::function<bool(NodeId)>& live) {
    const auto mode = r.get<std::uint8_t>();
    if (mode > 1) throw FormatError("invalid schema mode");
    const auto field_count = r.get<std::uint32_t>();
    if (field_count > r.section_left() / 6) throw FormatError("count of metadata fields exceeds the data in the file");
    std::vector<FieldSpec> specs;
    std::vector<bool> indexed;
    for (std::uint32_t f = 0; f < field_count; ++f) {
        std::string name = r.get_string();
        const auto type = r.get<std::uint8_t>();
        const auto is_indexed = r.get<std::uint8_t>();
        if (type > static_cast<std::uint8_t>(FieldType::Tags)) throw FormatError("invalid field type");
        if (is_indexed > 1) throw FormatError("invalid payload-index flag");
        specs.push_back({std::move(name), static_cast<FieldType>(type)});
        indexed.push_back(is_indexed == 1);
    }
    MetadataStore m(mode == 1 ? Schema::strict(specs) : Schema::dynamic(specs));  // checks names

    const auto string_count = r.get<std::uint32_t>();
    if (string_count > r.section_left() / 4) throw FormatError("count of strings exceeds the data in the file");
    m.strings_.reserve(string_count);
    for (std::uint32_t id = 0; id < string_count; ++id) {
        std::string s = r.get_string();
        if (!m.string_ids_.emplace(s, id).second) throw FormatError("duplicate string in the metadata dictionary");
        m.strings_.push_back(std::move(s));
    }
    m.refs_.assign(string_count, 0);
    m.dead_strings_ = string_count;  // each becomes used as rows reference it

    for (std::size_t f = 0; f < m.cols_.size(); ++f) {
        Column& c = m.cols_[f];
        const auto rows = r.get_count(5, "metadata rows");
        if (rows > slots) throw FormatError("more metadata rows than vectors");
        std::int64_t previous = -1;
        for (std::uint64_t i = 0; i < rows; ++i) {
            const auto slot = r.get<std::uint32_t>();
            if (slot >= slots || std::int64_t(slot) <= previous) throw FormatError("metadata rows out of range or out of order");
            if (!live(slot)) throw FormatError("metadata on a removed vector");
            previous = slot;
            c.ensure(std::size_t(slot) + 1);
            switch (c.type) {
                case FieldType::Int: c.ints[slot] = r.get<std::int64_t>(); break;
                case FieldType::Float: {
                    const double v = r.get<double>();
                    if (!std::isfinite(v)) throw FormatError("a metadata float is NaN or infinite");
                    c.floats[slot] = v;
                    break;
                }
                case FieldType::Bool: {
                    const auto v = r.get<std::uint8_t>();
                    if (v > 1) throw FormatError("invalid boolean value");
                    c.bools[slot] = v;
                    break;
                }
                case FieldType::Keyword: {
                    const auto id = r.get<std::uint32_t>();
                    if (id >= string_count) throw FormatError("string reference out of range");
                    c.keywords[slot] = id;
                    break;
                }
                case FieldType::Tags: {
                    const auto count = r.get<std::uint32_t>();
                    if (count > r.section_left() / 4) throw FormatError("count of tags exceeds the data in the file");
                    std::vector<std::uint32_t> ids;
                    ids.reserve(count);
                    for (std::uint32_t t = 0; t < count; ++t) {
                        const auto id = r.get<std::uint32_t>();
                        if (id >= string_count) throw FormatError("string reference out of range");
                        if (std::find(ids.begin(), ids.end(), id) != ids.end()) throw FormatError("repeated tag");
                        ids.push_back(id);
                    }
                    c.tags[slot] = std::move(ids);
                    break;
                }
            }
            c.present[slot] = 1;
            ++c.present_count;
            m.ref_row(c, slot);
        }
    }
    for (std::size_t f = 0; f < specs.size(); ++f)
        if (indexed[f]) m.index_field(specs[f].name);  // recounts from the rows (rejects numeric fields)
    return m;
}

// ---------------------------------------------------------------------------
// FlatIndex
// ---------------------------------------------------------------------------

void FlatIndex::save(const std::string& path) const {
    save_to_file(path, [&](std::ostream& out) { save(out); });
}

void FlatIndex::save(std::ostream& out) const {
    const auto running = searches_.enter();  // a save reads: changes are refused meanwhile
    const persist::StreamErrorsThrow errors(out);
    persist::write_header(out, IndexKind::Flat);
    persist::write_section(out, kConf, [&](Writer& w) {
        w.put<std::uint64_t>(dim());
        w.put<std::uint32_t>(static_cast<std::uint32_t>(metric_));
    });
    persist::write_section(out, kNode, [&](Writer& w) {
        w.put<std::uint64_t>(vectors_.size());
        for (NodeId n = 0; n < vectors_.size(); ++n) {
            w.put<std::uint64_t>(ids_.external(n));
            w.put_floats(vectors_.get(n).data(), dim());
        }
    });
    persist::write_section(out, kMeta, [&](Writer& w) { metadata_.save(w); });
    write_end(out);
}

std::unique_ptr<FlatIndex> FlatIndex::load(const std::string& path) {
    std::ifstream file = open_for_reading(path);
    return load(file);
}

std::unique_ptr<FlatIndex> FlatIndex::load(std::istream& in) {
    return guarded_load([&] {
        Reader r(in);
        r.read_header(IndexKind::Flat);

        r.begin_section(kConf, "CONF");
        const std::size_t dim = read_dim(r);
        const Metric metric = read_metric(r);
        r.end_section();
        auto index = std::make_unique<FlatIndex>(dim, metric);

        // Vectors are stored as saved (already normalized for cosine), bypassing add().
        r.begin_section(kNode, "NODE");
        const auto n = r.get_count(8 + 4 * std::uint64_t(dim), "vectors");
        std::vector<float> v(dim);
        for (std::uint64_t i = 0; i < n; ++i) {
            const auto id = r.get<std::uint64_t>();
            r.get_floats(v.data(), dim);
            index->ids_.add(id);  // a repeated user ID throws (and becomes a FormatError)
            index->vectors_.add(v);
        }
        r.end_section();
        index->live_ = static_cast<std::size_t>(n);

        r.begin_section(kMeta, "META");
        index->metadata_ = MetadataStore::load(r, index->vectors_.size(), [](NodeId) { return true; });
        r.end_section();
        read_end(r);
        return index;
    });
}

// ---------------------------------------------------------------------------
// HnswIndex
// ---------------------------------------------------------------------------

void HnswIndex::save(const std::string& path) const {
    save_to_file(path, [&](std::ostream& out) { save(out); });
}

void HnswIndex::save(std::ostream& out) const {
    const auto running = searches_.enter();  // a save reads: changes are refused meanwhile
    const persist::StreamErrorsThrow errors(out);
    const IdMap& ids = storage_.ids();
    const GraphStorage& graph = storage_.graph();
    persist::write_header(out, IndexKind::Hnsw);
    persist::write_section(out, kConf, [&](Writer& w) {
        w.put<std::uint64_t>(dim());
        w.put<std::uint32_t>(static_cast<std::uint32_t>(metric_));
        w.put<std::uint64_t>(params_.M);
        w.put<std::uint64_t>(params_.ef_construction);
        w.put<std::uint64_t>(params_.seed);
        w.put<std::uint8_t>(params_.repair_on_remove ? 1 : 0);
        w.put<std::uint64_t>(planner_.max_exact_matches);
        w.put<double>(planner_.max_exact_fraction);
        w.put<std::uint64_t>(planner_.sample_size);
        w.put<double>(planner_.max_ef_multiplier);
    });
    persist::write_section(out, kNode, [&](Writer& w) {
        w.put<std::uint64_t>(storage_.size());
        for (NodeId n = 0; n < storage_.size(); ++n) {
            const int level = graph.level(n);
            w.put<std::uint64_t>(ids.external(n));
            w.put<std::uint8_t>(ids.is_deleted(n) ? 1 : 0);
            w.put<std::uint8_t>(static_cast<std::uint8_t>(level));
            w.put_floats(storage_.vectors().get(n).data(), dim());
            for (int l = 0; l <= level; ++l) {
                const auto links = graph.links(n, l);
                const std::size_t count = GraphStorage::count(links);
                w.put<std::uint32_t>(static_cast<std::uint32_t>(count));
                for (std::size_t i = 0; i < count; ++i) w.put<std::uint32_t>(links[i]);
            }
        }
    });
    std::ostringstream rng_state;
    const persist::StreamErrorsThrow rng_errors(rng_state);
    rng_state << rng_;  // the standard's portable text form of the generator's state
    persist::write_section(out, kStat, [&](Writer& w) {
        w.put<std::uint32_t>(entry_);
        w.put<std::int32_t>(max_level_);
        w.put<std::uint64_t>(free_slots_.size());
        for (NodeId s : free_slots_) w.put<std::uint32_t>(s);
        w.put_string(rng_state.str());
    });
    persist::write_section(out, kMeta, [&](Writer& w) { metadata_.save(w); });
    write_end(out);
}

std::unique_ptr<HnswIndex> HnswIndex::load(const std::string& path) {
    std::ifstream file = open_for_reading(path);
    return load(file);
}

std::unique_ptr<HnswIndex> HnswIndex::load(std::istream& in) {
    return guarded_load([&] {
        Reader r(in);
        r.read_header(IndexKind::Hnsw);

        r.begin_section(kConf, "CONF");
        const std::size_t dim = read_dim(r);
        const Metric metric = read_metric(r);
        HnswParams params;
        params.M = static_cast<std::size_t>(r.get<std::uint64_t>());
        params.ef_construction = static_cast<std::size_t>(r.get<std::uint64_t>());
        params.seed = r.get<std::uint64_t>();
        const auto repair = r.get<std::uint8_t>();
        if (repair > 1) throw FormatError("invalid repair flag");
        params.repair_on_remove = repair == 1;
        PlannerParams planner;
        planner.max_exact_matches = static_cast<std::size_t>(r.get<std::uint64_t>());
        planner.max_exact_fraction = r.get<double>();
        planner.sample_size = static_cast<std::size_t>(r.get<std::uint64_t>());
        planner.max_ef_multiplier = r.get<double>();
        r.end_section();
        validate(planner);
        auto index = std::make_unique<HnswIndex>(dim, metric, params);  // checks the settings
        index->planner_ = planner;

        // Every slot, live or removed, with its level, vector and links.
        r.begin_section(kNode, "NODE");
        const auto n = r.get_count(8 + 1 + 1 + 4 * std::uint64_t(dim) + 4, "vectors");
        if (n >= kEmpty) throw FormatError("too many vectors");  // GCOVR_EXCL_BR_LINE: needs a file over 100 GB
        const std::size_t M0 = 2 * params.M;
        std::vector<float> v(dim);
        std::vector<NodeId> links;
        std::vector<std::uint8_t> removed(static_cast<std::size_t>(n));
        std::vector<int> levels(static_cast<std::size_t>(n));
        Storage& st = index->storage_;
        for (NodeId slot = 0; slot < n; ++slot) {
            const auto id = r.get<std::uint64_t>();
            const auto is_removed = r.get<std::uint8_t>();
            if (is_removed > 1) throw FormatError("invalid removed flag");
            const int level = r.get<std::uint8_t>();
            r.get_floats(v.data(), dim);
            if (is_removed) {
                st.insert_unmapped(id, v, level);  // holds its old user ID, not findable
                st.ids().retire(slot);
            } else {
                st.insert(id, v, level);  // a repeated live user ID throws
            }
            removed[slot] = is_removed;
            levels[slot] = level;
            for (int l = 0; l <= level; ++l) {
                const auto count = r.get<std::uint32_t>();
                if (count > (l == 0 ? M0 : params.M)) throw FormatError("a link list is longer than its capacity");
                links.resize(count);
                for (std::uint32_t i = 0; i < count; ++i) {
                    const auto nb = r.get<std::uint32_t>();
                    if (nb >= n || nb == slot) throw FormatError("a link points outside the index or to itself");
                    if (std::find(links.begin(), links.begin() + i, nb) != links.begin() + i)
                        throw FormatError("a repeated link");
                    links[i] = nb;
                }
                st.graph().set_links(slot, l, links);
            }
        }
        r.end_section();

        // Entry point, top level, free slots, random generator.
        r.begin_section(kStat, "STAT");
        const auto entry = r.get<std::uint32_t>();
        const auto max_level = r.get<std::int32_t>();
        const auto free_count = r.get_count(4, "free slots");
        std::size_t live = 0, removed_count = 0;
        int top = -1;
        for (NodeId s = 0; s < n; ++s) {
            if (removed[s]) {
                ++removed_count;
            } else {
                ++live;
                top = std::max(top, levels[s]);
            }
        }
        if (live == 0) {
            if (entry != kEmpty || max_level != -1) throw FormatError("an empty index must have no entry point");
        } else if (entry >= n || removed[entry] || levels[entry] != max_level || max_level != top) {
            throw FormatError("invalid entry point or top level");
        }
        if (free_count != removed_count) throw FormatError("the free list does not match the removed vectors");
        std::vector<NodeId> free_slots;
        free_slots.reserve(static_cast<std::size_t>(free_count));
        std::vector<bool> seen(static_cast<std::size_t>(n), false);
        for (std::uint64_t i = 0; i < free_count; ++i) {
            const auto s = r.get<std::uint32_t>();
            if (s >= n || !removed[s] || seen[s]) throw FormatError("invalid free list");
            seen[s] = true;
            free_slots.push_back(s);
        }
        std::istringstream rng_state(r.get_string());
        const persist::StreamErrorsThrow rng_errors(rng_state);
        rng_state >> index->rng_;
        if (rng_state.fail() || !(rng_state >> std::ws).eof()) throw FormatError("invalid random generator state");
        r.end_section();
        index->entry_ = entry;
        index->max_level_ = max_level;
        index->live_ = live;
        index->free_slots_ = std::move(free_slots);

        r.begin_section(kMeta, "META");
        index->metadata_ = MetadataStore::load(r, static_cast<std::size_t>(n), [&](NodeId s) { return !removed[s]; });
        r.end_section();
        read_end(r);
        return index;
    });
}

}  // namespace vecdb