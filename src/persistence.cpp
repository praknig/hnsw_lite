/**
 * @file persistence.cpp
 * @brief Saving and loading indexes (file format version 1; see serialize.h).
 *
 * Sections, in order:
 *   Flat: CONF (dimension, metric; version 3: truncation), NODE (each vector with its user ID, and for
 *         cosine in version 2 the vector as given), META, END
 *   (Version 3 adds CODC after NODE for both kinds: quantization settings, and
 *   when trained the codec state and every slot's code.)
 *   HNSW: CONF (dimension, metric, HNSW settings, planner settings; version 3:
 *         truncation),
 *         NODE (each slot: user ID, removed flag, level, vector, for cosine in
 *         version 2 the vector as given, links per level),
 *         STAT (entry point, top level, free slots in order, random generator state as one u64),
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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

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
constexpr std::uint32_t kCodec = persist::tag("CODC");  // version 3

/// CODC: quantization settings, then (if trained) the codec's state and every
/// slot's code. Written for every index in version 3, even without a codec.
void write_codec(Writer& w, const QuantizationParams& q, const VectorCodec* codec, const std::vector<std::uint8_t>& codes,
                 std::size_t slots) {
    w.put<std::uint8_t>(static_cast<std::uint8_t>(q.codec));
    w.put<std::uint64_t>(q.train_after);
    w.put<std::uint8_t>(q.rerank ? 1 : 0);
    w.put<std::uint64_t>(q.rerank_factor);
    const bool trained = codec && codec->trained();
    w.put<std::uint8_t>(trained ? 1 : 0);
    if (!trained) return;
    const std::vector<float> state = codec->state();
    w.put<std::uint64_t>(state.size());
    w.put_floats(state.data(), state.size());
    w.put<std::uint64_t>(slots);
    w.bytes(codes.data(), slots * codec->code_size());
}

/// Reads what write_codec wrote; every value is checked.
struct LoadedCodec {
    QuantizationParams params;
    std::unique_ptr<VectorCodec> codec;
    std::vector<std::uint8_t> codes;
};
LoadedCodec read_codec(Reader& r, std::size_t dim, Metric metric, std::size_t slots) {
    LoadedCodec c;
    const auto kind = r.get<std::uint8_t>();
    if (kind > static_cast<std::uint8_t>(CodecKind::SQ8)) throw FormatError("unknown codec");
    c.params.codec = static_cast<CodecKind>(kind);
    c.params.train_after = static_cast<std::size_t>(r.get<std::uint64_t>());
    const auto rerank = r.get<std::uint8_t>();
    if (rerank > 1) throw FormatError("invalid re-rank flag");
    c.params.rerank = rerank == 1;
    c.params.rerank_factor = static_cast<std::size_t>(r.get<std::uint64_t>());
    validate(c.params);  // invalid settings become a FormatError (see guarded_load)
    c.codec = make_codec(c.params.codec, dim, metric);
    const auto trained = r.get<std::uint8_t>();
    if (trained > 1 || (trained == 1 && !c.codec)) throw FormatError("invalid codec training flag");
    if (trained == 0) return c;
    const auto state_size = r.get_count(4, "codec state values");
    std::vector<float> state(static_cast<std::size_t>(state_size));
    r.get_floats(state.data(), state.size());
    c.codec->restore(state);  // checks the state
    const auto count = r.get_count(c.codec->code_size(), "codes");
    if (count != slots) throw FormatError("the number of codes does not match the vectors");
    c.codes.resize(static_cast<std::size_t>(count) * c.codec->code_size());
    r.bytes(c.codes.data(), c.codes.size());
    return c;
}
constexpr std::uint32_t kEnd = persist::tag("END ");

/// Largest dimension accepted from a file (16 million floats per vector).
constexpr std::uint64_t kMaxDim = std::uint64_t(1) << 24;

/// A temporary file name next to `path`, unique to this save, so several saves
/// to the same path at once (from threads or processes) never write into one
/// temporary file: "<path>.tmp-<counter>-<mix>". The counter makes names unique
/// within this program; the mix of thread, time and a memory address separates
/// programs.
std::string temporary_path(const std::string& path) {
    static std::atomic<std::uint64_t> counter{0};
    const std::uint64_t n = counter.fetch_add(1);
    std::uint64_t mix = std::uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    mix ^= std::uint64_t(std::chrono::steady_clock::now().time_since_epoch().count()) * 0x9E3779B97F4A7C15ull;
    mix ^= std::uint64_t(reinterpret_cast<std::uintptr_t>(&counter)) << 7;
    char suffix[48];
    std::snprintf(suffix, sizeof suffix, ".tmp-%llx-%016llx", static_cast<unsigned long long>(n),
                  static_cast<unsigned long long>(mix));
    return path + suffix;
}

/// Writes to a temporary file, then renames it over `path`; removes the
/// temporary file if anything fails, so an existing file at `path` is never
/// damaged. Saves to the same path at the same time each write their own
/// temporary file, and the last rename wins: the result is always one complete file.
template <class Write>
void save_to_file(const std::string& path, Write&& write) {
    const std::string tmp = temporary_path(path);
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

void FlatIndex::save(const std::string& path, const OperationOptions& options) const {
    save_to_file(path, [&](std::ostream& out) { save(out, options); });  // a Cancelled save removes the temporary file
}

void FlatIndex::save(std::ostream& out, const OperationOptions& options) const {
    const auto guard = lock_.read();  // a save reads: searches continue, changes wait
    const persist::StreamErrorsThrow errors(out);
    detail::OperationControl control(options, vectors_.size());  // progress counts records written
    persist::write_header(out, IndexKind::Flat);
    persist::write_section(out, kConf, [&](Writer& w) {
        w.put<std::uint64_t>(dim());
        w.put<std::uint32_t>(static_cast<std::uint32_t>(metric_));
        w.put<std::uint64_t>(quant_.truncate_to);  // version 3: Matryoshka truncation
    });
    persist::write_section(out, kNode, [&](Writer& w) {
        w.put<std::uint64_t>(vectors_.size());
        for (NodeId n = 0; n < vectors_.size(); ++n) {
            w.put<std::uint64_t>(ids_.external(n));
            w.put_floats(vectors_.get(n).data(), indexed_dim());
            if (originals_) w.put_floats(originals_->get(n).data(), dim());  // version 2: cosine originals (3: and truncation)
            if (!w.counting()) control.advance();
        }
    });
    persist::write_section(out, kCodec, [&](Writer& w) { write_codec(w, quant_, codec_.get(), codes_, vectors_.size()); });
    persist::write_section(out, kMeta, [&](Writer& w) { metadata_.save(w); });
    write_end(out);
    control.finish();
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
        QuantizationParams truncation;
        if (r.version() >= 3) truncation.truncate_to = static_cast<std::size_t>(r.get<std::uint64_t>());
        r.end_section();
        auto index = std::make_unique<FlatIndex>(dim, metric, Schema::dynamic(), truncation);  // checks the truncation
        const std::size_t stored = index->indexed_dim();

        // Vectors are stored as saved (already normalized for cosine), bypassing add().
        r.begin_section(kNode, "NODE");
        // Version 2 cosine files hold each original after the stored vector; for
        // version 1 files the stored (normalized) vector stands in for it.
        const bool has_original = (r.version() >= 2 && metric == Metric::Cosine) || stored < dim;
        const auto n = r.get_count(8 + 4 * std::uint64_t(stored) + (has_original ? 4 * std::uint64_t(dim) : 0), "vectors");
        std::vector<float> v(stored), original(dim);
        for (std::uint64_t i = 0; i < n; ++i) {
            const auto id = r.get<std::uint64_t>();
            r.get_floats(v.data(), stored);
            if (has_original) r.get_floats(original.data(), dim);
            index->ids_.add(id);  // a repeated user ID throws (and becomes a FormatError)
            index->vectors_.add(v);
            if (index->originals_) index->originals_->add(has_original ? original : v);
        }
        r.end_section();
        index->live_ = static_cast<std::size_t>(n);

        if (r.version() >= 3) {
            r.begin_section(kCodec, "CODC");
            LoadedCodec c = read_codec(r, stored, metric, index->vectors_.size());
            c.params.truncate_to = index->quant_.truncate_to;  // set by CONF
            r.end_section();
            index->quant_ = c.params;
            index->codec_ = std::move(c.codec);
            index->codes_ = std::move(c.codes);
        }
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

void HnswIndex::save(const std::string& path, const OperationOptions& options) const {
    save_to_file(path, [&](std::ostream& out) { save(out, options); });  // a Cancelled save removes the temporary file
}

void HnswIndex::save(std::ostream& out, const OperationOptions& options) const {
    const auto guard = lock_.read();  // a save reads: searches continue, changes wait
    const persist::StreamErrorsThrow errors(out);
    detail::OperationControl control(options, storage_.size());  // progress counts records written
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
        w.put<std::uint64_t>(params_.quantization.truncate_to);  // version 3: Matryoshka truncation
    });
    persist::write_section(out, kNode, [&](Writer& w) {
        w.put<std::uint64_t>(storage_.size());
        for (NodeId n = 0; n < storage_.size(); ++n) {
            const int level = graph.level(n);
            w.put<std::uint64_t>(ids.external(n));
            w.put<std::uint8_t>(ids.is_deleted(n) ? 1 : 0);
            w.put<std::uint8_t>(static_cast<std::uint8_t>(level));
            w.put_floats(storage_.vectors().get(n).data(), indexed_dim());
            if (storage_.originals()) w.put_floats(storage_.originals()->get(n).data(), dim());  // version 2: cosine originals (3: and truncation)
            if (!w.counting()) control.advance();
            for (int l = 0; l <= level; ++l) {
                const auto links = graph.links(n, l);
                const std::size_t count = GraphStorage::count(links);
                w.put<std::uint32_t>(static_cast<std::uint32_t>(count));
                for (std::size_t i = 0; i < count; ++i) w.put<std::uint32_t>(links[i]);
            }
        }
    });
    persist::write_section(out, kCodec, [&](Writer& w) {
        write_codec(w, params_.quantization, codec_.get(), codes_, storage_.size());
    });
    persist::write_section(out, kStat, [&](Writer& w) {
        w.put<std::uint32_t>(entry_);
        w.put<std::int32_t>(max_level_);
        w.put<std::uint64_t>(free_slots_.size());
        for (NodeId s : free_slots_) w.put<std::uint32_t>(s);
        w.put<std::uint64_t>(rng_.state);  // the whole generator state: portable by construction
    });
    persist::write_section(out, kMeta, [&](Writer& w) { metadata_.save(w); });
    write_end(out);
    control.finish();
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
        if (r.version() >= 3) params.quantization.truncate_to = static_cast<std::size_t>(r.get<std::uint64_t>());
        r.end_section();
        validate(planner);
        auto index = std::make_unique<HnswIndex>(dim, metric, params);  // checks the settings, including the truncation
        const std::size_t stored = index->indexed_dim();
        index->planner_ = planner;

        // Every slot, live or removed, with its level, vector and links.
        r.begin_section(kNode, "NODE");
        const bool has_original = (r.version() >= 2 && metric == Metric::Cosine) || stored < dim;  // see FlatIndex::load
        const auto n = r.get_count(8 + 1 + 1 + 4 * std::uint64_t(stored) + (has_original ? 4 * std::uint64_t(dim) : 0) + 4, "vectors");
        if (n >= kEmpty) throw FormatError("too many vectors");  // GCOVR_EXCL_BR_LINE: needs a file over 100 GB
        const std::size_t M0 = 2 * params.M;
        std::vector<float> v(stored), original(dim);
        std::vector<NodeId> links;
        std::vector<std::uint8_t> removed(static_cast<std::size_t>(n));
        std::vector<int> levels(static_cast<std::size_t>(n));
        Storage& st = index->storage_;
        for (NodeId slot = 0; slot < n; ++slot) {
            const auto id = r.get<std::uint64_t>();
            const auto is_removed = r.get<std::uint8_t>();
            if (is_removed > 1) throw FormatError("invalid removed flag");
            const int level = r.get<std::uint8_t>();
            r.get_floats(v.data(), stored);
            if (has_original) r.get_floats(original.data(), dim);
            const std::span<const float> as_given = has_original ? std::span<const float>(original) : std::span<const float>(v);
            if (is_removed) {
                st.insert_unmapped(id, v, level, as_given);  // holds its old user ID, not findable
                st.ids().retire(slot);
            } else {
                st.insert(id, v, level, as_given);  // a repeated live user ID throws
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
        if (r.version() >= 3) {
            r.begin_section(kCodec, "CODC");
            LoadedCodec c = read_codec(r, stored, metric, static_cast<std::size_t>(n));
            c.params.truncate_to = index->params_.quantization.truncate_to;  // set by CONF
            r.end_section();
            index->params_.quantization = c.params;
            index->codec_ = std::move(c.codec);
            index->codes_ = std::move(c.codes);
        }

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
        index->rng_.state = r.get<std::uint64_t>();  // any value is a valid state
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