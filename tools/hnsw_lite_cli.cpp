// hnsw-lite: inspect, verify, check and export hnsw-lite files and durable directories,
// without writing code. It never changes what it reads: a durable directory is
// recovered from a temporary copy (opening it in place would take its lock, and could
// trim a torn log tail).
//
//   hnsw-lite inspect PATH
//   hnsw-lite verify PATH
//   hnsw-lite health PATH [--recall-sample N]
//   hnsw-lite export PATH [--format jsonl|csv] [--output FILE] [--no-vectors] [--limit N]
//
// Exit status: 0 success, 1 a damaged file or a failed operation, 2 a usage error.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

#include "collection.h"
#include "disk_index.h"
#include "flat_index.h"
#include "hnsw_index.h"
#include "inspect.h"
#include "serialize.h"
#include "sparse.h"

namespace fs = std::filesystem;
using namespace vecdb;

namespace {

constexpr const char* kUsage =
    "hnsw-lite: inspect, verify, check and export hnsw-lite files and durable directories\n"
    "\n"
    "usage:\n"
    "  hnsw-lite inspect PATH        what it holds: kind, size, settings, metadata fields\n"
    "  hnsw-lite verify PATH         load it with every check (a directory: recover it); exit 1 if damaged\n"
    "  hnsw-lite health PATH [--recall-sample N]\n"
    "                                the graph's health (HNSW indexes, collections with vectors), flat and disk indexes\n"
    "  hnsw-lite export PATH [--format jsonl|csv] [--output FILE] [--no-vectors] [--limit N]\n"
    "                                every record or document, as JSON Lines (default) or CSV\n"
    "\n"
    "PATH is a file written by save(), a disk index file, or a durable directory; it is never changed.\n"
    "Exit status: 0 success, 1 a damaged file or a failed operation, 2 a usage error.\n";

struct Usage : std::runtime_error {
    using std::runtime_error::runtime_error;
};

const char* metric_name(Metric m) {
    switch (m) {
        case Metric::L2: return "l2";
        case Metric::InnerProduct: return "inner product";
        default: return "cosine";
    }
}

const char* codec_name(CodecKind c) {
    switch (c) {
        case CodecKind::None: return "none";
        case CodecKind::SQ8: return "SQ8";
        case CodecKind::Binary: return "binary";
        case CodecKind::PQ: return "PQ";
        case CodecKind::F16: return "float16";
        default: return "bfloat16";
    }
}

/// A copy of a durable directory in the system's temporary folder, removed on exit.
struct TempCopy {
    fs::path dir;
    explicit TempCopy(const std::string& source) {
        static std::atomic<unsigned> serial{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        dir = fs::temp_directory_path() / ("hnsw-lite-cli-" + std::to_string(stamp) + "-" + std::to_string(serial++));
        fs::create_directories(dir);
        for (const auto& e : fs::directory_iterator(source))
            if (e.is_regular_file() && e.path().filename() != "LOCK") fs::copy_file(e.path(), dir / e.path().filename());
    }
    ~TempCopy() {
        std::error_code ignored;
        fs::remove_all(dir, ignored);
    }
    TempCopy(const TempCopy&) = delete;
    TempCopy& operator=(const TempCopy&) = delete;
};

/// What PATH holds, loaded (a durable directory: recovered from a copy).
struct Loaded {
    StoredInfo info;
    RecoveryStats recovery;
    std::unique_ptr<TempCopy> copy;  // declared first: destroyed after the index closes
    std::unique_ptr<HnswIndex> hnsw;
    std::unique_ptr<FlatIndex> flat;
    std::unique_ptr<SparseIndex> sparse;
    std::unique_ptr<Collection> collection;
    std::unique_ptr<DiskIndex> disk;
};

std::string newest_snapshot(const std::string& dir, const StoredInfo& info) {
    if (info.snapshots.empty()) throw FormatError("a durable directory without a snapshot: " + dir);
    char name[64];
    std::snprintf(name, sizeof name, "snapshot-%020llu.idx", static_cast<unsigned long long>(info.snapshots.back()));
    return (fs::path(dir) / name).string();
}

Loaded load(const std::string& path) {
    Loaded l;
    l.info = identify(path);
    if (!l.info.durable) {
        switch (l.info.kind) {
            case StoredKind::Hnsw: l.hnsw = HnswIndex::load(path); break;
            case StoredKind::Flat: l.flat = FlatIndex::load(path); break;
            case StoredKind::Sparse: l.sparse = SparseIndex::load(path); break;
            case StoredKind::Collection: l.collection = Collection::load(path); break;
            case StoredKind::Disk: l.disk = DiskIndex::open(path); break;
        }
        return l;
    }
    // A durable directory: its settings from the newest snapshot, then a full recovery
    // (snapshot and log) of a copy.
    const std::string snapshot = newest_snapshot(path, l.info);
    l.copy = std::make_unique<TempCopy>(path);
    const std::string dir = l.copy->dir.string();
    switch (l.info.kind) {
        case StoredKind::Hnsw: {
            const auto s = HnswIndex::load(snapshot);
            l.hnsw = HnswIndex::open_durable(dir, s->dim(), s->metric(), s->params(), s->schema(), {}, &l.recovery);
            break;
        }
        case StoredKind::Flat: {
            const auto s = FlatIndex::load(snapshot);
            l.flat = FlatIndex::open_durable(dir, s->dim(), s->metric(), s->schema(), s->quantization(), {}, {}, &l.recovery);
            break;
        }
        case StoredKind::Sparse: l.sparse = SparseIndex::open_durable(dir, Schema::dynamic(), {}, &l.recovery); break;
        case StoredKind::Collection: {
            CollectionOptions o = Collection::load(snapshot)->options();
            o.maintenance.mode = Maintenance::Manual;
            l.collection = Collection::open(dir, o, &l.recovery);
            break;
        }
        case StoredKind::Disk: throw FormatError("a disk index is a file, not a durable directory");  // identify() never reports this
    }
    return l;
}

std::string with_commas(std::uint64_t n) {
    std::string s = std::to_string(n);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<std::size_t>(i), ",");
    return s;
}

bool reserved(const std::string& field) { return field == "_key" || field == "_expires_at"; }  // a collection's own

std::string fields_line(const Schema& schema, const std::vector<std::string>& indexed) {
    std::string out = schema.mode() == SchemaMode::Strict ? "strict schema" : "dynamic schema";
    std::vector<FieldSpec> shown;
    for (const FieldSpec& f : schema.fields())
        if (!reserved(f.name)) shown.push_back(f);
    if (shown.empty()) return out + ", no fields";
    out += ":";
    for (std::size_t i = 0; i < shown.size(); ++i) {
        const FieldSpec& f = shown[i];
        out += (i ? ", " : " ") + f.name + " (" + field_type_name(f.type);
        if (std::find(indexed.begin(), indexed.end(), f.name) != indexed.end()) out += ", indexed";
        out += ")";
    }
    return out;
}

void row(const char* key, const std::string& value) { std::printf("%-14s %s\n", key, value.c_str()); }

std::string lsn_list(const std::vector<std::uint64_t>& v) {
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + std::to_string(v[i]);
    return out;
}

void describe(const std::string& path, const Loaded& l) {
    const StoredInfo& i = l.info;
    row("path", path);
    row("kind", std::string(stored_kind_name(i.kind)) + (i.durable ? " (durable directory)" : " (file)"));
    if (i.version) row("format", "version " + std::to_string(i.version));
    row("size", with_commas(i.bytes) + " bytes");
    if (i.durable) {
        char id[24];
        std::snprintf(id, sizeof id, "%016llx", static_cast<unsigned long long>(i.identity));
        row("identity", id);
        row("snapshots", std::to_string(i.snapshots.size()) + " (LSN " + lsn_list(i.snapshots) + ")");
        row("log", std::to_string(i.segments.size()) + (i.segments.size() == 1 ? " segment" : " segments") +
                       (i.segments.empty() ? "" : " (first LSN " + lsn_list(i.segments) + ")"));
        row("recovered", std::to_string(l.recovery.replayed) + " log records replayed" +
                             (l.recovery.dropped_unfinished ? ", an unfinished last one dropped" : ""));
    }
    auto vectors = [&](std::size_t live, std::size_t deleted, std::size_t dim, Metric m) {
        row("vectors", with_commas(live) + (deleted ? " (+" + with_commas(deleted) + " removed, not yet compacted)" : ""));
        row("dimensions", std::to_string(dim));
        row("metric", metric_name(m));
    };
    if (l.hnsw) {
        const HnswIndex& h = *l.hnsw;
        vectors(h.size(), h.deleted_count(), h.dim(), h.metric());
        row("graph", "M " + std::to_string(h.params().M) + ", ef_construction " + std::to_string(h.params().ef_construction));
        const QuantizationParams& q = h.params().quantization;
        row("compression", q.codec == CodecKind::None ? std::string("none")
                                                      : std::string(codec_name(q.codec)) + (!h.has_vector_values() ? ", codes only" : ", with full vectors") +
                                                            (h.codec_trained() ? "" : " (not trained yet)"));
        row("metadata", fields_line(h.schema(), h.payload_indexes()));
    } else if (l.flat) {
        const FlatIndex& f = *l.flat;
        vectors(f.size(), f.deleted_count(), f.dim(), f.metric());
        const QuantizationParams& q = f.quantization();
        row("compression", q.codec == CodecKind::None ? std::string("none") : std::string(codec_name(q.codec)) + (!f.has_vector_values() ? ", codes only" : ""));
        row("lists (IVF)", f.lists_trained() ? "trained" : "none or not trained");
        row("metadata", fields_line(f.schema(), f.payload_indexes()));
    } else if (l.sparse) {
        const SparseIndex& s = *l.sparse;
        row("vectors", with_commas(s.size()));
        row("terms", with_commas(s.terms()) + " (" + with_commas(s.postings()) + " postings)");
        row("metadata", fields_line(s.schema(), s.payload_indexes()));
    } else if (l.collection) {
        const Collection& c = *l.collection;
        const CollectionOptions o = c.options();
        row("documents", with_commas(c.size()));
        std::string sides;
        if (o.dim) sides += "vectors (" + std::to_string(o.dim) + " dimensions, " + metric_name(o.metric) + ")";
        if (o.text) sides += std::string(sides.empty() ? "" : ", ") + "text";
        if (o.sparse) sides += std::string(sides.empty() ? "" : ", ") + "sparse vectors";
        row("sides", sides.empty() ? "metadata only" : sides);
        std::string settings;
        if (o.ttl.enabled) settings += "expiry" + (o.ttl.default_ms ? " (default " + std::to_string(o.ttl.default_ms) + " ms)" : std::string());
        if (o.string_ids) settings += std::string(settings.empty() ? "" : ", ") + "string keys";
        row("settings", settings.empty() ? "none" : settings);
        row("metadata", fields_line(c.schema(), c.payload_indexes()));
    } else if (l.disk) {
        const DiskIndex& d = *l.disk;
        vectors(d.size(), 0, d.dim(), d.metric());
        row("graph", "max degree " + std::to_string(d.max_degree()));
    }
}

// ---------------------------------------------------------------- export

std::string number(double x) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.17g", x);
    return buf;
}
std::string number(float x) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(x));  // enough digits to read back exactly
    return buf;
}

std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out + "\"";
}

std::string json_value(const Value& v) {
    switch (v.type) {
        case FieldType::Int: return std::to_string(v.i);
        case FieldType::Float: return number(v.f);
        case FieldType::Bool: return v.b ? "true" : "false";
        case FieldType::Keyword: return json_string(v.s);
        default: {
            std::string out = "[";
            for (std::size_t i = 0; i < v.tags.size(); ++i) out += (i ? "," : "") + json_string(v.tags[i]);
            return out + "]";
        }
    }
}

std::string json_vector(const std::vector<float>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + number(v[i]);
    return out + "]";
}

std::string json_sparse(const SparseVector& v) {
    std::string out = "{\"indices\":[";
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + std::to_string(v.indices[i]);
    out += "],\"values\":[";
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + number(v.values[i]);
    return out + "]}";
}

std::string json_metadata(const Metadata& m, const std::vector<std::string>& names) {
    std::string out = "{";
    bool first = true;
    for (const std::string& name : names)
        if (const auto v = m.get(name)) {
            out += (first ? "" : ",") + json_string(name) + ":" + json_value(*v);
            first = false;
        }
    return out + "}";
}

std::string csv_cell(const std::string& s) {
    if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
    std::string out = "\"";
    for (const char c : s) out += c == '"' ? std::string("\"\"") : std::string(1, c);
    return out + "\"";
}

/// One exported row: the columns in order, as JSON text.
struct Row {
    std::vector<std::pair<std::string, std::string>> cells;
    void add(const std::string& name, const std::string& json) { cells.emplace_back(name, json); }
};

struct Exporter {
    std::ostream& out;
    bool csv;
    std::vector<std::string> fields;  // metadata columns, in schema order
    bool header_done = false;
    std::size_t rows = 0;

    void write(const Row& r, const Metadata& m) {
        if (!csv) {
            out << "{";
            for (std::size_t i = 0; i < r.cells.size(); ++i) out << (i ? "," : "") << json_string(r.cells[i].first) << ":" << r.cells[i].second;
            out << (r.cells.empty() ? "" : ",") << "\"metadata\":" << json_metadata(m, fields) << "}\n";
        } else {
            if (!header_done) {
                for (std::size_t i = 0; i < r.cells.size(); ++i) out << (i ? "," : "") << csv_cell(r.cells[i].first);
                for (const std::string& f : fields) out << "," << csv_cell(f);
                out << "\n";
                header_done = true;
            }
            for (std::size_t i = 0; i < r.cells.size(); ++i) {
                std::string cell = r.cells[i].second;
                if (cell.size() >= 2 && cell.front() == '"' && cell.back() == '"' && r.cells[i].first != "vector") {
                    std::string text;  // a plain JSON string: the text itself, unescaped
                    for (std::size_t k = 1; k + 1 < cell.size(); ++k) {
                        if (cell[k] == '\\' && k + 2 < cell.size()) {
                            const char e = cell[++k];
                            text += e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e;
                        } else {
                            text += cell[k];
                        }
                    }
                    cell = text;
                }
                out << (i ? "," : "") << csv_cell(cell);
            }
            for (const std::string& f : fields) {
                const auto v = m.get(f);
                std::string cell;
                if (v) cell = v->type == FieldType::Keyword ? v->s : json_value(*v);
                out << "," << csv_cell(cell);
            }
            out << "\n";
        }
        ++rows;
    }
};

std::vector<std::string> field_names(const Schema& s) {
    std::vector<std::string> names;
    for (const FieldSpec& f : s.fields()) names.push_back(f.name);
    return names;
}

std::size_t export_all(const Loaded& l, std::ostream& out, bool csv, bool vectors, std::size_t limit) {
    Exporter ex{out, csv, {}, false, 0};
    const auto more = [&] { return limit == 0 || ex.rows < limit; };
    if (l.hnsw || l.flat) {
        ex.fields = field_names(l.hnsw ? l.hnsw->schema() : l.flat->schema());
        ScanOptions o;
        o.limit = 1000;
        while (more()) {
            const Page page = l.hnsw ? l.hnsw->scan(o) : l.flat->scan(o);
            for (const Record& r : page.records) {
                if (!more()) break;
                Row row;
                row.add("id", std::to_string(r.id));
                if (vectors) row.add("vector", json_vector(r.vector));
                ex.write(row, r.metadata);
            }
            if (!page.next) break;
            o.after = page.next;
        }
    } else if (l.sparse) {
        ex.fields = field_names(l.sparse->schema());
        for (const SparseRecord& r : l.sparse->records()) {
            if (!more()) break;
            Row row;
            row.add("id", std::to_string(r.id));
            if (vectors) row.add("sparse", json_sparse(r.vector));
            ex.write(row, r.metadata);
        }
    } else if (l.collection) {
        const CollectionOptions opts = l.collection->options();
        ex.fields = field_names(l.collection->schema());
        std::erase_if(ex.fields, reserved);  // reported as the key and expires_at columns
        CollectionScanOptions o;
        o.limit = 1000;
        o.with_vectors = vectors;
        while (more()) {
            const CollectionPage page = l.collection->scan(o);
            for (const DocumentRecord& d : page.documents) {
                if (!more()) break;
                Row row;
                row.add("id", std::to_string(d.id));
                if (opts.string_ids) row.add("key", d.key ? json_string(*d.key) : "null");
                if (opts.ttl.enabled) row.add("expires_at", d.expires_at ? std::to_string(*d.expires_at) : "null");
                if (vectors && opts.dim) row.add("vector", d.vector ? json_vector(*d.vector) : "null");
                if (vectors && opts.sparse) row.add("sparse", d.sparse ? json_sparse(*d.sparse) : "null");
                if (opts.text) row.add("text", json_string(d.text));
                ex.write(row, d.metadata);
            }
            if (!page.next) break;
            o.after = page.next;
        }
    } else {
        throw std::runtime_error("export is not available for disk indexes (export the index they were built from)");
    }
    return ex.rows;
}

// ---------------------------------------------------------------- commands

int cmd_inspect(const std::string& path) {
    const Loaded l = load(path);
    describe(path, l);
    return 0;
}

int cmd_verify(const std::string& path) {
    const Loaded l = load(path);  // every loader checks checksums, structure and values
    std::size_t n = 0;
    if (l.hnsw) n = l.hnsw->size();
    if (l.flat) n = l.flat->size();
    if (l.sparse) n = l.sparse->size();
    if (l.collection) n = l.collection->size();
    if (l.disk) n = l.disk->size();
    std::printf("OK: %s %s, %s %s%s\n", stored_kind_name(l.info.kind), l.info.durable ? "directory" : "file", with_commas(n).c_str(),
                l.collection ? "documents" : "vectors",
                l.info.durable ? (", recovered with " + std::to_string(l.recovery.replayed) + " log records").c_str() : "");
    return 0;
}

int cmd_health(const std::string& path, std::size_t recall_sample) {
    const Loaded l = load(path);
    HealthOptions o;
    o.recall_sample = recall_sample;
    HealthReport r;
    if (l.hnsw) r = l.hnsw->health(o);
    else if (l.flat) r = l.flat->health(o);
    else if (l.disk) r = l.disk->health(o);
    else if (l.collection && l.collection->dense_index()) r = l.collection->dense_index()->health(o);
    else throw std::runtime_error("health applies to HNSW, flat and disk indexes, and collections with vectors");
    row("live", with_commas(r.live));
    row("free slots", with_commas(r.free_slots) + " of " + with_commas(r.slots));
    if (!r.levels.empty()) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.2f (fewest %zu)", r.mean_degree, r.min_degree);
        row("degree", buf);
        row("underfull", with_commas(r.underfull));
        row("unreachable", with_commas(r.unreachable));
        row("no incoming", with_commas(r.no_incoming));
        row("to removed", with_commas(r.links_to_removed) + " links");
        row("stale links", with_commas(r.stale_links));
    }
    if (r.lists) row("lists (IVF)", std::to_string(r.lists) + ", largest " + with_commas(r.largest_list) + ", " + std::to_string(r.empty_lists) + " empty");
    if (r.codec_error) row("codec error", number(*r.codec_error));
    if (r.recall) row("recall", number(*r.recall) + " (estimated on " + std::to_string(recall_sample) + " stored vectors)");
    return 0;
}

int cmd_export(const std::string& path, const std::string& format, const std::string& output, bool vectors, std::size_t limit) {
    if (format != "jsonl" && format != "csv") throw Usage("--format must be jsonl or csv");
    const Loaded l = load(path);
    std::size_t n = 0;
    if (output.empty()) {
        n = export_all(l, std::cout, format == "csv", vectors, limit);
        std::cout.flush();
    } else {
        std::ofstream out(output, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write " + output);
        n = export_all(l, out, format == "csv", vectors, limit);
        out.flush();
        if (!out) throw std::runtime_error("writing " + output + " failed");
        std::fprintf(stderr, "exported %s rows to %s\n", with_commas(n).c_str(), output.c_str());
    }
    return 0;
}

std::size_t count_arg(const std::string& name, const std::string& value) {
    std::size_t used = 0;
    unsigned long long n = 0;
    try {
        n = std::stoull(value, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != value.size() || value.empty() || value[0] == '-') throw Usage(name + " needs a whole number, not '" + value + "'");
    return static_cast<std::size_t>(n);
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
    // Output byte for byte as on other systems: in text mode, Windows would turn every
    // "\n" into "\r\n", changing newlines inside exported values.
    (void)_setmode(_fileno(stdout), _O_BINARY);
#endif
    const std::vector<std::string> args(argv + 1, argv + argc);
    try {
        if (args.empty() || args[0] == "--help" || args[0] == "-h" || args[0] == "help") {
            std::fputs(kUsage, args.empty() ? stderr : stdout);
            return args.empty() ? 2 : 0;
        }
        const std::string& cmd = args[0];
        if (args.size() < 2) throw Usage(cmd + " needs a PATH");
        const std::string& path = args[1];
        std::string format = "jsonl", output;
        bool vectors = true;
        std::size_t limit = 0, recall = 100;
        for (std::size_t i = 2; i < args.size(); ++i) {
            const std::string& a = args[i];
            auto next = [&]() -> const std::string& {
                if (i + 1 >= args.size()) throw Usage(a + " needs a value");
                return args[++i];
            };
            if (cmd == "export" && a == "--format") format = next();
            else if (cmd == "export" && a == "--output") output = next();
            else if (cmd == "export" && a == "--no-vectors") vectors = false;
            else if (cmd == "export" && a == "--limit") limit = count_arg(a, next());
            else if (cmd == "health" && a == "--recall-sample") recall = count_arg(a, next());
            else throw Usage("unknown option for " + cmd + ": " + a);
        }
        if (!fs::exists(path)) {
            std::fprintf(stderr, "hnsw-lite: %s: no such file or directory\n", path.c_str());
            return 1;
        }
        if (cmd == "inspect") return cmd_inspect(path);
        if (cmd == "verify") return cmd_verify(path);
        if (cmd == "health") return cmd_health(path, recall);
        if (cmd == "export") return cmd_export(path, format, output, vectors, limit);
        throw Usage("unknown command: " + cmd);
    } catch (const Usage& e) {
        std::fprintf(stderr, "hnsw-lite: %s\n\n%s", e.what(), kUsage);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hnsw-lite: %s\n", e.what());
        if (args.size() >= 1 && args[0] == "verify") std::fprintf(stderr, "FAILED: %s\n", args.size() > 1 ? args[1].c_str() : "");
        return 1;
    }
}