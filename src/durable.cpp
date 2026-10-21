/**
 * @file durable.cpp
 * @brief Durable index directories: creating, recovering, checkpointing; see durable.h.
 *
 * The order of steps keeps a directory recoverable whatever moment a crash hits:
 *  - creating: DURABLE (the identity) is written first, atomically; without it a
 *    directory is an unfinished creation and is created again;
 *  - checkpoint: the new log segment exists before the snapshot is written, so a
 *    crash in between leaves the older snapshot and both segments;
 *  - snapshots are written to a temporary file, synced, renamed, and the
 *    directory synced: a partial snapshot never appears under its real name;
 *  - pruning comes last, once the new snapshot is safely in place.
 */
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>

#include "collection.h"
#include "flat_index.h"
#include "hnsw_index.h"
#include "serialize.h"

namespace vecdb {

namespace fs = std::filesystem;

namespace {

constexpr char kMetaMagic[8] = {'H', 'N', 'S', 'W', 'D', 'U', 'R', '1'};
enum Kind : std::uint32_t { kHnsw = 1, kFlat = 2, kCollection = 3 };

std::string join(const std::string& dir, const std::string& name) { return (fs::path(dir) / name).string(); }

std::string numbered(const char* prefix, std::uint64_t lsn, const char* suffix) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%s%020" PRIu64 "%s", prefix, lsn, suffix);
    return buf;
}
std::string snapshot_name(std::uint64_t lsn) { return numbered("snapshot-", lsn, ".idx"); }
std::string segment_name(std::uint64_t lsn) { return numbered("wal-", lsn, ".log"); }

/// LSNs of the files named prefix<20 digits>suffix, ascending.
std::vector<std::uint64_t> numbered_files(const std::string& dir, const std::string& prefix, const std::string& suffix) {
    std::vector<std::uint64_t> out;
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.size() != prefix.size() + 20 + suffix.size() || n.rfind(prefix, 0) != 0 || n.substr(n.size() - suffix.size()) != suffix) continue;
        const std::string digits = n.substr(prefix.size(), 20);
        if (!std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; })) continue;
        out.push_back(std::stoull(digits));
    }
    std::sort(out.begin(), out.end());
    return out;
}

/// Writes a file through `write`, then syncs, renames into place and syncs the
/// directory: it appears complete under `path`, or not at all.
template <class Write>
void write_durably(const std::string& dir, const std::string& path, Write&& write) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot create " + tmp);
        write(out);
        out.flush();
        if (!out) throw std::runtime_error("cannot write " + tmp);  // GCOVR_EXCL_LINE: a full or failing disk
    }
    wal::sync_path(tmp);
    fs::rename(tmp, path);
    wal::sync_directory(dir);
}

void write_meta(const std::string& dir, Kind kind, std::uint64_t identity) {
    unsigned char m[32] = {};
    std::memcpy(m, kMetaMagic, 8);
    persist::to_little_endian(std::uint32_t(kind), m + 8);
    persist::to_little_endian(identity, m + 16);
    persist::to_little_endian(persist::crc32(0, m, 28), m + 28);
    write_durably(dir, join(dir, "DURABLE"), [&](std::ostream& out) { out.write(reinterpret_cast<const char*>(m), sizeof m); });
}

std::uint64_t read_meta(const std::string& dir, Kind kind) {
    std::ifstream in(join(dir, "DURABLE"), std::ios::binary);
    unsigned char m[32];
    if (!in.read(reinterpret_cast<char*>(m), sizeof m) || std::memcmp(m, kMetaMagic, 8) != 0 ||
        persist::from_little_endian<std::uint32_t>(m + 28) != persist::crc32(0, m, 28))
        throw FormatError("not a durable index directory (DURABLE damaged): " + dir);
    if (persist::from_little_endian<std::uint32_t>(m + 8) != kind)
        throw std::invalid_argument("durable directory holds another kind of index: " + dir);
    return persist::from_little_endian<std::uint64_t>(m + 16);
}

std::uint64_t random_identity() {
    std::random_device rd;
    return (std::uint64_t(rd()) << 32 | rd()) | 1;  // never 0
}

/// What opening found or created, before the index takes it over.
template <class Index>
struct Opened {
    std::unique_ptr<Index> index;
    std::optional<WalWriter> log;
    std::shared_ptr<detail::DurableState> state;
    bool dropped = false;  // an unfinished final operation was dropped: checkpoint at once
};

/// Creates or recovers a durable directory. `create` makes the empty index for a
/// new directory; `load` reads a snapshot.
template <class Index, class Create, class Load>
Opened<Index> open_dir(const std::string& dir, Kind kind, Create&& create, Load&& load, const DurableOptions& options,
                       RecoveryStats* stats) {
    if (options.keep_snapshots < 1) throw std::invalid_argument("DurableOptions::keep_snapshots must be at least 1");
    if (options.group_interval.count() < 1) throw std::invalid_argument("DurableOptions::group_interval must be at least 1 ms");
    fs::create_directories(dir);
    FileLock lock(join(dir, "LOCK"));
    for (const auto& e : fs::directory_iterator(dir))  // a snapshot cut short by a crash
        if (e.path().extension() == ".tmp") fs::remove(e.path());
    RecoveryStats st;
    Opened<Index> o;
    std::uint64_t identity = 0;
    if (!fs::exists(join(dir, "DURABLE"))) {  // new, or a creation cut short: (re)create
        for (const std::uint64_t l : numbered_files(dir, "snapshot-", ".idx")) fs::remove(join(dir, snapshot_name(l)));
        for (const std::uint64_t l : numbered_files(dir, "wal-", ".log")) fs::remove(join(dir, segment_name(l)));
        identity = random_identity();
        write_meta(dir, kind, identity);
        o.index = create();
        write_durably(dir, join(dir, snapshot_name(1)), [&](std::ostream& out) { o.index->save(out); });
        o.log = WalWriter::create(join(dir, segment_name(1)), identity, 1);
        wal::sync_directory(dir);
        st.created = true;
        st.snapshot_lsn = 1;
    } else {
        identity = read_meta(dir, kind);
        // The newest snapshot that loads; older ones are fallbacks.
        const std::vector<std::uint64_t> snaps = numbered_files(dir, "snapshot-", ".idx");
        for (auto it = snaps.rbegin(); it != snaps.rend() && !o.index; ++it) {
            try {
                o.index = load(join(dir, snapshot_name(*it)));
                st.snapshot_lsn = *it;
            } catch (const FormatError&) {  // GCOVR_EXCL_BR_LINE: other errors (a snapshot that cannot be opened) propagate
                ++st.fallbacks;
            }
        }
        if (!o.index) throw FormatError("no snapshot in " + dir + " can be read");
        // Every record from the snapshot on, across segments, without gaps.
        const std::vector<std::uint64_t> segs = numbered_files(dir, "wal-", ".log");
        std::vector<WalRecord> records;
        std::uint64_t expected = st.snapshot_lsn;
        std::string last;
        for (std::size_t i = 0; i < segs.size(); ++i) {
            WalScan s = wal::read_segment(join(dir, segment_name(segs[i])));
            if (s.identity != identity) throw FormatError("log segment from another index in " + dir);
            if (s.first_lsn != segs[i]) throw FormatError("log segment named for another LSN in " + dir);
            const std::uint64_t end = s.first_lsn + s.records.size();
            if (end <= expected && !(end == expected && i + 1 == segs.size())) continue;  // before the snapshot  // GCOVR_EXCL_BR_LINE: checkpoints leave no empty segment before another
            if (s.first_lsn > expected) throw FormatError("log records missing before LSN " + std::to_string(s.first_lsn) + " in " + dir);
            if (s.torn_tail() && i + 1 != segs.size())
                throw FormatError("a torn record in an earlier log segment (history lost) in " + dir);
            st.torn_bytes = s.file_size - s.valid_end;
            ++st.segments;
            for (WalRecord& r : s.records)
                if (r.lsn >= expected) records.push_back(std::move(r));  // GCOVR_EXCL_BR_LINE: checkpoints start segments at snapshots, so no segment straddles one
            expected = end;
            last = join(dir, segment_name(segs[i]));
        }
        st.replayed = records.size();
        st.dropped_unfinished = o.dropped = o.index->replay(records);
        // Continue the last segment (cutting off a torn tail), or start one.
        if (!last.empty()) o.log = WalWriter::open(last, identity);
        else o.log = WalWriter::create(join(dir, segment_name(expected)), identity, expected);
        if (o.log->next_lsn() != expected) throw FormatError("log segments do not continue the snapshot in " + dir);  // GCOVR_EXCL_LINE: checked above
    }
    o.state = std::make_shared<detail::DurableState>(dir, identity, std::move(lock), options);
    if (stats) *stats = st;
    return o;
}

/// A checkpoint, given the index's log and a way to save it (the caller holds a
/// lock that excludes changes). Returns the snapshot's LSN.
template <class Save>
std::uint64_t checkpoint_dir(detail::DurableState& d, ChangeLog& log, Save&& save) {
    const std::uint64_t lsn = log.writer().next_lsn();
    const std::string snap = join(d.dir, snapshot_name(lsn));
    if (fs::exists(snap)) return lsn;  // nothing logged since the last one
    // 1. A new segment, so that the snapshot covers exactly what precedes it. If an
    //    earlier attempt rotated already and then failed to write its snapshot, the
    //    log is still that segment, with nothing in it: retry the snapshot alone.
    const std::string segment = join(d.dir, segment_name(lsn));
    if (log.writer().path() != segment) {
        if (!log.writer().broken()) log.writer().sync();
        log.rotate(WalWriter::create(segment, d.identity, lsn));  // heals a failed log: the snapshot holds the true state
        wal::sync_directory(d.dir);
    }
    // 2. The snapshot.
    write_durably(d.dir, snap, save);
    // 3. Prune: keep the newest snapshots, and the segments the oldest kept one needs.
    const std::vector<std::uint64_t> snaps = numbered_files(d.dir, "snapshot-", ".idx");
    if (snaps.size() > d.options.keep_snapshots) {
        const std::size_t drop = snaps.size() - d.options.keep_snapshots;
        for (std::size_t i = 0; i < drop; ++i) fs::remove(join(d.dir, snapshot_name(snaps[i])));
        const std::uint64_t oldest_kept = snaps[drop];
        for (const std::uint64_t l : numbered_files(d.dir, "wal-", ".log"))
            if (l < oldest_kept) fs::remove(join(d.dir, segment_name(l)));
        wal::sync_directory(d.dir);
    }
    return lsn;
}

}  // namespace

// ---------------------------------------------------------------- HnswIndex

std::unique_ptr<HnswIndex> HnswIndex::open_durable(const std::string& dir, std::size_t dim, Metric metric, const HnswParams& params,
                                                   Schema schema, const DurableOptions& options, RecoveryStats* stats) {
    Opened<HnswIndex> o = open_dir<HnswIndex>(
        dir, kHnsw, [&] { return std::make_unique<HnswIndex>(dim, metric, params, std::move(schema)); },
        [](const std::string& path) { return HnswIndex::load(path); }, options, stats);
    if (o.index->dim() != dim || o.index->metric() != metric)
        throw std::invalid_argument("open_durable: the directory holds an index of another dimension or metric");
    o.index->log_.set_sync_each(options.durability == Durability::Synced);
    o.index->attach_log(std::move(*o.log));
    o.index->durable_ = std::move(o.state);
    o.index->durable_->start([raw = o.index.get()] { raw->log_.flush(); });
    if (o.dropped) (void)o.index->checkpoint();  // never replay the dropped record again
    return std::move(o.index);
}

std::uint64_t HnswIndex::checkpoint() {
    if (!durable_) throw std::logic_error("checkpoint: not a durable index (open it with open_durable)");
    const std::lock_guard<std::mutex> one(durable_->checkpoint_mutex);
    const auto guard = lock_.read();  // no change can run until the snapshot is written
    return checkpoint_dir(*durable_, log_, [&](std::ostream& out) { save_unlocked(out, {}); });
}

// ---------------------------------------------------------------- FlatIndex

std::unique_ptr<FlatIndex> FlatIndex::open_durable(const std::string& dir, std::size_t dim, Metric metric, Schema schema,
                                                   const QuantizationParams& quantization, const IvfParams& ivf,
                                                   const DurableOptions& options, RecoveryStats* stats) {
    Opened<FlatIndex> o = open_dir<FlatIndex>(
        dir, kFlat, [&] { return std::make_unique<FlatIndex>(dim, metric, std::move(schema), quantization, ivf); },
        [](const std::string& path) { return FlatIndex::load(path); }, options, stats);
    if (o.index->dim() != dim || o.index->metric() != metric)
        throw std::invalid_argument("open_durable: the directory holds an index of another dimension or metric");
    o.index->log_.set_sync_each(options.durability == Durability::Synced);
    o.index->attach_log(std::move(*o.log));
    o.index->durable_ = std::move(o.state);
    o.index->durable_->start([raw = o.index.get()] { raw->log_.flush(); });
    if (o.dropped) (void)o.index->checkpoint();  // never replay the dropped record again
    return std::move(o.index);
}

std::uint64_t FlatIndex::checkpoint() {
    if (!durable_) throw std::logic_error("checkpoint: not a durable index (open it with open_durable)");
    const std::lock_guard<std::mutex> one(durable_->checkpoint_mutex);
    const auto guard = lock_.read();
    return checkpoint_dir(*durable_, log_, [&](std::ostream& out) { save_unlocked(out, {}); });
}

// ---------------------------------------------------------------- Collection

std::unique_ptr<Collection> Collection::open(const std::string& dir, CollectionOptions options, RecoveryStats* stats) {
    const DurableOptions durability = options.durability;
    const MaintenancePolicy maintenance = options.maintenance;
    options.maintenance.mode = Maintenance::Manual;  // no background work until the collection is set up
    RecoveryStats local;
    RecoveryStats* st = stats ? stats : &local;
    Opened<Collection> o = open_dir<Collection>(
        dir, kCollection, [&] { return std::make_unique<Collection>(options); },
        [](const std::string& path) { return Collection::load(path); }, durability, st);
    const CollectionOptions& got = o.index->options_;
    if (got.dim != options.dim || got.metric != options.metric || got.text != options.text || got.sparse != options.sparse)
        throw std::invalid_argument("Collection::open: the directory holds a collection with another dimension, metric or sides");
    o.index->options_.durability = durability;
    o.index->log_.set_sync_each(durability.durability == Durability::Synced);
    o.index->log_.attach(std::move(*o.log));
    o.index->durable_ = std::move(o.state);
    o.index->durable_->start([raw = o.index.get()] { raw->log_.flush(); });
    o.index->checkpoint_lsn_ = st->snapshot_lsn;
    if (o.dropped) (void)o.index->checkpoint();  // never replay the dropped record again
    o.index->set_maintenance(maintenance);
    return std::move(o.index);
}

std::uint64_t Collection::checkpoint() {
    if (!durable_) throw std::logic_error("checkpoint: not a durable collection (open it with Collection::open)");
    if (searching_here()) throw std::logic_error("checkpoint: refused from inside a search of this collection");
    const std::lock_guard<std::mutex> one(durable_->checkpoint_mutex);
    const std::shared_lock<std::shared_mutex> lock(mutex_);  // no change can run until the snapshot is written
    const std::uint64_t lsn = checkpoint_dir(*durable_, log_, [&](std::ostream& out) { save_unlocked(out); });
    checkpoint_lsn_ = lsn;
    checkpoint_time_ = std::chrono::steady_clock::now().time_since_epoch().count();
    return lsn;
}

}  // namespace vecdb
