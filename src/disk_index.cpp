/**
 * @file disk_index.cpp
 * @brief Building, opening and reading a disk-based index; see disk_index.h.
 */
#include "disk_index.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <mutex>
#include <future>
#include <deque>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "prepared_vector.h"
#include "serialize.h"
#include "thread_pool.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vecdb {


// ---------------------------------------------------------------- BlockFile

BlockFile::BlockFile(const std::string& path) {
#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
    LARGE_INTEGER size;
    if (!GetFileSizeEx(h, &size)) {
        CloseHandle(h);
        throw std::runtime_error("cannot read the size of " + path);
    }
    handle_ = reinterpret_cast<std::intptr_t>(h);
    size_ = static_cast<std::uint64_t>(size.QuadPart);
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("cannot read the size of " + path);
    }
    handle_ = fd;
    size_ = static_cast<std::uint64_t>(st.st_size);
#endif
}

BlockFile::~BlockFile() {
#ifdef _WIN32
    CloseHandle(reinterpret_cast<HANDLE>(handle_));
#else
    ::close(static_cast<int>(handle_));
#endif
}

void BlockFile::read(std::uint64_t offset, void* buffer, std::size_t n) const {
    if (offset > size_ || n > size_ - offset) throw FormatError("read past the end of the file");
    if (latency_.count() > 0) std::this_thread::sleep_for(latency_);  // simulated slow storage
    auto* p = static_cast<char*>(buffer);
    while (n > 0) {
#ifdef _WIN32
        OVERLAPPED ov{};
        ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFu);
        ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
        DWORD got = 0;
        const DWORD want = static_cast<DWORD>(std::min<std::size_t>(n, 1u << 30));
        if (!ReadFile(reinterpret_cast<HANDLE>(handle_), p, want, &got, &ov) || got == 0)
            throw std::runtime_error("read error");  // GCOVR_EXCL_LINE: needs a failing disk
#else
        const ssize_t got = ::pread(static_cast<int>(handle_), p, n, static_cast<off_t>(offset));
        if (got <= 0) throw std::runtime_error("read error");  // GCOVR_EXCL_LINE: needs a failing disk (sizes are checked above)
#endif
        p += got;
        n -= static_cast<std::size_t>(got);
        offset += static_cast<std::uint64_t>(got);
    }
}

// ---------------------------------------------------------------- IoPool

namespace detail {

/// Threads reading node blocks for any number of concurrent searches: tasks
/// are queued and their results returned through futures.
class IoPool {
public:
    explicit IoPool(std::size_t threads) {
        for (std::size_t i = 0; i < threads; ++i) workers_.emplace_back([this] { run(); });
    }
    ~IoPool() {
        {
            const std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();  // queued reads finish first
    }
    std::future<DiskIndex::Node> submit(std::packaged_task<DiskIndex::Node()> task) {
        auto f = task.get_future();
        {
            const std::lock_guard<std::mutex> lock(m_);
            tasks_.push_back(std::move(task));
        }
        cv_.notify_one();
        return f;
    }

private:
    void run() {
        for (;;) {
            std::packaged_task<DiskIndex::Node()> task;
            {
                std::unique_lock<std::mutex> lock(m_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (tasks_.empty()) return;  // stopping, and nothing left
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();  // an exception is stored in the future
        }
    }
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::packaged_task<DiskIndex::Node()>> tasks_;
    std::vector<std::thread> workers_;
    bool stop_ = false;
};

}  // namespace detail

DiskIndex::~DiskIndex() = default;

// ---------------------------------------------------------------- layout

namespace {

constexpr char kMagic[8] = {'H', 'N', 'S', 'W', 'D', 'I', 'S', 'K'};
constexpr std::uint32_t kDiskVersion = 1;
constexpr std::uint32_t kByteOrder = 0x01020304;
constexpr std::uint32_t kNoNeighbor = 0xFFFFFFFFu;
enum Section { kBlockCrc = 0, kPq = 1, kIds = 2, kMeta = 3, kSections = 4 };

/// Little-endian appending and reading of plain numbers.
struct Bytes {
    std::string data;
    template <class T> void put(T v) {
        unsigned char b[sizeof(T)];
        persist::to_little_endian(v, b);
        data.append(reinterpret_cast<const char*>(b), sizeof(T));
    }
    void put_floats(const float* v, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) put(v[i]);
    }
};
template <class T> T load_le(const unsigned char* p) { return persist::from_little_endian<T>(p); }

struct Layout {
    std::size_t node_bytes, nodes_per_block, blocks_per_node;
    std::uint64_t node_blocks;
    Layout(std::size_t dim, std::size_t R, std::uint64_t n) {
        node_bytes = 4 * dim + 4 + 4 * R;
        if (node_bytes <= DiskIndex::kBlock) {
            nodes_per_block = DiskIndex::kBlock / node_bytes;
            blocks_per_node = 1;
            node_blocks = (n + nodes_per_block - 1) / nodes_per_block;
        } else {
            nodes_per_block = 0;  // a node takes whole blocks
            blocks_per_node = (node_bytes + DiskIndex::kBlock - 1) / DiskIndex::kBlock;
            node_blocks = n * blocks_per_node;
        }
    }
};

/// Writes `fill` to a temporary file next to `path`, then renames it into place
/// (an existing file is replaced only once the new one is complete).
template <class Fill>
void write_atomically(const std::string& path, Fill&& fill) {
    static std::atomic<unsigned> counter{0};
    const std::string tmp = path + ".tmp-" + std::to_string(counter++) + "-" + std::to_string(std::random_device{}());
    try {
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot create " + tmp);
            fill(out);
            out.flush();
            if (!out) throw std::runtime_error("cannot write " + tmp);
        }
        std::filesystem::rename(tmp, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);
        throw;
    }
}

}  // namespace

// ---------------------------------------------------------------- build

void DiskIndex::build(const std::string& path, std::span<const std::uint64_t> ids, std::span<const float> vectors, std::size_t dim,
                      Metric metric, const DiskIndexParams& params, std::span<const Metadata> metadata, Schema schema) {
    if (dim == 0 || vectors.size() != ids.size() * dim || ids.empty())
        throw std::invalid_argument("DiskIndex::build: one vector of dim values per ID, at least one");
    if (!metadata.empty() && metadata.size() != ids.size()) throw std::invalid_argument("DiskIndex::build: one metadata entry per vector");
    {
        std::unordered_set<std::uint64_t> seen;
        for (std::uint64_t id : ids)
            if (!seen.insert(id).second) throw std::invalid_argument("DiskIndex::build: repeated ID");
    }
    MetadataStore store(std::move(schema));
    for (std::size_t i = 0; i < metadata.size(); ++i) store.validate(metadata[i]);
    for (std::size_t i = 0; i < metadata.size(); ++i) (void)store.write(static_cast<NodeId>(i), metadata[i]);
    if (metadata.empty())
        for (std::size_t i = 0; i < ids.size(); ++i) (void)store.write(static_cast<NodeId>(i), Metadata());

    const VamanaGraph g = VamanaGraph::build(vectors, dim, metric, params.graph);  // checks the vectors and settings
    const std::size_t n = g.size(), R = params.graph.R;
    const Layout layout(dim, R, n);

    // PQ codes of the stored (prepared) vectors, for the in-memory search guide.
    PQCodec pq(dim, metric, params.pq_subspaces == 0 ? std::max<std::size_t>(1, dim / 2) : params.pq_subspaces);
    {
        const std::size_t take = std::min<std::size_t>(n, 100000);
        std::vector<float> sample(take * dim);
        for (std::size_t i = 0; i < take; ++i) {
            const auto v = g.vector(static_cast<NodeId>(i * n / take));
            std::copy(v.begin(), v.end(), sample.begin() + std::ptrdiff_t(i * dim));
        }
        pq.train(sample);
    }
    std::vector<std::uint8_t> codes(n * pq.code_size());
    for (NodeId i = 0; i < n; ++i) pq.encode(g.vector(i), codes.data() + std::size_t(i) * pq.code_size());

    write_atomically(path, [&](std::ofstream& out) {
        // Node blocks, with a checksum for each.
        std::vector<std::uint32_t> crcs;
        std::string block(kBlock, '\0');
        auto record = [&](NodeId i, char* at) {
            Bytes r;
            const auto v = g.vector(i);
            r.put_floats(v.data(), dim);
            const auto nbs = g.neighbors(i);
            r.put<std::uint32_t>(static_cast<std::uint32_t>(nbs.size()));
            for (std::size_t s = 0; s < R; ++s) r.put<std::uint32_t>(s < nbs.size() ? nbs[s] : kNoNeighbor);
            std::memcpy(at, r.data.data(), r.data.size());
        };
        out.write(std::string(kBlock, '\0').data(), kBlock);  // the header, written last
        if (layout.nodes_per_block > 0) {
            for (std::uint64_t b = 0; b < layout.node_blocks; ++b) {
                std::fill(block.begin(), block.end(), '\0');
                for (std::size_t s = 0; s < layout.nodes_per_block; ++s) {
                    const std::uint64_t i = b * layout.nodes_per_block + s;
                    if (i < n) record(static_cast<NodeId>(i), block.data() + s * layout.node_bytes);
                }
                crcs.push_back(persist::crc32(0, block.data(), kBlock));
                out.write(block.data(), kBlock);
            }
        } else {
            std::string big(layout.blocks_per_node * kBlock, '\0');
            for (NodeId i = 0; i < n; ++i) {
                std::fill(big.begin(), big.end(), '\0');
                record(i, big.data());
                for (std::size_t b = 0; b < layout.blocks_per_node; ++b) crcs.push_back(persist::crc32(0, big.data() + b * kBlock, kBlock));
                out.write(big.data(), std::streamsize(big.size()));
            }
        }
        // The sections held in memory.
        Bytes sections[kSections];
        for (std::uint32_t c : crcs) sections[kBlockCrc].put(c);
        const auto state = pq.state();
        sections[kPq].put<std::uint64_t>(pq.parts());
        sections[kPq].put<std::uint64_t>(state.size());
        sections[kPq].put_floats(state.data(), state.size());
        sections[kPq].data.append(reinterpret_cast<const char*>(codes.data()), codes.size());
        for (std::uint64_t id : ids) sections[kIds].put(id);
        {
            std::ostringstream meta;
            persist::write_section(meta, persist::tag("META"), [&](persist::Writer& w) { store.save(w); });
            sections[kMeta].data = meta.str();
        }
        std::uint64_t offset = kBlock * (1 + layout.node_blocks);
        Bytes header;
        header.data.append(kMagic, 8);
        header.put<std::uint32_t>(kDiskVersion);
        header.put<std::uint32_t>(kByteOrder);
        header.put<std::uint64_t>(dim);
        header.put<std::uint32_t>(static_cast<std::uint32_t>(metric));
        header.put<std::uint32_t>(0);
        header.put<std::uint64_t>(n);
        header.put<std::uint64_t>(R);
        header.put<std::uint64_t>(g.medoid());
        header.put<std::uint64_t>(layout.node_bytes);
        header.put<std::uint64_t>(layout.nodes_per_block);
        header.put<std::uint64_t>(layout.blocks_per_node);
        header.put<std::uint64_t>(layout.node_blocks);
        for (auto& sec : sections) {
            header.put<std::uint64_t>(offset);
            header.put<std::uint64_t>(sec.data.size());
            header.put<std::uint32_t>(persist::crc32(0, sec.data.data(), sec.data.size()));
            out.write(sec.data.data(), std::streamsize(sec.data.size()));
            offset += sec.data.size();
        }
        header.put<std::uint32_t>(persist::crc32(0, header.data.data(), header.data.size()));  // the header's own checksum
        out.seekp(0);
        out.write(header.data.data(), std::streamsize(header.data.size()));
    });
}

// ---------------------------------------------------------------- open

std::unique_ptr<DiskIndex> DiskIndex::open(const std::string& path, const DiskOpenOptions& options) {
    std::unique_ptr<DiskIndex> d(new DiskIndex());
    d->file_ = std::make_unique<BlockFile>(path);
    d->file_->set_simulated_latency(options.simulated_latency);
    const BlockFile& f = *d->file_;
    if (f.size() < kBlock) throw FormatError("not a disk index (too short)");
    std::vector<unsigned char> h(kBlock);
    f.read(0, h.data(), kBlock);
    if (std::memcmp(h.data(), kMagic, 8) != 0) throw FormatError("not a disk index (wrong magic)");
    if (load_le<std::uint32_t>(&h[8]) != kDiskVersion) throw FormatError("unsupported disk index version");
    if (load_le<std::uint32_t>(&h[12]) != kByteOrder) throw FormatError("wrong byte order mark");
    const std::size_t header_size = 8 + 4 + 4 + 8 + 4 + 4 + 8 * 7 + kSections * 20;
    if (load_le<std::uint32_t>(&h[header_size]) != persist::crc32(0, h.data(), header_size)) throw FormatError("header checksum mismatch");
    const auto dim = load_le<std::uint64_t>(&h[16]);
    const auto metric = load_le<std::uint32_t>(&h[24]);
    const auto n = load_le<std::uint64_t>(&h[32]);
    const auto R = load_le<std::uint64_t>(&h[40]);
    const auto medoid = load_le<std::uint64_t>(&h[48]);
    if (dim == 0 || dim > (1u << 20) || R == 0 || R > (1u << 16) || n == 0 || n >= kNoNeighbor || medoid >= n)
        throw FormatError("invalid disk index sizes");
    if (metric != std::uint32_t(Metric::L2) && metric != std::uint32_t(Metric::Cosine)) throw FormatError("invalid metric");
    const Layout layout(dim, R, n);
    if (load_le<std::uint64_t>(&h[56]) != layout.node_bytes || load_le<std::uint64_t>(&h[64]) != layout.nodes_per_block ||
        load_le<std::uint64_t>(&h[72]) != layout.blocks_per_node || load_le<std::uint64_t>(&h[80]) != layout.node_blocks)
        throw FormatError("inconsistent node layout");
    d->dim_ = dim;
    d->R_ = R;
    d->metric_ = static_cast<Metric>(metric);
    d->distance_ = get_distance(d->metric_);
    d->medoid_ = static_cast<NodeId>(medoid);
    d->node_bytes_ = layout.node_bytes;
    d->nodes_per_block_ = layout.nodes_per_block;
    d->blocks_per_node_ = layout.blocks_per_node;
    // Sections: in order, right after the node blocks, ending the file.
    std::vector<unsigned char> sec[kSections];
    std::uint64_t expected = kBlock * (1 + layout.node_blocks);
    for (std::size_t s = 0; s < std::size_t(kSections); ++s) {
        const unsigned char* e = &h[88 + 20 * s];
        const auto offset = load_le<std::uint64_t>(e), size = load_le<std::uint64_t>(e + 8);
        if (offset != expected || size > f.size() || offset > f.size() - size) throw FormatError("invalid section offsets");
        sec[s].resize(static_cast<std::size_t>(size));
        f.read(offset, sec[s].data(), sec[s].size());
        if (persist::crc32(0, sec[s].data(), sec[s].size()) != load_le<std::uint32_t>(e + 16)) throw FormatError("section checksum mismatch");
        expected = offset + size;
    }
    if (expected != f.size()) throw FormatError("unexpected data at the end of the file");
    // Block checksums.
    if (sec[kBlockCrc].size() != 4 * layout.node_blocks) throw FormatError("wrong number of block checksums");
    d->block_crc_.resize(static_cast<std::size_t>(layout.node_blocks));
    for (std::size_t b = 0; b < d->block_crc_.size(); ++b) d->block_crc_[b] = load_le<std::uint32_t>(&sec[kBlockCrc][4 * b]);
    // PQ codebooks and codes.
    {
        const auto& p = sec[kPq];
        if (p.size() < 16) throw FormatError("PQ section too short");
        const auto parts = load_le<std::uint64_t>(&p[0]), state_size = load_le<std::uint64_t>(&p[8]);
        if (parts == 0 || parts > dim || state_size > (p.size() - 16) / 4) throw FormatError("invalid PQ section");
        std::vector<float> state(static_cast<std::size_t>(state_size));
        for (std::size_t i = 0; i < state.size(); ++i) state[i] = load_le<float>(&p[16 + 4 * i]);
        try {
            d->codec_ = std::make_unique<PQCodec>(dim, d->metric_, parts);
            d->codec_->restore(state);
        } catch (const std::invalid_argument& e) {
            throw FormatError(std::string("invalid PQ codebooks: ") + e.what());
        }
        const std::size_t at = 16 + 4 * state.size();
        if (p.size() - at != n * parts) throw FormatError("wrong number of PQ codes");
        d->codes_.assign(p.begin() + std::ptrdiff_t(at), p.end());
        for (std::uint8_t c : d->codes_)
            if (c >= d->codec_->centroids()) throw FormatError("a PQ code refers to a centroid that does not exist");
    }
    // IDs.
    if (sec[kIds].size() != 8 * n) throw FormatError("wrong number of IDs");
    d->ids_.resize(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < n; ++i) {
        d->ids_[i] = load_le<std::uint64_t>(&sec[kIds][8 * i]);
        if (!d->node_of_.emplace(d->ids_[i], static_cast<NodeId>(i)).second) throw FormatError("repeated ID");
        d->by_id_.emplace_back(d->ids_[i], static_cast<NodeId>(i));
    }
    std::sort(d->by_id_.begin(), d->by_id_.end());
    if (options.io_threads > 0) d->io_ = std::make_unique<detail::IoPool>(options.io_threads);
    // Metadata, in the usual section format.
    try {
        std::istringstream in(std::string(reinterpret_cast<const char*>(sec[kMeta].data()), sec[kMeta].size()));
        persist::Reader r(in);
        r.begin_section(persist::tag("META"), "META");
        d->metadata_ = MetadataStore::load(r, static_cast<std::size_t>(n), [](NodeId) { return true; });
        r.end_section();
    } catch (const FormatError&) {
        throw;
    } catch (const std::exception& e) {
        throw FormatError(std::string("invalid metadata: ") + e.what());
    }
    return d;
}

// ---------------------------------------------------------------- nodes

std::optional<NodeId> DiskIndex::node_of(std::uint64_t id) const {
    const auto it = node_of_.find(id);
    if (it == node_of_.end()) return std::nullopt;
    return it->second;
}

std::uint64_t DiskIndex::node_block(NodeId node) const {
    return nodes_per_block_ > 0 ? node / nodes_per_block_ : std::uint64_t(node) * blocks_per_node_;
}

std::uint64_t DiskIndex::node_offset(NodeId node) const {
    if (nodes_per_block_ > 0) return kBlock * (1 + node_block(node)) + std::uint64_t(node % nodes_per_block_) * node_bytes_;
    return kBlock * (1 + node_block(node));
}

DiskIndex::Node DiskIndex::read_node(NodeId node) const {
    if (node >= size()) throw std::out_of_range("DiskIndex::read_node: unknown node");
    const std::uint64_t first = node_block(node);
    std::vector<unsigned char> buf(blocks_per_node_ * kBlock);
    file_->read(kBlock * (1 + first), buf.data(), buf.size());
    blocks_read_.fetch_add(blocks_per_node_, std::memory_order_relaxed);
    for (std::size_t b = 0; b < blocks_per_node_; ++b)
        if (persist::crc32(0, buf.data() + b * kBlock, kBlock) != block_crc_[first + b]) throw FormatError("node block checksum mismatch");
    const unsigned char* r = buf.data() + (node_offset(node) - kBlock * (1 + first));
    Node out;
    out.vector.resize(dim_);
    for (std::size_t j = 0; j < dim_; ++j) out.vector[j] = load_le<float>(r + 4 * j);
    const auto degree = load_le<std::uint32_t>(r + 4 * dim_);
    if (degree > R_) throw FormatError("invalid node degree");
    for (std::size_t s = 0; s < degree; ++s) {
        const auto nb = load_le<std::uint32_t>(r + 4 * dim_ + 4 + 4 * s);
        if (nb >= size()) throw FormatError("a neighbor does not exist");
        out.neighbors.push_back(nb);
    }
    return out;
}

// ---------------------------------------------------------------- search

std::vector<SearchResult> DiskIndex::search(std::span<const float> query, std::size_t k, const DiskSearchParams& params,
                                            const SearchOptions& options, DiskSearchStats* stats) const {
    if (params.L < 1 || params.beam_width < 1) throw std::invalid_argument("DiskSearchParams: L and beam_width must be at least 1");
    const detail::SearchStopScope stop(options, options.stats);  // deadline and cancel flag
    PreparedVector q(dim_);
    q.prepare(query, metric_);  // checks the query; normalizes for cosine
    std::optional<CompiledFilter> compiled;
    if (options.filter) compiled.emplace(*options.filter, metadata_);  // throws for an unknown field under a strict schema
    auto eligible = [&](NodeId n) {
        return (!compiled || compiled->matches(n)) && (!options.predicate || options.predicate(ids_[n]));
    };
    DiskSearchStats st;
    std::vector<std::pair<float, NodeId>> exact;  // nodes read, with exact distances
    std::size_t matches = 0;                      // of which eligible
    PreparedVector row(dim_);
    // Reads a round's nodes: concurrently through the I/O pool if there is one,
    // else one after another. Every read is waited for before an error is passed
    // on, so when search returns (even by throwing) none of its reads is still
    // running; memory safety does not depend on it (destroying the index first
    // destroys the pool, which finishes its reads while the file is open). Nodes
    // are returned in order, so results never depend on timing.
    auto fetch = [&](const std::vector<NodeId>& nodes) {
        std::vector<Node> out;
        out.reserve(nodes.size());
        if (!io_ || nodes.size() < 2) {
            for (NodeId n : nodes) out.push_back(read_node(n));
            return out;
        }
        std::vector<std::future<Node>> pending;
        pending.reserve(nodes.size());
        for (NodeId n : nodes) pending.push_back(io_->submit(std::packaged_task<Node()>([this, n] { return read_node(n); })));
        for (auto& f : pending) f.wait();
        for (auto& f : pending) out.push_back(f.get());  // the first error, if any, is thrown here
        return out;
    };
    auto account = [&](NodeId n, const Node& node) {
        ++st.nodes_read;
        st.blocks_read += blocks_per_node_;
        row.prepare(node.vector, Metric::L2);  // already prepared: copied and padded
        exact.emplace_back(index_distance(distance_(q.data(), row.data(), q.stride()), metric_ == Metric::Cosine), n);
        if (eligible(n)) ++matches;
    };
    const std::size_t L0 = std::max(params.L, k);
    bool direct = false;
    if (k > 0 && compiled && !options.predicate) {
        // A selective filter: read its matches directly (exact) instead of searching the graph.
        std::vector<NodeId> matching;
        for (NodeId n = 0; n < size() && matching.size() <= L0; ++n)
            if (compiled->matches(n)) matching.push_back(n);
        if (matching.size() <= L0) {
            direct = true;
            for (std::size_t i = 0; i < matching.size(); i += params.beam_width) {
                if (detail::search_should_stop()) {
                    st.stopped_early = true;
                    break;
                }
                ++st.rounds;
                const std::vector<NodeId> round(matching.begin() + std::ptrdiff_t(i),
                                                matching.begin() + std::ptrdiff_t(std::min(matching.size(), i + params.beam_width)));
                const auto nodes = fetch(round);
                for (std::size_t j = 0; j < round.size(); ++j) account(round[j], nodes[j]);
            }
        }
    }
    if (k > 0 && !direct) {
        std::vector<float> table;
        codec_->prepare(q.values(), table);
        const std::size_t cs = codec_->code_size();
        auto approx = [&](NodeId n) { return codec_->distance(table, codes_.data() + std::size_t(n) * cs); };
        struct Pending {
            float d;
            NodeId node;
            bool read;
        };
        // Sorted by PQ distance and never truncated: if fewer than k matching
        // vectors turn up, L widens and the search continues from here.
        std::vector<Pending> candidates{{approx(medoid_), medoid_, false}};
        std::unordered_set<NodeId> seen{medoid_};
        std::size_t L = L0;
        std::vector<NodeId> batch;
        for (;;) {
            if (detail::search_should_stop()) {
                st.stopped_early = true;
                break;
            }
            batch.clear();  // the W best unread among the first L candidates
            for (std::size_t i = 0; i < candidates.size() && i < L && batch.size() < params.beam_width; ++i)
                if (!candidates[i].read) {
                    candidates[i].read = true;
                    batch.push_back(candidates[i].node);
                }
            if (batch.empty()) {
                if (matches >= k || candidates.size() <= L) break;  // enough, or everything reachable was read
                L *= 2;
                continue;
            }
            ++st.rounds;
            const auto nodes = fetch(batch);
            for (std::size_t b = 0; b < batch.size(); ++b) {
                const Node& node = nodes[b];
                account(batch[b], node);
                for (NodeId nb : node.neighbors) {
                    if (!seen.insert(nb).second) continue;
                    const float d = approx(nb);
                    const auto at = std::upper_bound(candidates.begin(), candidates.end(), d,
                                                     [](float x, const Pending& c) { return x < c.d; });
                    candidates.insert(at, {d, nb, false});
                }
            }
        }
    }
    std::sort(exact.begin(), exact.end());
    std::vector<SearchResult> results;
    for (std::size_t i = 0; i < exact.size() && results.size() < k; ++i)
        if (eligible(exact[i].second)) results.push_back({ids_[exact[i].second], exact[i].first});
    if (options.stats) options.stats->nodes_visited = st.nodes_read;
    if (stats) *stats = st;
    return results;
}

std::vector<std::vector<SearchResult>> DiskIndex::search_batch(std::span<const float> queries, std::size_t k,
                                                               const DiskSearchParams& params, std::size_t threads,
                                                               const SearchOptions& options) const {
    if (queries.size() % dim_ != 0) throw std::invalid_argument("DiskIndex::search_batch: queries must be whole vectors");
    const std::size_t nq = queries.size() / dim_;
    std::vector<std::vector<SearchResult>> results(nq);
    SearchOptions each = options;
    each.stats = nullptr;  // one SearchStats cannot be written from several threads
    std::atomic<bool> stopped{false};
    auto one = [&](std::size_t i) {
        DiskSearchStats st;
        results[i] = search(queries.subspan(i * dim_, dim_), k, params, each, &st);
        if (st.stopped_early) stopped = true;
    };
    const std::size_t workers = std::min(effective_threads(threads), nq);
    if (workers <= 1) {
        for (std::size_t i = 0; i < nq; ++i) one(i);
    } else {
        ThreadPool pool(workers);
        pool.parallel_for(nq, one);
    }
    if (options.stats) options.stats->stopped_early = stopped.load();
    return results;
}

std::optional<std::vector<SearchResult>> DiskIndex::search_by_id(std::uint64_t id, std::size_t k, const DiskSearchParams& params,
                                                                 const SearchOptions& options, bool include_self) const {
    const auto node = node_of(id);
    if (!node) return std::nullopt;
    const std::vector<float> v = read_node(*node).vector;
    if (include_self || k == 0) return search(v, k, params, options);
    auto results = search(v, k == SIZE_MAX ? k : k + 1, params, options);  // one more, in case the record itself is among them
    std::erase_if(results, [&](const SearchResult& r) { return r.id == id; });
    if (results.size() > k) results.resize(k);
    return results;
}

std::optional<std::vector<float>> DiskIndex::get_vector(std::uint64_t id) const {
    const auto node = node_of(id);
    if (!node) return std::nullopt;
    return read_node(*node).vector;
}

std::optional<Record> DiskIndex::get(std::uint64_t id) const {
    const auto node = node_of(id);
    if (!node) return std::nullopt;
    return Record{id, read_node(*node).vector, metadata_.read(*node)};
}

Page DiskIndex::scan(const ScanOptions& options) const {
    if (options.limit == 0) throw std::invalid_argument("scan: limit must be at least 1");
    std::optional<CompiledFilter> compiled;
    if (options.filter) compiled.emplace(*options.filter, metadata_);
    auto it = options.after ? std::upper_bound(by_id_.begin(), by_id_.end(), std::make_pair(*options.after, kEmpty)) : by_id_.begin();
    Page page;
    for (; it != by_id_.end(); ++it) {
        if (compiled && !compiled->matches(it->second)) continue;
        if (page.records.size() == options.limit) {  // one more matches: there is a next page
            page.next = page.records.back().id;
            break;
        }
        Record r;
        r.id = it->first;
        if (options.with_vectors) r.vector = read_node(it->second).vector;
        if (options.with_metadata) r.metadata = metadata_.read(it->second);
        page.records.push_back(std::move(r));
    }
    return page;
}

}  // namespace vecdb