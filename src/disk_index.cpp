/**
 * @file disk_index.cpp
 * @brief Building, opening and reading a disk-based index; see disk_index.h.
 */
#include "disk_index.h"
#include "half.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <functional>
#include <chrono>
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

#include "kmeans.h"
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
    // FILE_SHARE_DELETE: as on POSIX, the file can be deleted or replaced (an
    // index rebuilt in place) while open; this handle keeps reading the old one.
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
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
    if (::fstat(fd, &st) != 0) {  // GCOVR_EXCL_START: fstat of a file that just opened does not fail
        ::close(fd);
        throw std::runtime_error("cannot read the size of " + path);
    }  // GCOVR_EXCL_STOP
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
    if (latency_.count() > 0) {  // simulated slow storage: at least latency_, on every platform
        // sleep_for may wake early as steady_clock measures it (MinGW sleeps with the coarse
        // Windows timer while steady_clock reads the precise performance counter): finish
        // the wait against steady_clock itself.
        const auto until = std::chrono::steady_clock::now() + latency_;
        std::this_thread::sleep_until(until);
        while (std::chrono::steady_clock::now() < until) std::this_thread::yield();
    }
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
constexpr std::uint32_t kDiskVersion = 1;      // float32 vectors
constexpr std::uint32_t kDiskVersionHalf = 2;  // half-precision vectors (DiskVectorFormat in the reserved field)
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
    Layout(std::size_t dim, std::size_t R, std::uint64_t n, DiskVectorFormat format) {
        node_bytes = (format == DiskVectorFormat::Float32 ? 4 : 2) * dim + 4 + 4 * R;
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

/// Moves `from` over `to`, replacing it. As on POSIX, `to` may be open (a
/// DiskIndex reading it keeps reading the old file). On Windows, in order:
/// a rename with POSIX semantics (atomic; Windows 10 or later on NTFS); the
/// classic MoveFileEx (when `to` is not open); or moving the open `to` aside
/// (allowed: BlockFile opens with FILE_SHARE_DELETE), moving `from` in and
/// deleting the old file, which disappears once its last reader closes (not
/// atomic: for a moment `to` does not exist).
void replace_file(const std::string& from, const std::string& to) {
#ifdef _WIN32
    const std::wstring wfrom = std::filesystem::absolute(from).wstring(), wto = std::filesystem::absolute(to).wstring();
    {
        // FILE_RENAME_INFO with Flags (newer SDKs make it a union with
        // ReplaceIfExists; this layout is the same), class FileRenameInfoEx (22).
        struct RenameInfo {
            DWORD flags;
            HANDLE root;
            DWORD length;
            WCHAR name[1];
        };
        static_assert(offsetof(RenameInfo, root) == offsetof(FILE_RENAME_INFO, RootDirectory) &&
                          offsetof(RenameInfo, length) == offsetof(FILE_RENAME_INFO, FileNameLength) &&
                          offsetof(RenameInfo, name) == offsetof(FILE_RENAME_INFO, FileName),
                      "RenameInfo must match FILE_RENAME_INFO");
        HANDLE h = CreateFileW(wfrom.c_str(), DELETE | SYNCHRONIZE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            std::vector<char> buf(sizeof(RenameInfo) + wto.size() * sizeof(WCHAR), 0);
            auto* info = reinterpret_cast<RenameInfo*>(buf.data());
            info->flags = 0x1 | 0x2;  // FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS
            info->root = nullptr;
            info->length = static_cast<DWORD>(wto.size() * sizeof(WCHAR));
            std::memcpy(info->name, wto.c_str(), wto.size() * sizeof(WCHAR));
            const BOOL ok = SetFileInformationByHandle(h, static_cast<FILE_INFO_BY_HANDLE_CLASS>(22), info, static_cast<DWORD>(buf.size()));
            CloseHandle(h);
            if (ok) return;
        }
    }
    if (MoveFileExW(wfrom.c_str(), wto.c_str(), MOVEFILE_REPLACE_EXISTING)) return;
    const std::wstring aside = wto + L".old-" + std::to_wstring(std::random_device{}());
    if (MoveFileExW(wto.c_str(), aside.c_str(), 0)) {
        if (MoveFileExW(wfrom.c_str(), wto.c_str(), 0)) {
            DeleteFileW(aside.c_str());  // pending until its readers close
            return;
        }
        MoveFileExW(aside.c_str(), wto.c_str(), 0);  // put the old file back
    }
    throw std::runtime_error("cannot replace " + to);
#else
    std::filesystem::rename(from, to);
#endif
}

/// Writes `fill` to a temporary file next to `path`, then moves it into place
/// (an existing file is replaced only once the new one is complete, even while
/// a DiskIndex has it open).
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
            if (!out) throw std::runtime_error("cannot write " + tmp);  // GCOVR_EXCL_LINE: needs a failing or full disk
        }
        replace_file(tmp, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);
        throw;
    }
}

}  // namespace

// ---------------------------------------------------------------- build

namespace {

/// Supplies node `i` (called for 0, 1, 2, ... in order): its stored vector and neighbors.
using NodeSource = std::function<void(NodeId, std::vector<float>&, std::vector<NodeId>&)>;

/// Writes a complete index file (see the layout in disk_index.h), replacing
/// `path` only once complete. Every builder goes through here, so every builder
/// writes the same format.
void write_index(const std::string& path, std::size_t dim, Metric metric, std::size_t R, NodeId medoid, std::size_t n,
                 DiskVectorFormat format, const NodeSource& next, const PQCodec& pq, const std::vector<std::uint8_t>& codes,
                 std::span<const std::uint64_t> ids, const MetadataStore& store) {
    const Layout layout(dim, R, n, format);
    write_atomically(path, [&](std::ofstream& out) {
        std::vector<std::uint32_t> crcs;
        std::vector<float> vec;
        std::vector<NodeId> nbs;
        auto record = [&](NodeId i, char* at) {
            next(i, vec, nbs);
            Bytes r;
            if (format == DiskVectorFormat::Float32) {
                r.put_floats(vec.data(), dim);
            } else {
                for (std::size_t j = 0; j < dim; ++j) {
                    if (format == DiskVectorFormat::Float16 && std::fabs(vec[j]) > kF16Max)  // L2 values only: cosine ones are at most 1
                        throw std::invalid_argument("DiskIndex: float16 holds values up to 65504 in magnitude; use DiskVectorFormat::BFloat16");
                    r.put<std::uint16_t>(format == DiskVectorFormat::Float16 ? float_to_f16(vec[j]) : float_to_bf16(vec[j]));
                }
            }
            r.put<std::uint32_t>(static_cast<std::uint32_t>(nbs.size()));
            for (std::size_t s = 0; s < R; ++s) r.put<std::uint32_t>(s < nbs.size() ? nbs[s] : kNoNeighbor);
            std::memcpy(at, r.data.data(), r.data.size());
        };
        out.write(std::string(DiskIndex::kBlock, '\0').data(), DiskIndex::kBlock);  // the header, written last
        if (layout.nodes_per_block > 0) {
            std::string block(DiskIndex::kBlock, '\0');
            for (std::uint64_t b = 0; b < layout.node_blocks; ++b) {
                std::fill(block.begin(), block.end(), '\0');
                for (std::size_t s = 0; s < layout.nodes_per_block; ++s) {
                    const std::uint64_t i = b * layout.nodes_per_block + s;
                    if (i < n) record(static_cast<NodeId>(i), block.data() + s * layout.node_bytes);
                }
                crcs.push_back(persist::crc32(0, block.data(), DiskIndex::kBlock));
                out.write(block.data(), DiskIndex::kBlock);
            }
        } else {
            std::string big(layout.blocks_per_node * DiskIndex::kBlock, '\0');
            for (NodeId i = 0; i < n; ++i) {
                std::fill(big.begin(), big.end(), '\0');
                record(i, big.data());
                for (std::size_t b = 0; b < layout.blocks_per_node; ++b)
                    crcs.push_back(persist::crc32(0, big.data() + b * DiskIndex::kBlock, DiskIndex::kBlock));
                out.write(big.data(), std::streamsize(big.size()));
            }
        }
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
        std::uint64_t offset = DiskIndex::kBlock * (1 + layout.node_blocks);
        Bytes header;
        header.data.append(kMagic, 8);
        // Float32 files stay version 1, byte for byte as before; half precision is version 2.
        header.put<std::uint32_t>(format == DiskVectorFormat::Float32 ? kDiskVersion : kDiskVersionHalf);
        header.put<std::uint32_t>(kByteOrder);
        header.put<std::uint64_t>(dim);
        header.put<std::uint32_t>(static_cast<std::uint32_t>(metric));
        header.put<std::uint32_t>(static_cast<std::uint32_t>(format));  // reserved (0) in version 1
        header.put<std::uint64_t>(n);
        header.put<std::uint64_t>(R);
        header.put<std::uint64_t>(medoid);
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

/// Checks IDs (unique) and metadata (validated) and fills a metadata store.
MetadataStore checked_metadata(std::span<const std::uint64_t> ids, std::span<const Metadata> metadata, Schema schema) {
    if (!metadata.empty() && metadata.size() != ids.size()) throw std::invalid_argument("DiskIndex: one metadata entry per vector");
    std::unordered_set<std::uint64_t> seen;
    for (std::uint64_t id : ids)
        if (!seen.insert(id).second) throw std::invalid_argument("DiskIndex: repeated ID");
    MetadataStore store(std::move(schema));
    for (const Metadata& m : metadata) store.validate(m);
    for (std::size_t i = 0; i < ids.size(); ++i) (void)store.write(static_cast<NodeId>(i), metadata.empty() ? Metadata() : metadata[i]);
    return store;
}

std::size_t pq_parts(const DiskIndexParams& params, std::size_t dim) {
    return params.pq_subspaces == 0 ? std::max<std::size_t>(1, dim / 2) : params.pq_subspaces;
}

}  // namespace

void DiskIndex::build(const std::string& path, std::span<const std::uint64_t> ids, std::span<const float> vectors, std::size_t dim,
                      Metric metric, const DiskIndexParams& params, std::span<const Metadata> metadata, Schema schema) {
    if (dim == 0 || vectors.size() != ids.size() * dim || ids.empty())
        throw std::invalid_argument("DiskIndex::build: one vector of dim values per ID, at least one");
    const MetadataStore store = checked_metadata(ids, metadata, std::move(schema));
    const VamanaGraph g = VamanaGraph::build(vectors, dim, metric, params.graph);  // checks the vectors and settings
    const std::size_t n = g.size();
    PQCodec pq(dim, metric, pq_parts(params, dim));
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
    write_index(path, dim, metric, params.graph.R, g.medoid(), n, params.vectors,
                [&](NodeId i, std::vector<float>& v, std::vector<NodeId>& nbs) {
                    const auto x = g.vector(i);
                    v.assign(x.begin(), x.end());
                    const auto ns = g.neighbors(i);
                    nbs.assign(ns.begin(), ns.end());
                },
                pq, codes, ids, store);
}

// ---------------------------------------------------------------- build from a file

void write_vectors_file(const std::string& path, std::span<const float> vectors, std::size_t dim) {
    if (dim == 0 || vectors.size() % dim != 0) throw std::invalid_argument("write_vectors_file: whole vectors of dim values");
    if (vectors.size() / dim > 0xFFFFFFFFull || dim > 0xFFFFFFFFull)  // GCOVR_EXCL_BR_LINE: needs 16 GB of floats
        throw std::invalid_argument("write_vectors_file: at most 2^32 - 1 vectors and dimensions");  // GCOVR_EXCL_LINE
    write_atomically(path, [&](std::ofstream& out) {
        Bytes b;
        b.put<std::uint32_t>(static_cast<std::uint32_t>(vectors.size() / dim));
        b.put<std::uint32_t>(static_cast<std::uint32_t>(dim));
        b.put_floats(vectors.data(), vectors.size());
        out.write(b.data.data(), std::streamsize(b.data.size()));
    });
}

void DiskIndex::build_from_file(const std::string& path, const std::string& vectors_path, Metric metric, const LargeBuildParams& params,
                                std::span<const std::uint64_t> ids_in, std::span<const Metadata> metadata, Schema schema,
                                LargeBuildStats* stats) {
    validate(params.index.graph);
    if (metric != Metric::L2 && metric != Metric::Cosine) throw std::invalid_argument("DiskIndex: L2 and cosine only");
    if (params.replicas < 1) throw std::invalid_argument("LargeBuildParams::replicas must be at least 1");
    BlockFile src(vectors_path);
    unsigned char head[8];
    if (src.size() < 8) throw std::invalid_argument("not a vectors file (too short)");
    src.read(0, head, 8);
    const std::size_t n = load_le<std::uint32_t>(head), dim = load_le<std::uint32_t>(head + 4);
    if (n == 0 || dim == 0 || src.size() != 8 + std::uint64_t(n) * dim * 4)
        throw std::invalid_argument("not a vectors file (its size does not match its count and dimension)");
    std::vector<std::uint64_t> own_ids;
    if (ids_in.empty()) {
        own_ids.resize(n);
        for (std::size_t i = 0; i < n; ++i) own_ids[i] = i;
    } else if (ids_in.size() != n) {
        throw std::invalid_argument("DiskIndex::build_from_file: one ID per vector");
    }
    const std::span<const std::uint64_t> ids = ids_in.empty() ? std::span<const std::uint64_t>(own_ids) : ids_in;
    auto read_raw = [&](std::size_t i, float* out) {  // one vector as stored in the file
        std::vector<unsigned char> buf(dim * 4);
        src.read(8 + std::uint64_t(i) * dim * 4, buf.data(), buf.size());
        for (std::size_t j = 0; j < dim; ++j) out[j] = load_le<float>(&buf[4 * j]);
    };
    const std::size_t R = params.index.graph.R;
    const std::size_t stride = PreparedVector(dim).stride();
    const std::size_t per_vector = 4 * stride + 8 * R + 96;  // vectors, graph and bookkeeping while building a partition
    const std::size_t cap = params.memory_budget / per_vector;
    if (cap < std::max<std::size_t>(R + 2, 64)) throw std::invalid_argument("LargeBuildParams::memory_budget is too small for this dimension and R");

    LargeBuildStats st;
    if (n <= cap && params.partitions <= 1) {  // fits: one in-memory build
        std::vector<float> all(n * dim);
        for (std::size_t i = 0; i < n; ++i) read_raw(i, all.data() + i * dim);
        build(path, ids, all, dim, metric, params.index, metadata, std::move(schema));
        st.partitions = 1;
        st.largest_partition = n;
        if (stats) *stats = st;
        return;
    }
    const MetadataStore store = checked_metadata(ids, metadata, std::move(schema));

    // Temporary files, all in one directory, removed whatever happens.
    const std::filesystem::path tmp = path + ".build-" + std::to_string(std::random_device{}());
    std::filesystem::create_directories(tmp);
    struct Cleanup {
        std::filesystem::path dir;
        ~Cleanup() {
            std::error_code ignored;
            std::filesystem::remove_all(dir, ignored);
        }
    } cleanup{tmp};

    // 1. A sample: partition centroids (k-means), the PQ codebooks, the medoid.
    // Every vector needs `replicas` different partitions with room. Fewer than
    // n * replicas / cap partitions can ever be full, so P >= that + replicas
    // guarantees it: no partition ever exceeds the budget. Automatically, also
    // two thirds of slack: fewer vectors pushed out of their nearest partitions
    // (measured: recall@10 0.952 with 25% slack, 1.000 with two thirds).
    const std::size_t replicas = params.replicas;
    if (replicas < 2) throw std::invalid_argument("LargeBuildParams::replicas must be at least 2 with several partitions (partitions that share no vectors cannot connect)");
    const std::size_t needed = (n * replicas + cap - 1) / cap + replicas;
    if (params.partitions > 0 && params.partitions < needed)
        throw std::invalid_argument("LargeBuildParams::partitions is too few for the memory budget (at least " + std::to_string(needed) + ")");
    const std::size_t P = params.partitions > 0 ? params.partitions : std::max(needed, (n * replicas * 5 / 3 + cap - 1) / cap);
    const std::size_t take = std::min(n, std::min(cap, std::max<std::size_t>(P * 64, 20000)));
    PreparedVector pv(dim);
    std::vector<float> raw(dim), sample(take * dim);
    std::vector<std::size_t> sample_index(take);
    for (std::size_t s = 0; s < take; ++s) {
        sample_index[s] = s * n / take;
        read_raw(sample_index[s], raw.data());
        pv.prepare(raw, metric);  // checks the values; normalizes for cosine
        std::copy_n(pv.values().begin(), dim, sample.begin() + std::ptrdiff_t(s * dim));
    }
    if (take < P) throw std::invalid_argument("LargeBuildParams: more partitions than the sample can train");  // GCOVR_EXCL_LINE: P <= n / (cap / replicas) is far below the sample
    const KMeansResult parts = kmeans(sample, dim, KMeansParams{P, 20, params.index.graph.seed, 0});
    const std::size_t partitions = parts.centroids.size() / dim;
    PQCodec pq(dim, metric, pq_parts(params.index, dim));
    pq.train(sample);
    NodeId medoid = 0;
    {
        std::vector<double> mean(dim, 0.0);
        for (std::size_t s = 0; s < take; ++s)
            for (std::size_t j = 0; j < dim; ++j) mean[j] += sample[s * dim + j];
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t s = 0; s < take; ++s) {
            double d = 0;
            for (std::size_t j = 0; j < dim; ++j) d += (sample[s * dim + j] - mean[j] / double(take)) * (sample[s * dim + j] - mean[j] / double(take));
            if (d < best) {
                best = d;
                medoid = static_cast<NodeId>(sample_index[s]);
            }
        }
    }

    // 2. One pass: each vector joins its `replicas` nearest partitions that are not
    // full (k-means does not balance partitions; a full one would exceed the
    // budget), and gets its PQ code. P allows 25% slack, so there is room.
    std::vector<std::uint8_t> codes(n * pq.code_size());
    std::vector<std::ofstream> members(partitions);
    std::vector<std::size_t> sizes(partitions, 0);
    for (std::size_t p = 0; p < partitions; ++p) {
        members[p].open(tmp / ("members-" + std::to_string(p)), std::ios::binary);
        if (!members[p]) throw std::runtime_error("cannot create a temporary file in " + tmp.string());  // GCOVR_EXCL_LINE: the directory was just created
    }
    std::vector<std::pair<double, std::size_t>> nearest(partitions);  // not `near`: <windows.h> defines near and far as macros
    for (std::size_t i = 0; i < n; ++i) {
        read_raw(i, raw.data());
        pv.prepare(raw, metric);
        const auto v = pv.values();
        pq.encode(v, codes.data() + i * pq.code_size());
        for (std::size_t p = 0; p < partitions; ++p) {
            double d = 0;
            for (std::size_t j = 0; j < dim; ++j) d += (double(v[j]) - parts.centroids[p * dim + j]) * (double(v[j]) - parts.centroids[p * dim + j]);
            nearest[p] = {d, p};
        }
        std::sort(nearest.begin(), nearest.end());
        std::size_t joined = 0;
        for (std::size_t r = 0; r < partitions && joined < replicas; ++r) {
            const std::size_t p = nearest[r].second;
            if (sizes[p] >= cap) continue;  // full: the next nearest
            unsigned char b[4];
            persist::to_little_endian(static_cast<std::uint32_t>(i), b);
            members[p].write(reinterpret_cast<const char*>(b), 4);
            ++sizes[p];
            ++joined;
        }
        if (joined < replicas) throw std::logic_error("build_from_file: no partition with room");  // GCOVR_EXCL_LINE: impossible, see P
    }
    for (auto& m : members) {
        m.close();
        if (!m) throw std::runtime_error("cannot write a temporary file in " + tmp.string());  // GCOVR_EXCL_LINE: needs a failing or full disk
    }

    // 3. A Vamana graph per partition, built in memory one at a time, written in ID order.
    for (std::size_t p = 0; p < partitions; ++p) {
        st.largest_partition = std::max(st.largest_partition, sizes[p]);
        if (sizes[p] == 0) continue;
        std::vector<std::uint32_t> ids_of(sizes[p]);
        {
            std::ifstream in(tmp / ("members-" + std::to_string(p)), std::ios::binary);
            std::vector<unsigned char> buf(4 * sizes[p]);
            in.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size()));
            for (std::size_t j = 0; j < sizes[p]; ++j) ids_of[j] = load_le<std::uint32_t>(&buf[4 * j]);
        }
        std::vector<float> vectors(sizes[p] * dim);
        for (std::size_t j = 0; j < sizes[p]; ++j) read_raw(ids_of[j], vectors.data() + j * dim);  // ascending: mostly sequential
        VamanaParams gp = params.index.graph;
        gp.seed += p;
        const VamanaGraph g = VamanaGraph::build(vectors, dim, metric, gp);
        std::ofstream out(tmp / ("graph-" + std::to_string(p)), std::ios::binary);
        Bytes rec;
        for (std::size_t j = 0; j < sizes[p]; ++j) {
            rec.data.clear();
            rec.put<std::uint32_t>(ids_of[j]);
            const auto nbs = g.neighbors(static_cast<NodeId>(j));
            rec.put<std::uint32_t>(static_cast<std::uint32_t>(nbs.size()));
            for (NodeId q : nbs) rec.put<std::uint32_t>(ids_of[q]);
            out.write(rec.data.data(), std::streamsize(rec.data.size()));
        }
        if (!out) throw std::runtime_error("cannot write a temporary file in " + tmp.string());  // GCOVR_EXCL_LINE: needs a failing or full disk
    }
    st.partitions = partitions;

    // 4. Merge and write, one pass in ID order: each vector's lists from its
    // partitions, joined and, beyond R, pruned again with distances from the PQ
    // codes held in memory (no random reads).
    struct Reader {
        std::ifstream in;
        std::uint32_t id = 0, degree = 0;
        bool more = false;
        void advance() {
            unsigned char b[8];
            more = bool(in.read(reinterpret_cast<char*>(b), 8));
            if (more) {
                id = load_le<std::uint32_t>(b);
                degree = load_le<std::uint32_t>(b + 4);
            }
        }
    };
    std::vector<Reader> readers(partitions);
    for (std::size_t p = 0; p < partitions; ++p) {
        readers[p].in.open(tmp / ("graph-" + std::to_string(p)), std::ios::binary);
        readers[p].advance();
    }
    std::vector<float> decoded(dim), other(dim), table;
    const float alpha = params.index.graph.alpha;
    auto code_of = [&](std::size_t i) { return codes.data() + i * pq.code_size(); };
    auto tie = [](NodeId i, NodeId q) {  // equal distances: by a hash of the pair, not by node number
        std::uint64_t z = (std::uint64_t(i) << 32 | q) + 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };
    // 4a. Merge, in ID order, into an adjacency table (R numbers per vector,
    // held in memory like the PQ codes): each vector's lists from its partitions,
    // joined and, beyond R, pruned again with distances from the PQ codes.
    std::vector<std::uint32_t> adjacency(n * R, kNoNeighbor), degree(n, 0);
    std::vector<NodeId> nbs;
    for (NodeId i = 0; i < n; ++i) {
        nbs.clear();
        for (auto& r : readers) {
            if (!r.more || r.id != i) continue;
            std::vector<unsigned char> b(4 * std::size_t(r.degree));
            r.in.read(reinterpret_cast<char*>(b.data()), std::streamsize(b.size()));
            for (std::size_t s = 0; s < r.degree; ++s) nbs.push_back(load_le<std::uint32_t>(&b[4 * s]));
            r.advance();
        }
        std::sort(nbs.begin(), nbs.end());
        nbs.erase(std::unique(nbs.begin(), nbs.end()), nbs.end());
        std::erase(nbs, i);
        if (nbs.size() > R) {
            ++st.merged_and_pruned;
            read_raw(i, raw.data());
            pv.prepare(raw, metric);
            pq.prepare(pv.values(), table);
            std::vector<std::pair<float, NodeId>> c;
            for (NodeId q : nbs) c.emplace_back(pq.distance(table, code_of(q)), q);
            // Ordering equal distances by node number would make every identical
            // copy prefer the same lowest-numbered ones (found by dk37).
            std::sort(c.begin(), c.end(), [&](const auto& x, const auto& y) {
                if (x.first != y.first) return x.first < y.first;
                return tie(i, x.second) < tie(i, y.second);
            });
            std::vector<NodeId> kept;
            std::vector<bool> dropped(c.size(), false);
            for (std::size_t a2 = 0; a2 < c.size() && kept.size() < R; ++a2) {
                if (dropped[a2]) continue;
                kept.push_back(c[a2].second);
                pq.decode(code_of(c[a2].second), decoded);
                for (std::size_t z = a2 + 1; z < c.size(); ++z) {
                    if (dropped[z] || c[z].first <= 0) continue;
                    pq.decode(code_of(c[z].second), other);
                    float d = 0;
                    for (std::size_t j = 0; j < dim; ++j) d += (decoded[j] - other[j]) * (decoded[j] - other[j]);
                    if (metric == Metric::Cosine) d *= 0.5f;  // 1 - dot = |a - b|^2 / 2 for unit vectors
                    if (alpha * d <= c[z].first) dropped[z] = true;
                }
            }
            nbs = kept;
        }
        degree[i] = static_cast<std::uint32_t>(nbs.size());
        std::copy(nbs.begin(), nbs.end(), adjacency.begin() + std::ptrdiff_t(std::size_t(i) * R));
    }
    // 4b. Connectivity repair, as for in-memory graphs: graphs built per partition
    // need not join into one reachable whole (found by dk37: 75 of 1,200 vectors
    // unreachable). Each unreachable vector gets an edge from a reachable vector
    // it links to (near it by construction), preferring a free slot, then
    // replacing a neighbor that keeps another incoming edge; repeated until all
    // are reachable (at most 16 rounds).
    auto out = [&](NodeId v) { return adjacency.begin() + std::ptrdiff_t(std::size_t(v) * R); };
    for (int round = 0; round < 16; ++round) {
        std::vector<bool> seen(n, false);
        std::vector<NodeId> queue{medoid};
        seen[medoid] = true;
        for (std::size_t h = 0; h < queue.size(); ++h)
            for (std::uint32_t s = 0; s < degree[queue[h]]; ++s) {
                const NodeId nb = out(queue[h])[s];
                if (!seen[nb]) {
                    seen[nb] = true;
                    queue.push_back(nb);
                }
            }
        if (queue.size() == n) break;
        // Incoming edges from reachable vectors only: an edge from an unreachable
        // one does not keep its target reachable (counting those made the repair
        // undo itself, round after round).
        std::vector<std::uint32_t> indegree(n, 0);
        for (NodeId v = 0; v < n; ++v)
            if (seen[v])
                for (std::uint32_t s = 0; s < degree[v]; ++s) ++indegree[out(v)[s]];
        for (NodeId u = 0; u < n; ++u) {
            if (seen[u]) continue;
            std::vector<NodeId> hosts;  // reachable vectors u links to, then the medoid
            for (std::uint32_t s = 0; s < degree[u]; ++s)
                if (seen[out(u)[s]]) hosts.push_back(out(u)[s]);
            hosts.push_back(medoid);
            NodeId host = hosts.front();
            std::uint32_t slot = std::uint32_t(R);  // R: append
            bool placed = false;
            for (NodeId h : hosts)
                if (degree[h] < R) {
                    host = h;
                    placed = true;
                    break;
                }
            for (std::size_t x = 0; !placed && x < hosts.size(); ++x)
                for (std::uint32_t s = degree[hosts[x]]; s-- > 0;)
                    if (indegree[out(hosts[x])[s]] >= 2) {
                        host = hosts[x];
                        slot = s;
                        placed = true;
                        break;
                    }
            if (!placed) slot = degree[host] - 1;  // GCOVR_EXCL_LINE: every slot's target has one incoming edge only
            if (std::find(out(host), out(host) + degree[host], u) != out(host) + degree[host]) continue;  // GCOVR_EXCL_LINE: u would be reachable
            if (slot == R) {
                out(host)[degree[host]++] = u;
            } else {
                --indegree[out(host)[slot]];
                out(host)[slot] = u;
            }
            ++indegree[u];
            seen[u] = true;  // reachable through host; its own links follow next round
        }
    }
    // 4c. Write, one sequential pass in ID order.
    write_index(path, dim, metric, R, medoid, n, params.index.vectors,
                [&](NodeId i, std::vector<float>& v, std::vector<NodeId>& list) {
                    read_raw(i, raw.data());
                    pv.prepare(raw, metric);
                    v.assign(pv.values().begin(), pv.values().end());
                    list.assign(out(i), out(i) + degree[i]);
                },
                pq, codes, ids, store);
    if (stats) *stats = st;
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
    const auto version = load_le<std::uint32_t>(&h[8]);
    if (version != kDiskVersion && version != kDiskVersionHalf) throw FormatError("unsupported disk index version");
    if (load_le<std::uint32_t>(&h[12]) != kByteOrder) throw FormatError("wrong byte order mark");
    const std::size_t header_size = 8 + 4 + 4 + 8 + 4 + 4 + 8 * 7 + kSections * 20;
    if (persist::kVerifyChecksums && load_le<std::uint32_t>(&h[header_size]) != persist::crc32(0, h.data(), header_size)) throw FormatError("header checksum mismatch");
    const auto dim = load_le<std::uint64_t>(&h[16]);
    const auto metric = load_le<std::uint32_t>(&h[24]);
    const auto n = load_le<std::uint64_t>(&h[32]);
    const auto R = load_le<std::uint64_t>(&h[40]);
    const auto medoid = load_le<std::uint64_t>(&h[48]);
    if (dim == 0 || dim > (1u << 20) || R == 0 || R > (1u << 16) || n == 0 || n >= kNoNeighbor || medoid >= n)
        throw FormatError("invalid disk index sizes");
    if (metric != std::uint32_t(Metric::L2) && metric != std::uint32_t(Metric::Cosine)) throw FormatError("invalid metric");
    // Version 1: float32, the field reserved (0). Version 2: a half-precision format.
    const auto format = load_le<std::uint32_t>(&h[28]);
    if (version == kDiskVersion ? format != 0 : (format != std::uint32_t(DiskVectorFormat::Float16) && format != std::uint32_t(DiskVectorFormat::BFloat16)))
        throw FormatError("invalid vector format for this disk index version");
    d->format_ = static_cast<DiskVectorFormat>(format);
    const Layout layout(dim, R, n, d->format_);
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
        if (persist::kVerifyChecksums && persist::crc32(0, sec[s].data(), sec[s].size()) != load_le<std::uint32_t>(e + 16)) throw FormatError("section checksum mismatch");
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
        if (persist::kVerifyChecksums && persist::crc32(0, buf.data() + b * kBlock, kBlock) != block_crc_[first + b]) throw FormatError("node block checksum mismatch");
    const unsigned char* r = buf.data() + (node_offset(node) - kBlock * (1 + first));
    Node out;
    out.vector.resize(dim_);
    const std::size_t value_bytes = format_ == DiskVectorFormat::Float32 ? 4 : 2;
    if (format_ == DiskVectorFormat::Float32)
        for (std::size_t j = 0; j < dim_; ++j) out.vector[j] = load_le<float>(r + 4 * j);
    else if (format_ == DiskVectorFormat::Float16)
        for (std::size_t j = 0; j < dim_; ++j) out.vector[j] = f16_to_float(load_le<std::uint16_t>(r + 2 * j));
    else
        for (std::size_t j = 0; j < dim_; ++j) out.vector[j] = bf16_to_float(load_le<std::uint16_t>(r + 2 * j));
    for (const float v : out.vector)  // behind a valid checksum, still a NaN or infinity would poison distances
        if (!std::isfinite(v)) throw FormatError("a stored vector is not finite");
    const auto degree = load_le<std::uint32_t>(r + value_bytes * dim_);
    if (degree > R_) throw FormatError("invalid node degree");
    for (std::size_t s = 0; s < degree; ++s) {
        const auto nb = load_le<std::uint32_t>(r + value_bytes * dim_ + 4 + 4 * s);
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