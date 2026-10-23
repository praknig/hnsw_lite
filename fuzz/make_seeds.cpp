/// Writes one valid file of each kind into fuzz/seeds/<target>/, for the fuzzers to start from.
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

#include "collection.h"
#include "disk_index.h"
#include "flat_index.h"
#include "hnsw_index.h"
#include "sparse.h"
#include "wal.h"
#include "wal_ops.h"

using namespace vecdb;
namespace fs = std::filesystem;

template <class X>
void write(const std::string& dir, const std::string& name, const X& x) {
    fs::create_directories(dir);
    std::ofstream out(dir + "/" + name, std::ios::binary);
    x.save(out);
}

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : "fuzz/seeds";
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1, 1);
    auto vec = [&](std::size_t dim) { std::vector<float> v(dim); for (float& x : v) x = u(rng); return v; };
    HnswIndex h(4, Metric::L2);
    for (std::uint64_t i = 0; i < 30; ++i) h.add(i, vec(4), Metadata().set("a", std::int64_t(i % 3)).set("k", "w").set_tags("t", {"x"}));
    (void)h.remove(3);
    write(root + "/1", "hnsw.idx", h);
    HnswParams q;
    q.quantization.codec = CodecKind::SQ8;
    q.quantization.train_after = 10;
    HnswIndex hq(4, Metric::Cosine, q);
    for (std::uint64_t i = 0; i < 20; ++i) hq.add(i, vec(4));
    write(root + "/1", "hnsw_sq8.idx", hq);
    IvfParams ivf;
    ivf.lists = 3;
    ivf.train_after = 10;
    FlatIndex f(4, Metric::InnerProduct, Schema::dynamic(), {}, ivf);
    for (std::uint64_t i = 0; i < 20; ++i) f.add(i, vec(4), Metadata().set("a", 1.5));
    write(root + "/2", "flat_ivf.idx", f);
    SparseIndex s;
    for (std::uint64_t i = 0; i < 20; ++i) s.add(i, {{std::uint32_t(i % 5), std::uint32_t(7 + i % 3)}, {1.0f, 0.5f}}, Metadata().set("b", true));
    (void)s.remove(4);
    write(root + "/3", "sparse.idx", s);
    CollectionOptions co;
    co.dim = 4;
    co.metric = Metric::L2;
    co.sparse = true;
    Collection c(co);
    for (std::uint64_t i = 0; i < 15; ++i) {
        Document d;
        d.vector = vec(4);
        d.text = "word " + std::to_string(i);
        d.sparse = {{std::uint32_t(i % 4)}, {1.0f}};
        d.metadata = Metadata().set("n", std::int64_t(i));
        c.add(i, d);
    }
    (void)c.remove(2);
    write(root + "/4", "collection.col", c);
    fs::create_directories(root + "/5");
    const std::string seg = root + "/5/segment.wal";
    fs::remove(seg);
    {
        HnswIndex logged(4, Metric::L2);
        logged.attach_log(WalWriter::create(seg, 7, 1));
        for (std::uint64_t i = 0; i < 10; ++i) logged.add(i, vec(4), Metadata().set("a", std::int64_t(i)));
        (void)logged.remove(2);
        logged.set_metadata(3, Metadata().set("a", std::int64_t(9)));
        (void)logged.detach_log();
    }
    fs::create_directories(root + "/6");
    std::vector<std::uint64_t> ids;
    std::vector<float> flat;
    for (std::uint64_t i = 0; i < 40; ++i) {
        ids.push_back(i);
        const auto v = vec(4);
        flat.insert(flat.end(), v.begin(), v.end());
    }
    DiskIndexParams dp;
    dp.graph.R = 8;
    dp.graph.L = 16;
    DiskIndex::build(root + "/6/disk.idx", ids, flat, 4, Metric::L2, dp);
    return 0;
}