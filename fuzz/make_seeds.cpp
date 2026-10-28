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
    HnswParams half;  // half precision, codes only: the loader's checks on float16 codes
    half.quantization.codec = CodecKind::F16;
    half.quantization.keep_full_vectors = false;
    half.quantization.rerank = false;
    HnswIndex hh(4, Metric::L2, half);
    for (std::uint64_t i = 0; i < 20; ++i) hh.add(i, vec(4));
    write(root + "/1", "hnsw_f16.idx", hh);
    HnswParams rq;  // RaBitQ, codes only, 6 dimensions (not a power of two): its state and codes
    rq.quantization.codec = CodecKind::RaBitQ;
    rq.quantization.keep_full_vectors = false;
    rq.quantization.rerank = false;
    rq.quantization.train_after = 10;
    HnswIndex hr(6, Metric::L2, rq);
    for (std::uint64_t i = 0; i < 20; ++i) {
        auto v = vec(4);
        v.push_back(v[0] - v[1]);
        v.push_back(v[2] + v[3]);
        hr.add(i, v);
    }
    write(root + "/1", "hnsw_rabitq.idx", hr);
    IvfParams ivf;
    ivf.lists = 3;
    ivf.train_after = 10;
    FlatIndex f(4, Metric::InnerProduct, Schema::dynamic(), {}, ivf);
    for (std::uint64_t i = 0; i < 20; ++i) f.add(i, vec(4), Metadata().set("a", 1.5));
    write(root + "/2", "flat_ivf.idx", f);
    QuantizationParams bq;  // bfloat16 codes only, with IVF lists
    bq.codec = CodecKind::BF16;
    bq.keep_full_vectors = false;
    bq.rerank = false;
    FlatIndex fb(4, Metric::L2, Schema::dynamic(), bq, ivf);
    for (std::uint64_t i = 0; i < 20; ++i) fb.add(i, vec(4));
    write(root + "/2", "flat_bf16.idx", fb);
    QuantizationParams fr;  // RaBitQ with full vectors, cosine
    fr.codec = CodecKind::RaBitQ;
    fr.train_after = 10;
    FlatIndex frq(4, Metric::Cosine, Schema::dynamic(), fr);
    for (std::uint64_t i = 0; i < 20; ++i) frq.add(i, vec(4));
    write(root + "/2", "flat_rabitq.idx", frq);
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
    CollectionOptions ko = co;  // with saved settings (expiry, string keys): the settings block
    ko.ttl = TtlOptions{true, 3600000};
    ko.string_ids = true;
    Collection kc(ko);
    for (std::uint64_t i = 0; i < 8; ++i) {
        Document d;
        d.vector = vec(4);
        kc.add("key" + std::to_string(i), d);
    }
    write(root + "/4", "collection_settings.col", kc);
    CollectionOptions to = ko;  // both settings blocks: runtime settings, then text analysis
    to.text = true;
    to.bm25.unicode = to.bm25.stop_words = to.bm25.stem = true;
    Collection tc(to);
    for (std::uint64_t i = 0; i < 8; ++i) {
        Document d;
        d.vector = vec(4);
        d.text = "running \u00C9cole don\u2019t " + std::to_string(i);
        tc.add("text" + std::to_string(i), d);
    }
    write(root + "/4", "collection_analysis.col", tc);
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
    {  // a sparse index's log: every record it writes
        const std::string dir = root + "/5/sparse-dir";
        fs::remove_all(dir);
        {
            auto sp = SparseIndex::open_durable(dir);
            for (std::uint64_t i = 0; i < 6; ++i) sp->add(i, SparseVector{{std::uint32_t(i), 7}, {1.0f, 0.5f}}, Metadata().set("a", std::int64_t(i)));
            (void)sp->update(1, SparseVector{{2}, {2.0f}});
            (void)sp->upsert(9, SparseVector{{3}, {1.0f}}, Metadata().set("a", std::int64_t(9)));
            (void)sp->set_metadata(2, Metadata().set("a", std::int64_t(5)));
            (void)sp->remove(3);
            (void)sp->remove_where(Filter::eq("a", std::int64_t(4)));
            sp->compact();
        }
        for (const auto& e : fs::directory_iterator(dir))
            if (e.path().filename().string().rfind("wal-", 0) == 0) fs::copy_file(e.path(), root + "/5/sparse.wal", fs::copy_options::overwrite_existing);
        fs::remove_all(dir);
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
    dp.vectors = DiskVectorFormat::Float16;  // format version 2: half-precision nodes
    DiskIndex::build(root + "/6/disk_f16.idx", ids, flat, 4, Metric::L2, dp);
    dp.vectors = DiskVectorFormat::BFloat16;
    DiskIndex::build(root + "/6/disk_bf16.idx", ids, flat, 4, Metric::Cosine, dp);
    return 0;
}