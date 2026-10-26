/**
 * @file fuzz_load.cpp
 * @brief libFuzzer harnesses for every loader. Build one per FUZZ_TARGET:
 *   1 HnswIndex, 2 FlatIndex, 3 SparseIndex, 4 Collection, 5 WAL segment, 6 DiskIndex.
 *
 * Each input is loaded; format errors are expected. A load that succeeds must give
 * an object that works: it is searched, saved, loaded again, and saved again, and
 * the two saves must be identical (a loader accepting inconsistent content fails
 * here, not only one that crashes).
 */
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "collection.h"
#include "disk_index.h"
#include "flat_index.h"
#include "hnsw_index.h"
#include "serialize.h"
#include "sparse.h"
#include "wal.h"

using namespace vecdb;

namespace {

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "FUZZ FINDING: %s\n", what);
    std::abort();
}

template <class Index>
std::string bytes_of(const Index& x) {
    std::ostringstream out;
    x.save(out);
    return std::move(out).str();
}

template <class Load>
void round_trip(const std::string& first, Load&& load) {
    std::istringstream in(first);
    auto again = load(in);
    if (bytes_of(*again) != first) fail("saving a loaded object, loading it, and saving again gave different bytes");
}

std::string temp_file(const char* ext) {
    return (std::filesystem::temp_directory_path() / ("fuzz_" + std::to_string(::getpid()) + ext)).string();
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, unsigned long size) {
    const std::string input(reinterpret_cast<const char*>(data), size);
    try {
#if FUZZ_TARGET == 1
        std::istringstream in(input);
        auto h = HnswIndex::load(in);
        if (h->size() > 0) {
            std::vector<float> q(h->dim(), 0.5f);
            (void)h->search(q, 5, 32);
            SearchOptions o;
            o.filter = Filter::exists("a");
            (void)h->search(q, 5, 32, o);
            HealthOptions ho;
            ho.recall_sample = 3;
            ho.codec_sample = 20;
            (void)h->health(ho);
        }
        round_trip(bytes_of(*h), [](std::istream& s) { return HnswIndex::load(s); });
#elif FUZZ_TARGET == 2
        std::istringstream in(input);
        auto f = FlatIndex::load(in);
        if (f->size() > 0) {
            std::vector<float> q(f->dim(), 0.5f);
            (void)f->search(q, 5);
            HealthOptions ho;
            ho.recall_sample = 3;
            ho.codec_sample = 20;
            (void)f->health(ho);
        }
        round_trip(bytes_of(*f), [](std::istream& s) { return FlatIndex::load(s); });
#elif FUZZ_TARGET == 3
        std::istringstream in(input);
        auto s = SparseIndex::load(in);
        (void)s->search({{1, 2, 3}, {1.0f, 1.0f, 1.0f}}, 5);
        (void)s->records();
        round_trip(bytes_of(*s), [](std::istream& x) { return SparseIndex::load(x); });
#elif FUZZ_TARGET == 4
        std::istringstream in(input);
        auto c = Collection::load(in);
        CollectionQuery q;
        if (c->options().dim > 0) q.vector.assign(c->options().dim, 0.5f);
        if (c->options().text) q.text = "a b c";
        if (c->options().sparse) q.sparse = {{1}, {1.0f}};
        (void)c->search(q, 5);
        round_trip(bytes_of(*c), [](std::istream& x) { return Collection::load(x); });
#elif FUZZ_TARGET == 5
        const std::string path = temp_file(".wal");
        { std::ofstream(path, std::ios::binary) << input; }
        try {
            const WalScan scan = wal::read_segment(path);
            if (scan.valid_end > scan.file_size) fail("valid_end beyond the file");
            // Every valid record must decode or be rejected cleanly, and replay must not crash.
            HnswIndex h(4, Metric::L2);
            try {
                (void)h.replay(scan.records);
            } catch (const FormatError&) {
            } catch (const std::invalid_argument&) {
            }
            SparseIndex sparse;  // the sparse index's own records (and every other kind, refused cleanly)
            try {
                (void)sparse.replay(scan.records);
            } catch (const FormatError&) {
            } catch (const std::invalid_argument&) {
            }
        } catch (...) {
            std::filesystem::remove(path);
            throw;
        }
        std::filesystem::remove(path);
#elif FUZZ_TARGET == 6
        const std::string path = temp_file(".disk");
        { std::ofstream(path, std::ios::binary) << input; }
        try {
            auto d = DiskIndex::open(path);
            if (d->size() > 0) {
                std::vector<float> q(d->dim(), 0.5f);
                (void)d->search(q, 5);
                (void)d->scan();
            }
        } catch (...) {
            std::filesystem::remove(path);
            throw;
        }
        std::filesystem::remove(path);
#else
#error "define FUZZ_TARGET"
#endif
    } catch (const FormatError&) {
    } catch (const std::invalid_argument&) {
    } catch (const std::runtime_error&) {
    }
    return 0;
}
