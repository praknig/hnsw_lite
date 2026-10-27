// The hnsw-lite command-line tool, run as a program: every command on every kind of
// file and durable directory, its exit statuses, both export formats, and that it never
// changes what it reads. HNSW_LITE_CLI is the path of the program built with this test.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#include "collection.h"
#include "disk_index.h"
#include "flat_index.h"
#include "hnsw_index.h"
#include "sparse.h"

using namespace vecdb;
namespace fs = std::filesystem;

namespace {
int failures = 0, checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("  FAILED: %s\n", what.c_str());
    }
}

const fs::path work = fs::temp_directory_path() / "hnsw-lite-test-cli";

/// Runs the tool with `args`; its exit status, and its output (stdout and stderr) in `out`.
int run(const std::string& args, std::string& out) {
    // A fresh file per command: if a command fails to start (on Windows, a just-written
    // file can stay locked for a moment), its output reads as empty, never as another's.
    static int serial = 0;
    const fs::path path = work / ("out-" + std::to_string(serial++) + ".txt");
    std::error_code ignored;
    fs::remove(path, ignored);
    const std::string file = path.string();
    std::string cmd = "\"" HNSW_LITE_CLI "\" " + args + " > \"" + file + "\" 2>&1";
#if defined(_WIN32)
    cmd = "\"" + cmd + "\"";  // cmd.exe strips one pair of quotes around the whole line
#endif
    int status = std::system(cmd.c_str());
#if !defined(_WIN32)
    status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
    std::ifstream in(file, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    out = text.str();
    return status;
}
std::string at(const std::string& name) { return "\"" + (work / name).string() + "\""; }
bool has(const std::string& out, const std::string& text) { return out.find(text) != std::string::npos; }
std::size_t lines(const std::string& s) {
    std::size_t n = 0;
    for (const char c : s) n += c == '\n';
    return n;
}
/// Every byte of every file in a directory, in name order: to see that nothing changed.
std::string contents(const fs::path& dir) {
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file()) files.push_back(e.path());
    std::sort(files.begin(), files.end());
    std::string all;
    for (const auto& f : files) {
        std::ifstream in(f, std::ios::binary);
        all += f.filename().string() + "\n" + std::string((std::istreambuf_iterator<char>(in)), {});
    }
    return all;
}
std::size_t leftover_copies() {
    std::size_t n = 0;
    for (const auto& e : fs::directory_iterator(fs::temp_directory_path()))
        n += e.path().filename().string().rfind("hnsw-lite-cli-", 0) == 0;
    return n;
}

void make_fixtures() {
    fs::remove_all(work);
    fs::create_directories(work);
    const std::string d = work.string() + "/";
    std::mt19937 rng(1); std::normal_distribution<float> nd(0, 1);
    auto vec = [&](std::size_t n) { std::vector<float> v(n); for (float& x : v) x = nd(rng); return v; };
    auto md = [](std::uint64_t i) { return Metadata().set("lang", i % 3 ? "en" : "fr").set("year", std::int64_t(2020 + i % 4)).set("note", "a, \"quoted\"\nline"); };
    HnswParams p; p.quantization.codec = CodecKind::SQ8; p.quantization.train_after = 50;
    HnswIndex h(8, Metric::Cosine, p);
    for (std::uint64_t i = 0; i < 120; ++i) h.add(i, vec(8), md(i));
    h.create_payload_index("lang"); (void)h.remove(7);
    h.save(d + "hnsw.idx");
    FlatIndex f(4, Metric::L2);
    for (std::uint64_t i = 0; i < 30; ++i) f.add(i, vec(4), md(i));
    f.save(d + "flat.idx");
    SparseIndex s;
    for (std::uint64_t i = 0; i < 20; ++i) s.add(i, SparseVector{{std::uint32_t(i % 5), 9}, {1.0f, 0.5f}}, md(i));
    s.save(d + "sparse.idx");
    CollectionOptions o = CollectionOptions::preset(4); o.sparse = true; o.string_ids = true; o.ttl = TtlOptions{true, 0}; o.maintenance.mode = Maintenance::Manual;
    Collection c(o);
    for (std::uint64_t i = 0; i < 10; ++i) { Document doc; doc.vector = vec(4); doc.text = "doc " + std::to_string(i); doc.sparse = SparseVector{{1}, {2.0f}}; doc.metadata = md(i); c.add("key" + std::to_string(i), doc); }
    c.save(d + "collection.col");
    // Durable directories, closed with log records after the last snapshot.
    { auto x = HnswIndex::open_durable(d + "hnsw.dir", 4, Metric::L2); for (std::uint64_t i = 0; i < 25; ++i) x->add(i, vec(4), md(i)); (void)x->checkpoint(); for (std::uint64_t i = 25; i < 40; ++i) x->add(i, vec(4), md(i)); }
    { auto x = FlatIndex::open_durable(d + "flat.dir", 4, Metric::L2); for (std::uint64_t i = 0; i < 12; ++i) x->add(i, vec(4), md(i)); }
    { auto x = SparseIndex::open_durable(d + "sparse.dir"); for (std::uint64_t i = 0; i < 9; ++i) x->add(i, SparseVector{{std::uint32_t(i)}, {1.0f}}, md(i)); (void)x->remove(3); }
    { auto x = Collection::open(d + "collection.dir", o); for (std::uint64_t i = 0; i < 6; ++i) { Document doc; doc.vector = vec(4); doc.text = "t"; doc.metadata = md(i); x->add("k" + std::to_string(i), doc); } }
    { std::vector<std::uint64_t> ids; std::vector<float> all; for (std::uint64_t i = 0; i < 300; ++i) { ids.push_back(i); auto v = vec(8); all.insert(all.end(), v.begin(), v.end()); }
      DiskIndex::build(d + "disk.idx", ids, all, 8, Metric::L2); }
    { std::ifstream in(d + "flat.idx", std::ios::binary); std::string b((std::istreambuf_iterator<char>(in)), {}); b[b.size() / 2] ^= 0x5A;
      std::ofstream(d + "damaged.idx", std::ios::binary) << b; }
    std::ofstream(d + "notes.txt") << "not an index";
    {  // edges: tags and control characters, an empty index, a space in a path, a damaged log, no snapshot
        FlatIndex t(2, Metric::L2);
        t.add(1, std::vector<float>{0.5f, -2.0f}, Metadata().set("tags", std::vector<std::string>{"a", "b"}).set("text", std::string("tab\there\x01")));
        t.save(d + "tags.idx");
        HnswIndex(4, Metric::L2).save(d + "empty.idx");
        fs::create_directories(d + "with space");
        fs::copy_file(d + "flat.idx", d + "with space/flat.idx");
        fs::create_directories(d + "broken.dir");
        for (const auto& e : fs::directory_iterator(d + "sparse.dir")) fs::copy_file(e.path(), d + "broken.dir/" + e.path().filename().string());
        for (const auto& e : fs::directory_iterator(d + "broken.dir"))
            if (e.path().filename().string().rfind("wal-", 0) == 0) {
                std::fstream log(e.path(), std::ios::in | std::ios::out | std::ios::binary);
                log.seekp(60);
                log.put('\x5A');  // inside the first record: damage, not a torn tail
            }
        fs::create_directories(d + "nosnap.dir");
        fs::copy_file(d + "sparse.dir/DURABLE", d + "nosnap.dir/DURABLE");
    }
}
}  // namespace

int main() {
    make_fixtures();
    const std::size_t copies_before = leftover_copies();
    std::vector<std::string> dirs{"hnsw.dir", "flat.dir", "sparse.dir", "collection.dir"};
    std::vector<std::string> before;
    for (const auto& d : dirs) before.push_back(contents(work / d));
    std::string out;

    // inspect: every kind, files and durable directories.
    for (const auto& [name, kind] : std::vector<std::pair<std::string, std::string>>{
             {"hnsw.idx", "hnsw (file)"}, {"flat.idx", "flat (file)"}, {"sparse.idx", "sparse (file)"}, {"collection.col", "collection (file)"},
             {"disk.idx", "disk (file)"}, {"hnsw.dir", "hnsw (durable directory)"}, {"flat.dir", "flat (durable directory)"},
             {"sparse.dir", "sparse (durable directory)"}, {"collection.dir", "collection (durable directory)"}}) {
        check(run("inspect " + at(name), out) == 0 && has(out, kind), "inspect " + name + ": " + out.substr(0, 200));
    }
    run("inspect " + at("hnsw.idx"), out);
    check(has(out, "119 (+1 removed") && has(out, "lang (keyword, indexed)") && has(out, "SQ8"), "inspect details: " + out);
    run("inspect " + at("hnsw.dir"), out);
    check(has(out, "15 log records replayed") && has(out, "vectors        40"), "inspect recovery: " + out);
    run("inspect " + at("collection.col"), out);
    check(has(out, "string keys") && !has(out, "_key"), "inspect hides reserved fields: " + out);

    // verify: good, damaged, foreign, missing.
    check(run("verify " + at("flat.idx"), out) == 0 && has(out, "OK: flat file, 30 vectors"), "verify ok: " + out);
    check(run("verify " + at("collection.dir"), out) == 0 && has(out, "recovered with"), "verify a directory: " + out);
    check(run("verify " + at("damaged.idx"), out) == 1 && has(out, "FAILED"), "verify damaged: " + out);
    check(run("verify " + at("notes.txt"), out) == 1 && has(out, "not an hnsw-lite file"), "verify foreign: " + out);
    check(run("verify " + at("missing.idx"), out) == 1 && has(out, "no such file"), "verify missing: " + out);

    // health.
    check(run("health " + at("hnsw.idx") + " --recall-sample 20", out) == 0 && has(out, "unreachable    0") && has(out, "recall"), "health: " + out);
    check(run("health " + at("flat.idx") + " --recall-sample 0", out) == 0 && has(out, "live           30"), "health flat: " + out);
    check(run("health " + at("sparse.idx"), out) == 1, "health refuses a sparse index: " + out);

    // export: JSON Lines and CSV, limits, without vectors, to a file.
    check(run("export " + at("hnsw.idx"), out) == 0 && lines(out) == 119 && out.rfind("{\"id\":", 0) == 0 && has(out, "\"vector\":["), "export jsonl");
    check(run("export " + at("collection.dir") + " --limit 3", out) == 0 && lines(out) == 3 && has(out, "\"key\":\"k"), "export a recovered collection: " + out);
    check(run("export " + at("sparse.idx") + " --no-vectors", out) == 0 && lines(out) == 20 && !has(out, "sparse"), "export without vectors");
    check(run("export " + at("flat.idx") + " --format csv --no-vectors --limit 2", out) == 0 && out.rfind("id,lang,note,year\n", 0) == 0 &&
              has(out, "\"a, \"\"quoted\"\"\nline\"") && lines(out) == 3 + 2,  // a header, two rows, each with a quoted newline
          "export csv: " + out);
    check(run("export " + at("flat.idx") + " --output " + at("flat.jsonl"), out) == 0 && fs::exists(work / "flat.jsonl") && has(out, "exported 30 rows"),
          "export to a file: " + out);
    check(run("export " + at("disk.idx"), out) == 1, "export refuses a disk index");

    // Edges.
    check(run("export " + at("hnsw.idx") + " --limit 0", out) == 0 && lines(out) == 119, "--limit 0 means all");
    check(run("export " + at("tags.idx"), out) == 0 && has(out, "\"tags\":[\"a\",\"b\"]") && has(out, "tab\\there\\u0001"), "export tags and control characters: " + out);
    check(run("export " + at("tags.idx") + " --format csv", out) == 0 && has(out, "\"[\"\"a\"\",\"\"b\"\"]\""), "csv tags: " + out);
    check(run("export " + at("empty.idx"), out) == 0 && out.empty(), "export of an empty index: " + out);
    check(run("inspect " + at("empty.idx"), out) == 0 && has(out, "vectors        0"), "inspect an empty index: " + out);
    check(run("inspect " + at("with space/flat.idx"), out) == 0 && has(out, "flat (file)"), "a path with a space: " + out);
    check(run("verify " + at("broken.dir"), out) == 1 && has(out, "FAILED"), "a damaged log: " + out);
    check(run("inspect " + at("nosnap.dir"), out) == 1 && has(out, "without a snapshot"), "a directory without a snapshot: " + out);
    check(run("health " + at("collection.col") + " --recall-sample 0", out) == 0 && has(out, "unreachable"), "health of a collection: " + out);
    check(run("health " + at("disk.idx") + " --recall-sample 0", out) == 0 && has(out, "live           300"), "health of a disk index: " + out);
    check(run("export " + at("flat.idx") + " --output", out) == 2, "an option without its value");
    check(run("health " + at("flat.idx") + " --recall-sample many", out) == 2, "a count that is not a number");

    // Usage errors.
    for (const std::string& args : std::vector<std::string>{"", "frobnicate " + at("flat.idx"), "export " + at("flat.idx") + " --format xml",
                                                            "export " + at("flat.idx") + " --limit -3", "inspect"})
        check(run(args, out) == 2, "usage error for '" + args + "': " + out);
    check(run("--help", out) == 0 && has(out, "usage:"), "--help");

    // Nothing it read changed, and no temporary copy is left.
    for (std::size_t i = 0; i < dirs.size(); ++i) check(contents(work / dirs[i]) == before[i], dirs[i] + " unchanged");
    check(leftover_copies() == copies_before, "no temporary copies left");

    fs::remove_all(work);
    std::printf("hnsw-lite command-line tool: %d checks, %d failed\n", checks, failures);
    return failures ? 1 : 0;
}