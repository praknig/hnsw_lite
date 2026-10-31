# Benchmarks

`bench/python/compare.py` compares hnsw-lite with other vector search libraries on your machine: hnswlib, FAISS (HNSW, and IVF-PQ as the memory-saving reference) and USearch.

## Running it

```bash
pip install -r bench/python/requirements.txt
python bench/python/compare.py --dataset glove-100-angular       # about 15 to 30 minutes
python bench/python/compare.py --dataset sift-128-euclidean
python bench/python/compare.py --synthetic 100000,96 --metric cosine   # no download
```

| Option | Default | What it does |
|---|---|---|
| `--dataset` | `glove-100-angular` | A standard [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) dataset, with exact neighbors: `glove-100-angular`, `sift-128-euclidean`, `fashion-mnist-784-euclidean`, `nytimes-256-angular`, `gist-960-euclidean`, `deep-image-96-angular`, `glove-25-angular`, `mnist-784-euclidean`. Downloaded once into `--cache`. |
| `--synthetic N,DIM` | | Clustered random data instead, with `--metric cosine` or `l2`; exact neighbors are computed. |
| `--libraries` | all five | Any of `hnsw-lite`, `hnswlib`, `faiss-hnsw`, `usearch`, `faiss-ivfpq`. A library that is not installed is skipped with a note. |
| `--queries`, `--filtered-queries` | 1000, 200 | Queries measured, unfiltered and filtered. |
| `--M`, `--ef-construction` | 16, 200 | Graph settings, the same for every graph library. |
| `--threads` | every core | For building and batch searches. |
| `--repeat` | 3 | Timed runs per point; the median is reported. |
| `--no-filters` | | Skip the filtered-search comparison. |
| `--out` | `bench-results` | Where the results go. |
| `--cache` | `~/.cache/hnsw-lite-bench` | Where downloaded datasets are kept. |
| `--dataset-url` | the `HNSW_BENCH_DATASET_URL` environment variable | A mirror to download the datasets from, with `{name}` where the dataset's name goes, for when ann-benchmarks' host refuses or is down. |

## What it produces

In `--out`, per dataset:

- **`<dataset>.md`:** a summary table: queries per second at a recall@10 of at least 0.90, 0.95 and 0.99 (one query per call, one thread), batch queries per second at 0.95, build time, memory and index file size; then filtered search at 1%, 10% and 30% of vectors matching; with the machine, settings and library versions.
- **`<dataset>-recall-qps.png`:** recall against queries per second, one curve per library, for single queries and batches.
- **`<dataset>-filtered.png`:** the same for each filter selectivity.
- **`<dataset>.json`:** every measured point, for your own analysis.

## How it keeps the comparison fair

- **Equal recall, not equal settings.** Each library's search beam (`ef`, or `nprobe` for IVF-PQ) is swept, and libraries are compared at the same recall: the same `ef` means different amounts of work in different libraries.
- **The same graph settings** (`M`, `ef_construction`) for every graph library.
- **One process per library,** so memory is measured cleanly and libraries cannot affect each other.
- **Like-for-like indexes.** Every library's speed, memory and size are measured on a plain index. For the filtered searches, hnsw-lite builds a second index carrying the filter field as metadata (outside the measurements), since the other libraries filter through an array outside their index.
- **The same Python calling pattern** for every library: one call per query, or one call per batch.
- **Cosine done right for each library:** FAISS has no cosine metric, so its vectors are normalized and searched by inner product, which ranks identically.
- **Medians of repeated runs,** after a warm-up.

## Reading the results honestly

- **Run on a quiet machine.** Timings on shared or virtual CPUs vary a lot between runs (on a shared single-core sandbox, the same library measured 47,000 and 69,000 queries per second in two runs). Close other programs; for published numbers, use a dedicated machine and state it.
- **Python overhead counts.** Per-query numbers include each library's Python call overhead, which matters most for small, fast queries; batch numbers show the libraries' own speed better.
- **Expect others to win somewhere.** FAISS IVF-PQ uses far less memory than any full-precision graph (at much lower recall without re-ranking); hnswlib is very fast on plain unfiltered search. hnsw-lite's differences show most in filtered search, updates and deletions, hybrid search and durability, which this script measures only in part.
- **Filtered search depends on scale.** hnsw-lite's query planner scans exactly when few vectors match (at most 2,000 by default) and uses ACORN above that, between 1% and 20% matching: on small datasets most filtered queries are exact scans. Run filtered comparisons on datasets of a million vectors or more to see ACORN.

## Native comparison and profiling

`bench/native_compare.cpp` compares hnsw-lite and hnswlib in C++ on identical data, without Python, and measures changes to the search: its first lines say how to build it (hnswlib is header-only; its headers are fetched once). Run without arguments, it reports both libraries' time per query and how far their results agree. To measure a change: build the program before it and keep that binary (the library is linked in statically), run `native_compare save` once, then alternate `native_compare time` between the old and new binaries; `native_compare profile` searches repeatedly for `perf record`, and prints a checksum of the results, which must not change when only speed should.

## Stress and load testing at scale

`bench/python/stress.py` tests hnsw-lite at a million vectors (by default) with checks, not only timings, and exits with 1 if any check fails:

```bash
pip install hnsw-lite-lib numpy psutil
python bench/python/stress.py                                   # 1,000,000 x 128, every phase
python bench/python/stress.py --vectors 100000 --duration 20    # a quick run
python bench/python/stress.py --phases build,recall,load        # some phases only
```

| Phase | What it does | What it checks |
|---|---|---|
| `build` | Inserts the vectors in batches on every core | Every vector stored; insert speed holding up as the index grows |
| `recall` | Exact ground truth by brute force, then searches at `ef` 32 to 256 | Recall@10 at least 0.9 at `ef` 128, and growing with `ef` |
| `load` | 1, 2, 4… up to `--clients` concurrent search clients, `--duration` seconds each | No errors; queries per second and latency percentiles; throughput growing with clients (on several cores) |
| `mixed` | Searchers and writers at once (inserts, updates, removes) | The exact final count; sampled ids present or absent as expected; sampled vectors identical to their last write; recall against a fresh brute force; upkeep leaving no vector unreachable |
| `durable` | A child process writes to a durable collection and is killed mid-write (`SIGKILL`; on Windows, `TerminateProcess`) | Every write it acknowledged survives recovery, with the right vector (one more may appear: written, but killed before acknowledging) |

Other options: `--dim` (128), `--clusters` (1,000: real embeddings are clustered), `--M`, `--ef-construction`, `--ef` (128, for the load and mixed phases), `--batch`, `--threads`, `--queries` (200), `--writers` (2), `--crash-after` (10 s), `--seed`, `--out`. Memory: about (4 × dim + 200) bytes per vector for the index plus the test's own copy of the data (4 × dim bytes per vector): about 1.3 GB at a million 128-dimensional vectors.

Results go to `stress-results/stress.md` (a summary with every check) and `stress.json`. Measured on one shared virtual core (so the load phase cannot scale with clients):

| Phase | 1,000,000 × 128 |
|---|---|
| Build | 270 s, about 3,700 inserts/s on one core |
| Recall@10 | 0.862 at `ef` 32, 0.970 at 64, 0.998 at 128, 1.000 at 256 |
| Load (`ef` 128) | 4,462 queries/s on one core; p50 0.19 ms, p99 0.49 ms |
| Mixed (30 s) | 59,185 searches during 6,954 changes; exact count and vectors; recall 0.998 |
| Crash | 4,239 acknowledged writes, all recovered in 0.7 s |

**What it found:** about 1 vector in 1,000 (1,119 of a million) could not be reached by a search before upkeep, which repaired them all in one second. A large bulk load is worth following with `upkeep()` (a collection with background maintenance does it for you).

## Nightly benchmark

`.github/workflows/benchmark.yml` runs `stress.py` every night at 02:00 UTC (and on demand: *Actions > Benchmark > Run workflow*, with the size and duration as inputs) and publishes the results to the **`benchmarks` branch**, created on the first run with no code history. It never commits to `master`: the README there shows the charts straight from the `benchmarks` branch, so they update without any commit to `master`, `master`'s protection and history stay untouched, and the bot's pushes trigger no other workflow.

On the branch: `README.md` (the latest run, the charts, a table of the last 30 runs), `latest-recall.png`, `latest-load.png`, `latest-build.png`, `history.png` (insert speed, queries per second, recall and p99 latency across nights; rates and latencies drawn from zero, so noise looks like noise), `latest.json`, and every run's full results in `runs/`. If any check fails, the results are still published and the job fails, so a regression is both recorded and visible.

**Sizing.** `stress.py --vectors auto` picks the largest run that fits in 70% of the free memory (about 1,400 bytes per vector at 128 dimensions: the index plus the test's copy of the data), at most `--max-vectors` (50 million); `--max-hours 4` stops building at two hours, so a run fits GitHub's six-hour limit. On GitHub's standard runner (16 GB) that is about 8 million vectors; 50 million uses about 70 GB, so `auto` chooses it with about 100 GB free.

**A bigger machine.** Set the repository variable `BENCHMARK_RUNNER` (*Settings > Secrets and variables > Actions > Variables*) to a [larger runner](https://docs.github.com/en/actions/using-github-hosted-runners/using-larger-runners)'s label, or to `self-hosted` for your own machine; the workflow uses it in `runs-on` without any change. Compare nights on the same machine: the charts mark run sizes when they differ.

**Locally:** `python bench/python/publish_benchmark.py --results stress-results --site <a directory>` draws the same charts and page from a `stress.py` run.

## Real-dataset benchmark

`.github/workflows/real-benchmarks.yml` runs `compare.py` on **SIFT1M** (`sift-128-euclidean`: 1,000,000 image descriptors, 128 dimensions, L2) and **GloVe** (`glove-100-angular`: 1,183,514 word vectors, 100 dimensions, cosine), with all 10,000 queries of each file, against hnswlib, FAISS (HNSW and IVF-PQ) and USearch. It runs on demand (*Actions > Real-dataset benchmark > Run workflow*, with the datasets and the number of queries as inputs) and on the 1st of each month, so the numbers follow the code. The datasets are downloaded once from ann-benchmarks and cached between runs. If that host refuses or is down, set the repository variable `BENCHMARK_DATASET_URL` to a mirror, with `{name}` where the dataset's name goes (for example `https://example.org/ann/{name}.hdf5`); locally, `compare.py --dataset-url` (or the environment variable `HNSW_BENCH_DATASET_URL`) does the same, and a file already in the cache folder is never downloaded again.

`bench/python/publish_comparison.py` publishes each run to the same **`benchmarks` branch** as the nightly benchmark, under `real-datasets/`: the [latest results](https://github.com/praknig/hnsw_lite/blob/benchmarks/real-datasets/README.md) (each dataset's summary table, charts and raw JSON, with the commit and the workflow run that produced them) and every earlier run in `real-datasets/history/`. A dataset rerun on its own keeps the others' latest results. The two workflows share a concurrency group, so they never push to the branch at the same time. Like the nightly benchmark, it uses `BENCHMARK_RUNNER` when set.

**What the published numbers show:** a fair comparison on one machine class (each summary names the CPU, the core count and every library's version), on data with real structure. **What they don't:** your hardware, your data or your filters. A shared cloud runner also adds some noise, so treat differences of a few percent as level, and rerun on your own machine (`python bench/python/compare.py --dataset sift-128-euclidean`) before deciding on numbers alone.

**Locally:** `python bench/python/publish_comparison.py --results <compare.py's --out> --site <a directory> --commit <sha>` builds the same pages from your own runs.

## Publishing results

State the machine (CPU, cores, memory), the operating system, the library versions (the summary lists them), the dataset and every setting, and publish the JSON with the charts, so anyone can reproduce or reanalyze them. The standard public comparison is [ann-benchmarks](https://github.com/erikbern/ann-benchmarks): an adapter for hnsw-lite there is a natural next step.