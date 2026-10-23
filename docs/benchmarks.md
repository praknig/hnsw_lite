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

## Publishing results

State the machine (CPU, cores, memory), the operating system, the library versions (the summary lists them), the dataset and every setting, and publish the JSON with the charts, so anyone can reproduce or reanalyze them. The standard public comparison is [ann-benchmarks](https://github.com/erikbern/ann-benchmarks): an adapter for hnsw-lite there is a natural next step.
