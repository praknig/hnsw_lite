# hnsw-lite benchmarks

Published by the nightly workflow (`.github/workflows/benchmark.yml`) from `bench/python/stress.py`. Every run is kept in `runs/`; this page shows the newest and the trends.

Comparisons on real datasets (SIFT1M, GloVe) against hnswlib, FAISS and USearch: [real-datasets](real-datasets/README.md).

## Latest: 2026-11-11 09:11 UTC, commit [`c679162`](https://github.com/praknig/hnsw_lite/commit/c6791625408c7f7adffbb372ea21758f7d7e1774)

**7,900,000 vectors of 128 dimensions**, M 16, ef_construction 200; 4 cores; Python 3.12.16. **16 of 17 checks passed** ([run](https://github.com/praknig/hnsw_lite/actions/runs/42988551364)).

![latest-recall.png](latest-recall.png)

![latest-load.png](latest-load.png)

![latest-build.png](latest-build.png)

## Trends

![history](history.png)

Runs on different machines or of different sizes are not directly comparable (sizes are marked on the chart when they vary).

| Date (UTC) | Commit | Vectors | Inserts/s | Recall@10 (ef 128) | Queries/s (ef 128) | p99 ms | Checks |
|---|---|---|---|---|---|---|---|
| 2026-11-11 09:11 | `c679162` | 7,900,000 | 4,781 | 0.872 | 2,788 | 1.15 | 16/17 **FAIL** |
| 2026-11-10 09:28 | `c679162` | 7,900,000 | 5,182 | 0.872 | 3,368 | 0.79 | 16/17 **FAIL** |
| 2026-11-09 09:26 | `c679162` | 8,000,000 | 4,623 | 0.873 | 2,691 | 1.10 | 16/17 **FAIL** |
| 2026-11-08 08:43 | `c679162` | 7,800,000 | 6,494 | 0.872 | 3,435 | 0.77 | 16/17 **FAIL** |
| 2026-11-07 08:34 | `c679162` | 8,000,000 | 4,673 | 0.873 | 2,956 | 0.97 | 16/17 **FAIL** |
| 2026-11-06 09:10 | `c679162` | 7,900,000 | 6,356 | 0.872 | 3,996 | 0.81 | 16/17 **FAIL** |
| 2026-11-05 09:15 | `c679162` | 7,900,000 | 5,224 | 0.872 | 3,349 | 0.91 | 16/17 **FAIL** |
| 2026-11-04 08:54 | `c679162` | 7,900,000 | 4,465 | 0.872 | 3,189 | 1.04 | 16/17 **FAIL** |
| 2026-11-03 09:11 | `7f587b1` | 7,900,000 | 7,961 | 0.869 | 3,971 | 0.76 | 16/17 **FAIL** |
| 2026-11-02 09:04 | `7f587b1` | 8,000,000 | 6,347 | 0.874 | 3,260 | 0.85 | 16/17 **FAIL** |
| 2026-11-01 08:50 | `5fdfda5` | 7,900,000 | 6,853 | 0.874 | 3,333 | 0.89 | 16/17 **FAIL** |
| 2026-10-31 08:14 | `d411208` | 8,000,000 | 5,615 | 0.869 | 2,953 | 0.96 | 16/17 **FAIL** |
| 2026-10-30 08:23 | `7108dfa` | 7,900,000 | 5,510 | 0.869 | 2,948 | 0.97 | 16/17 **FAIL** |
| 2026-10-29 08:19 | `7108dfa` | 8,000,000 | 5,433 | 0.874 | 2,516 | 1.21 | 16/17 **FAIL** |
| 2026-10-28 08:11 | `9f53b0b` | 8,000,000 | 6,106 | 0.874 | 2,778 | 1.06 | 16/17 **FAIL** |
| 2026-10-27 07:29 | `1fc4117` | 8,000,000 | 6,060 | 0.869 | 3,249 | 0.78 | 16/17 **FAIL** |
| 2026-10-26 11:39 | `1fc4117` | 7,900,000 | 5,881 | 0.874 | 2,865 | 1.05 | 16/17 **FAIL** |
| 2026-10-26 07:41 | `320d6e5` | 7,900,000 | 5,898 | 0.874 | 2,713 | 1.06 | 16/17 **FAIL** |

## Failed checks in the latest run

- recall@10 at ef 128 is at least 0.9: 0.872
