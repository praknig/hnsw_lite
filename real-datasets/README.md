# Real-dataset comparisons

hnsw-lite against hnswlib, FAISS (HNSW and IVF-PQ) and USearch on standard ann-benchmarks datasets, whose files include the exact neighbors of every query. Published by `.github/workflows/real-benchmarks.yml` from `bench/python/compare.py` ([methodology](https://github.com/praknig/hnsw_lite/blob/master/docs/benchmarks.md)). Graph libraries share M and ef_construction; each library's search beam is swept, so curves compare at equal recall, never at equal settings. Every run is kept in `history/`.

[Nightly benchmark](../README.md)

## glove-100-angular

Published 2026-10-26 02:28 UTC from commit [`7503b2c`](https://github.com/praknig/hnsw_lite/commit/7503b2c6ae580c0d602de9905678dc24902ede2e) ([run](https://github.com/praknig/hnsw_lite/actions/runs/40252597148)).

1,183,514 vectors of 100 dimensions (cosine), 10,000 queries; M 16, ef_construction 200; AMD EPYC 7763 64-Core Processor, 4 threads; median of 3 runs; 2026-10-26.

Queries per second at a recall@10 of at least 0.90 / 0.95 / 0.99 (one query per call, one thread), and build cost:

| Library | q/s @ 0.90 | q/s @ 0.95 | q/s @ 0.99 | batch q/s @ 0.95 | build (s) | memory (MB) | index file (MB) |
|---|---|---|---|---|---|---|---|
| hnsw-lite | 966 | — | — | — | 290.7 | 1260 | 1027 |
| hnswlib | 1,287 | — | — | — | 249.1 | 735 | 619 |
| faiss-hnsw | 1,283 | — | — | — | 210.9 | 653 | 614 |
| usearch | 469 | — | — | — | 395.9 | 774 | 619 |
| faiss-ivfpq | — | — | — | — | 39.9 | 85 | 15 |

Filtered search: queries per second at a recall@10 of at least 0.95 (one thread). The filter keeps vectors whose bucket (id % 100) is below the percentage, so matches are spread evenly; hnsw-lite filters on metadata stored in the index, hnswlib with a Python callback, FAISS with an ID selector:

| Library | 1% matching | 10% matching | 30% matching |
|---|---|---|---|
| hnsw-lite | 98.46 | — | — |
| hnswlib | 57.86 | — | — |
| faiss-hnsw | — | — | — |
| faiss-ivfpq | — | — | — |

— : the library did not reach that recall in the sweep, or does not support it.

Versions: hnsw-lite 0.1.0, hnswlib 0.8.0, faiss 1.15.1, usearch 2.26.6, numpy 2.5.4; Python 3.12.15.

![glove-100-angular-filtered](glove-100-angular/glove-100-angular-filtered.png)

![glove-100-angular-recall-qps](glove-100-angular/glove-100-angular-recall-qps.png)

## sift-128-euclidean

Published 2026-10-26 02:28 UTC from commit [`7503b2c`](https://github.com/praknig/hnsw_lite/commit/7503b2c6ae580c0d602de9905678dc24902ede2e) ([run](https://github.com/praknig/hnsw_lite/actions/runs/40252597148)).

1,000,000 vectors of 128 dimensions (l2), 10,000 queries; M 16, ef_construction 200; AMD EPYC 7763 64-Core Processor, 4 threads; median of 3 runs; 2026-10-26.

Queries per second at a recall@10 of at least 0.90 / 0.95 / 0.99 (one query per call, one thread), and build cost:

| Library | q/s @ 0.90 | q/s @ 0.95 | q/s @ 0.99 | batch q/s @ 0.95 | build (s) | memory (MB) | index file (MB) |
|---|---|---|---|---|---|---|---|
| hnsw-lite | 9,566 | 5,533 | 2,117 | 13,824 | 176.3 | 703 | 586 |
| hnswlib | 11,904 | 6,963 | 2,732 | 14,619 | 166.7 | 732 | 630 |
| faiss-hnsw | 11,890 | 7,103 | 3,969 | 15,825 | 174.4 | 656 | 626 |
| usearch | 4,247 | 2,631 | 1,095 | 8,731 | 244.8 | 733 | 630 |
| faiss-ivfpq | — | — | — | — | 37.6 | 154 | 25 |

Filtered search: queries per second at a recall@10 of at least 0.95 (one thread). The filter keeps vectors whose bucket (id % 100) is below the percentage, so matches are spread evenly; hnsw-lite filters on metadata stored in the index, hnswlib with a Python callback, FAISS with an ID selector:

| Library | 1% matching | 10% matching | 30% matching |
|---|---|---|---|
| hnsw-lite | — | 614 | 2,159 |
| hnswlib | 267 | 1,973 | 2,747 |
| faiss-hnsw | — | — | 3,072 |
| faiss-ivfpq | — | — | — |

— : the library did not reach that recall in the sweep, or does not support it.

Versions: hnsw-lite 0.1.0, hnswlib 0.8.0, faiss 1.15.1, usearch 2.26.6, numpy 2.5.4; Python 3.12.15.

![sift-128-euclidean-filtered](sift-128-euclidean/sift-128-euclidean-filtered.png)

![sift-128-euclidean-recall-qps](sift-128-euclidean/sift-128-euclidean-recall-qps.png)
