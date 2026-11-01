# sift-128-euclidean

1,000,000 vectors of 128 dimensions (l2), 10,000 queries; M 16, ef_construction 200; AMD EPYC 7763 64-Core Processor, 4 threads; median of 3 runs; 2026-11-01.

Queries per second at a recall@10 of at least 0.90 / 0.95 / 0.99 (one query per call, one thread), and build cost:

| Library | q/s @ 0.90 | q/s @ 0.95 | q/s @ 0.99 | batch q/s @ 0.95 | build (s) | memory (MB) | index file (MB) |
|---|---|---|---|---|---|---|---|
| hnsw-lite | 9,955 | 5,970 | 2,303 | 15,291 | 159.0 | 701 | 586 |
| hnswlib | 11,806 | 7,013 | 2,604 | 17,033 | 147.8 | 732 | 630 |
| faiss-hnsw | 10,508 | 6,335 | 3,539 | 18,383 | 156.7 | 665 | 626 |
| usearch | 3,669 | 2,330 | 905 | 8,350 | 248.3 | 732 | 630 |
| faiss-ivfpq | — | — | — | — | 37.4 | 156 | 25 |

Filtered search: queries per second at a recall@10 of at least 0.95 (one thread). The filter keeps vectors whose bucket (id % 100) is below the percentage, so matches are spread evenly; hnsw-lite filters on metadata stored in the index, hnswlib with a Python callback, FAISS with an ID selector:

| Library | 1% matching | 10% matching | 30% matching |
|---|---|---|---|
| hnsw-lite | — | 874 | 2,229 |
| hnswlib | 272 | 1,935 | 2,696 |
| faiss-hnsw | — | — | 3,176 |
| faiss-ivfpq | — | — | — |

— : the library did not reach that recall in the sweep, or does not support it.

Versions: hnsw-lite 0.1.0, hnswlib 0.8.0, faiss 1.15.1, usearch 2.26.6, numpy 2.5.4; Python 3.12.15.
