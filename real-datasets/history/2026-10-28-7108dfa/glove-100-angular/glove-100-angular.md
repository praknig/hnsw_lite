# glove-100-angular

1,183,514 vectors of 100 dimensions (cosine), 10,000 queries; M 16, ef_construction 200; AMD EPYC 7763 64-Core Processor, 4 threads; median of 3 runs; 2026-10-28.

Queries per second at a recall@10 of at least 0.90 / 0.95 / 0.99 (one query per call, one thread), and build cost:

| Library | q/s @ 0.90 | q/s @ 0.95 | q/s @ 0.99 | batch q/s @ 0.95 | build (s) | memory (MB) | index file (MB) |
|---|---|---|---|---|---|---|---|
| hnsw-lite | 1,063 | — | — | — | 261.0 | 1263 | 1027 |
| hnswlib | 1,411 | — | — | — | 224.4 | 735 | 619 |
| faiss-hnsw | 1,410 | — | — | — | 186.4 | 653 | 614 |
| usearch | 529 | — | — | — | 360.8 | 774 | 619 |
| faiss-ivfpq | — | — | — | — | 37.6 | 85 | 15 |

Filtered search: queries per second at a recall@10 of at least 0.95 (one thread). The filter keeps vectors whose bucket (id % 100) is below the percentage, so matches are spread evenly; hnsw-lite filters on metadata stored in the index, hnswlib with a Python callback, FAISS with an ID selector:

| Library | 1% matching | 10% matching | 30% matching |
|---|---|---|---|
| hnsw-lite | 234 | — | — |
| hnswlib | 63.03 | — | — |
| faiss-hnsw | — | — | — |
| faiss-ivfpq | — | — | — |

— : the library did not reach that recall in the sweep, or does not support it.

Versions: hnsw-lite 0.1.0, hnswlib 0.8.0, faiss 1.15.1, usearch 2.26.6, numpy 2.5.4; Python 3.12.15.
