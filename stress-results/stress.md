# hnsw-lite stress test

7,900,000 vectors of 128 dimensions (1000 clusters), M 16, ef_construction 200; 4 cores; Python 3.12.15.

## Build

7,900,000 vectors in 1285 s: **6,148 vectors/s**; index memory 5,658 MB (751 bytes per vector).

## Recall

| ef | recall@10 | queries/s (1 thread) |
|---|---|---|
| 32 | 0.565 | 5,344 |
| 64 | 0.750 | 4,009 |
| 128 | 0.869 | 2,590 |
| 256 | 0.935 | 1,661 |

## Load

ef 128, 60.0 s per level.

| clients | queries/s | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| 1 | 2,510 | 0.37 | 0.59 | 0.66 | 1.47 |
| 2 | 4,761 | 0.39 | 0.61 | 0.69 | 1.39 |
| 4 | 6,307 | 0.60 | 0.92 | 1.04 | 5.51 |

## Mixed (searches during changes)

60.0 s, 4 searchers, 2 writers: 105,694 searches (1,756/s, p99 9.78 ms), 15,977 inserts, 9,664 updates, 6,452 removes (533 changes/s). Recall@10 after: 0.854; after upkeep: 0.867.

## Crash recovery

Writer killed after 5,105 acknowledged writes; recovered 5,105 documents in 0.9 s.

## Checks

| Check | Result | Detail |
|---|---|---|
| every vector stored | pass | 7,900,000 of 7,900,000 |
| insert speed holds up as the index grows | pass | slowest batch 4,151/s, fastest 18,275/s |
| recall@10 at ef 128 is at least 0.9 | **FAIL** | 0.869 |
| recall grows with ef | pass | 0.565 -> 0.750 -> 0.869 -> 0.935 |
| no errors with 1 clients | pass |  |
| no errors with 2 clients | pass |  |
| no errors with 4 clients | pass |  |
| throughput grows with clients (searches run in parallel) | pass | 2,510/s with 1, 6,307/s with 4 |
| no errors during searches and changes | pass |  |
| the final count is exact | pass | 7,909,525 stored, 7,909,525 expected |
| sampled ids are present or absent as expected | pass | 5,000 sampled, wrong: [] |
| sampled vectors equal their last write | pass | wrong: [] |
| recall@10 stays at least 0.85 under churn (ef 128) | pass | 0.854 |
| upkeep leaves no vector unreachable | pass | 17613 before, 0 after (14.2 s) |
| the writer was killed mid-write | pass | exit code -9, 5,105 writes |
| every acknowledged write survived the crash | pass | missing: [] |
| recovered vectors are the ones written | pass | 511 checked, wrong: [] |
