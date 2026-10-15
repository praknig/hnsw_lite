# Compression and index types plan

The plan for items 4.1 to 4.4 of the [feature catalog](feature-catalog.md): quantization (SQ8, binary and product quantization), shorter vectors (Matryoshka truncation), an IVF index, and a disk-based index in the style of DiskANN.

## Contents

- [Why these belong together](#why-these-belong-together)
- [Phases](#phases)
- [Decisions](#decisions)
- [File format version 3](#file-format-version-3)
- [Testing](#testing)
- [How the work is organized](#how-the-work-is-organized)
- [Risks](#risks)
- [Status](#status)

## Why these belong together

All four share two foundations, built once in phase Q0:

- **A codec layer:** how each vector is stored (full floats, 8-bit codes, bits, or product-quantized codes) and how a distance is computed from that form. Today every vector is stored as full floats; the codec makes that one choice among several.
- **Training:** SQ8 learns each dimension's value range, while PQ and IVF learn centroids with k-means. Training happens once, from sample data, before codes can be computed.

## Phases

| Phase | Contents | Effort |
|---|---|---|
| **Q0. Foundations** | A `VectorCodec` interface; a training API (explicit `train(sample)`, or automatic from the first vectors added); a k-means routine (deterministic for a given seed, parallel); file format version 3; a benchmark reporting recall, memory and build time per setting | 3–4 days |
| **Q1. SQ8** | 8 bits per dimension, with per-dimension ranges learned from data. Distances are computed between the float query and the byte codes (asymmetric distance, more accurate than comparing codes), with SIMD kernels for AVX2, AVX-512 and NEON and a plain C++ fallback. About 4 times less memory for vectors | 3–4 days |
| **Q2. Matryoshka truncation** | Index only the first N dimensions of embeddings trained for it (an option per index); queries are truncated the same way, and cosine vectors re-normalized after truncation | 1 day |
| **Q3. Binary quantization** | 1 bit per dimension (its sign), with Hamming distance computed by `popcount`. About 32 times less memory, so re-ranking at full precision is essential for good results | 2–3 days |
| **Q4. Product quantization (PQ)** | Each vector is split into *m* parts, each coded as one of 256 centroids learned per part; distances use lookup tables built once per query. 16 to 64 times less memory | 1 week |
| **Q5. IVF index** | A second index type: k-means divides the space into cells, each holding a list of its vectors, and a search visits the `nprobe` nearest cells. First with full vectors in the lists, then combined with PQ as IVF-PQ (as in Faiss). It shares metadata, filters, locking, saving, deadlines and the record functions with the existing indexes | 1 week |
| **Q6. Disk-based index** | A Vamana graph (as in DiskANN) whose full vectors and links live on disk in 4 KB blocks, with PQ codes in memory to guide the search and disk reads batched per search step. Read-only in its first version: built, saved, loaded and searched, with changes requiring a rebuild. Updating it in place (as FreshDiskANN does) would be a later project. Full-precision vectors for re-ranking move to disk here (decision D6) | 4–6 weeks |

**Total: roughly 9 to 12 weeks.** The catalog's earlier estimate of "3+ weeks" for the disk-based index was optimistic: the on-disk layout, batched reads on Linux, macOS and Windows, and building indexes larger than memory each take real work.

Phases Q1 to Q4 are independent of each other once Q0 exists; Q5 uses Q0's k-means and, for IVF-PQ, Q4; Q6 uses Q4 (PQ codes guide its search) and Q0's codec layer.

## Decisions

| # | Decision | Options | Status |
|---|---|---|---|
| D6 | Where full-precision vectors live when an index is quantized (for re-ranking and `get_vector`) | (a) in memory as well: exact re-ranking, little memory saved; (b) not kept: the most memory saved, no re-ranking, `get_vector` returns decoded approximations; (c) on disk, read only for re-ranking candidates | **Decided: (a) first, then (c) with Q6.** Option (b) may later become a per-index setting |
| D7 | How HNSW builds its graph when vectors are quantized | From full floats (a better graph) or from codes (less memory while building) | **Decided: from floats** whenever they are available (D6 a or c) |
| D8 | When training happens | An explicit `train(sample)` before adding; automatic, on the first N vectors; or both | **Decided: both**: automatic by default, explicit for control |
| D9 | What `get_vector` returns for a truncated (Matryoshka) index | The full vector as given, or only the indexed dimensions | **Decided: the full vector**, keeping the existing promise that `get_vector` returns a vector exactly as given |
| D10 | IVF's first version | Full vectors in the lists, or IVF-PQ from the start | **Decided: full vectors first**, then IVF-PQ once Q4 exists |

Decisions D3 (Flat batch modes) and D4 (SQ8 as an option per index) from the [hardening and performance plan](hardening-and-performance-plan.md#decisions-needed) still apply; D4 is implemented by Q1.

## File format version 3

Version 2 already added the original vectors of cosine indexes, so these features need **version 3**:

- a section describing the codec and its trained state (SQ8 ranges, PQ codebooks), and the codes themselves;
- the Matryoshka setting (the number of indexed dimensions);
- new index kinds for IVF (kind 3) and the disk-based index (kind 4). The disk-based index is itself a file layout, so it is opened in place rather than read into memory.

Versions 1 and 2 stay readable, and new golden files for version 3 join the frozen ones for versions 1 and 2, as before.

## Testing

Each phase follows the usual pattern: test scenarios written first, edge cases, out-of-memory sweeps, ThreadSanitizer and AddressSanitizer, coverage, and documentation, with every claim checked. Compression adds:

- **Quality thresholds:** recall against exact results, for each codec, metric and setting, with and without re-ranking.
- **Fixed-answer tests** for codes and distances, so that every CPU's SIMD code produces identical codes (needed for the golden files to be identical on every platform).
- **Training edge cases:** too few vectors to train on, all-identical vectors, constant dimensions, values outside the trained range after training (clamped), and training on a filtered or empty sample.
- **For the disk-based index:** files larger than the memory available to the test, damaged and truncated files, and reads failing partway.

One limitation: the development environment used so far can measure recall, memory and correctness reliably, but not speed (it is a shared single core). The speed benefits, especially of the SIMD kernels and the disk-based index, must be measured on a quiet machine with dedicated cores.

## How the work is organized

All phases are developed on one branch, `feature/compression-and-index-types`, with **one commit per phase** (the branch's first commit holds this plan). Every commit is complete on its own: it builds, passes every test, and includes its documentation, so the branch is never broken and each phase can be reviewed by itself (`git show` of one commit).

Because the branch will be long-lived, `master` is merged into it whenever `master` changes, and CI runs on every push.

## Progress notes

**Q0 (done).** `kmeans` (k-means++ seeding, Lloyd's refinement, empty clusters refilled from the farthest point) and the `VectorCodec` interface, with SQ8 as its first implementation in plain C++, so the interface is proven by a real codec. Both are deterministic across CPUs, compilers and thread counts. `kmeans.cpp` and `codec.cpp` are compiled with `-ffp-contract=off` (GCC and Clang): built for ARM64 without it, SQ8 distances differed in their last bits from x86, while with it every result is bit-identical (checked by fingerprinting results on x86, ARM with the flag and ARM without). **Q1** therefore covers SQ8's SIMD kernels, its integration into the indexes (training, encoding on add, distances during search, re-ranking) and file format version 3.

## Risks

| Risk | Mitigation |
|---|---|
| A long-lived branch drifts from `master` | Merge `master` in regularly; keep every commit green |
| Quantization lowers recall more than expected on some data | Re-ranking at full precision (D6 a); recall measured per codec and metric before each phase is accepted |
| SIMD kernels for codes differ between CPUs | Fixed-answer tests on every CI platform, including the emulated CPUs |
| The disk-based index behaves differently on Windows, macOS and Linux | A small file-access layer with one implementation per platform, tested on every CI job |
| Speed claims cannot be verified in the development environment | Every speed claim is measured on a quiet machine before it is documented |

## Status

| Phase | Status |
|---|---|
| Q0. Foundations | **Done**: k-means, the codec interface and SQ8 in plain C++ (see below) |
| Q1. SQ8 | Not started |
| Q2. Matryoshka truncation | Not started |
| Q3. Binary quantization | Not started |
| Q4. Product quantization | Not started |
| Q5. IVF index | Not started |
| Q6. Disk-based index | Not started |