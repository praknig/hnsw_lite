#!/usr/bin/env python3
"""Benchmarks hnsw-lite against other vector search libraries.

    pip install hnsw-lite-lib hnswlib faiss-cpu usearch h5py matplotlib psutil
    python bench/python/compare.py --dataset glove-100-angular
    python bench/python/compare.py --dataset sift-128-euclidean --libraries hnsw-lite hnswlib faiss-hnsw
    python bench/python/compare.py --synthetic 100000,96 --metric cosine        # no download

Datasets are the standard ann-benchmarks files (with exact neighbors), downloaded
once into --cache. Each library runs in its own process, so memory is measured
cleanly. Measured: recall@10 against queries per second (one query per call on one
thread, and one batch call on all threads), build time, memory, index size, and
filtered search at several selectivities using each library's own filtering.
Results go to --out: a JSON file of raw numbers, PNG charts and a Markdown summary.

Fairness: graph libraries share M and ef_construction, and each library's search
beam is swept, so curves compare at equal recall, never at equal settings. Every
library is called from Python the same way, so per-call overhead is similar.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

DATASETS = ["glove-100-angular", "sift-128-euclidean", "fashion-mnist-784-euclidean", "nytimes-256-angular",
            "gist-960-euclidean", "deep-image-96-angular", "glove-25-angular", "mnist-784-euclidean"]
LIBRARIES = ["hnsw-lite", "hnswlib", "faiss-hnsw", "usearch", "faiss-ivfpq"]
EF_SWEEP = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
NPROBE_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128]
SELECTIVITIES = [1, 10, 30]  # percent of vectors matching the filter
K = 10


# ---------------------------------------------------------------- data

def exact_neighbors(train: np.ndarray, test: np.ndarray, metric: str, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Exact top-k by brute force (in chunks), optionally among the masked rows only."""
    ids = np.arange(len(train)) if mask is None else np.flatnonzero(mask)
    base = train[ids]
    if metric == "cosine":
        base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-30)
        test = test / np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-30)
    out = np.empty((len(test), k), dtype=np.int64)
    norms = (base ** 2).sum(axis=1)
    for s in range(0, len(test), 256):
        q = test[s:s + 256]
        d = -q @ base.T if metric == "cosine" else norms[None, :] - 2 * q @ base.T
        top = np.argpartition(d, min(k, d.shape[1] - 1), axis=1)[:, :k]
        order = np.take_along_axis(d, top, axis=1).argsort(axis=1)
        out[s:s + 256] = ids[np.take_along_axis(top, order, axis=1)]
    return out


def load_dataset(args) -> Path:
    """The dataset as an .npz in the cache: train, test, neighbors, metric, and the
    filter buckets with their exact filtered neighbors."""
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        n, dim = (int(x) for x in args.synthetic.split(","))
        name = f"synthetic-{n}-{dim}-{args.metric}"
    else:
        name = args.dataset
    npz = cache / f"{name}-q{args.queries}.npz"
    if npz.exists():
        return npz
    if args.synthetic:
        rng = np.random.default_rng(42)
        centers = rng.normal(0, 4, (max(10, n // 1000), dim)).astype(np.float32)
        train = (centers[rng.integers(0, len(centers), n)] + rng.normal(0, 1, (n, dim))).astype(np.float32)
        test = (centers[rng.integers(0, len(centers), args.queries)] + rng.normal(0, 1, (args.queries, dim))).astype(np.float32)
        metric = args.metric
        print(f"computing exact neighbors for {name}...", flush=True)
        neighbors = exact_neighbors(train, test, metric, K)
    else:
        import h5py
        hdf5 = cache / f"{name}.hdf5"
        if not hdf5.exists():
            url = f"http://ann-benchmarks.com/{name}.hdf5"
            print(f"downloading {url} ...", flush=True)
            urllib.request.urlretrieve(url, hdf5)
        with h5py.File(hdf5, "r") as f:
            train = np.asarray(f["train"], dtype=np.float32)
            test = np.asarray(f["test"], dtype=np.float32)[: args.queries]
            neighbors = np.asarray(f["neighbors"])[: args.queries, :K]
            metric = {"angular": "cosine", "euclidean": "l2"}[f.attrs["distance"]]
    # Filter buckets: uncorrelated with position (id % 100), and exact filtered neighbors.
    buckets = (np.arange(len(train)) % 100).astype(np.int64)
    fq = test[: min(len(test), args.filtered_queries)]
    filtered = {}
    for pct in SELECTIVITIES:
        print(f"computing exact neighbors for the {pct}% filter...", flush=True)
        filtered[f"filtered_{pct}"] = exact_neighbors(train, fq, metric, K, buckets < pct)
    np.savez(npz, train=train, test=test, neighbors=neighbors, metric=metric, buckets=buckets, **filtered)
    return npz


# ---------------------------------------------------------------- libraries

class Library:
    """One library's index: build, a search beam, single and batch searches, filters."""
    filtered = True

    def __init__(self, dim: int, metric: str, args):
        self.dim, self.metric, self.args = dim, metric, args

    def sweep(self):
        return EF_SWEEP

    def save(self, path: str) -> None:
        raise NotImplementedError

    def prepare_filters(self, x, threads) -> None:
        """Before the filtered searches (not measured): what the library's filtering needs."""


class HnswLite(Library):
    def build(self, x, threads):
        import hnsw_lite_lib as hl
        self.hl = hl
        p = hl.HnswParams()
        p.M, p.ef_construction = self.args.M, self.args.ef_construction
        self.params = p
        self.index = hl.HnswIndex(self.dim, hl.Metric.COSINE if self.metric == "cosine" else hl.Metric.L2, p)
        self.index.add(np.arange(len(x)), x, threads=threads)  # a plain index, like the others'

    def prepare_filters(self, x, threads):
        # The other libraries filter through an array outside the index; hnsw-lite filters on
        # metadata stored in it. A second index carries it, built outside the measurements, so
        # the plain index's memory, build time and size compare like for like.
        hl = self.hl
        self.index = hl.HnswIndex(self.dim, hl.Metric.COSINE if self.metric == "cosine" else hl.Metric.L2, self.params)
        self.index.add(np.arange(len(x)), x, metadata=[{"bucket": int(i % 100)} for i in range(len(x))], threads=threads)

    def set_beam(self, ef):
        self.ef = ef

    def query(self, q, k):
        return self.index.search(q, k=k, ef=self.ef)[0]

    def batch(self, qs, k, threads):
        return self.index.search_batch(qs, k=k, ef=self.ef, threads=threads)[0]

    def query_filtered(self, q, k, pct, buckets):
        return self.index.search(q, k=k, ef=self.ef, filter=self.hl.Filter.lt("bucket", pct))[0]

    def save(self, path):
        self.index.save(path)


class Hnswlib(Library):
    def build(self, x, threads):
        import hnswlib
        self.index = hnswlib.Index(space="cosine" if self.metric == "cosine" else "l2", dim=self.dim)
        self.index.init_index(max_elements=len(x), M=self.args.M, ef_construction=self.args.ef_construction, random_seed=42)
        self.index.add_items(x, np.arange(len(x)), num_threads=threads)

    def set_beam(self, ef):
        self.index.set_ef(ef)

    def query(self, q, k):
        return self.index.knn_query(q, k=k, num_threads=1)[0][0]

    def batch(self, qs, k, threads):
        return self.index.knn_query(qs, k=k, num_threads=threads)[0]

    def query_filtered(self, q, k, pct, buckets):
        # hnswlib's filtering: a Python callback per candidate.
        return self.index.knn_query(q, k=k, num_threads=1, filter=lambda i: buckets[i] < pct)[0][0]

    def save(self, path):
        self.index.save_index(path)


class FaissHnsw(Library):
    def build(self, x, threads):
        import faiss
        self.faiss = faiss
        faiss.omp_set_num_threads(threads)
        mt = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2  # cosine: normalized vectors, inner product
        self.index = faiss.IndexHNSWFlat(self.dim, self.args.M, mt)
        self.index.hnsw.efConstruction = self.args.ef_construction
        self.index.add(self._prep(x))
        self.selectors = {}

    def _prep(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.metric == "cosine":
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
        return x

    def set_beam(self, ef):
        self.index.hnsw.efSearch = ef
        self.ef = ef

    def query(self, q, k):
        self.faiss.omp_set_num_threads(1)
        return self.index.search(self._prep(q[None, :]), k)[1][0]

    def batch(self, qs, k, threads):
        self.faiss.omp_set_num_threads(threads)
        return self.index.search(self._prep(qs), k)[1]

    def query_filtered(self, q, k, pct, buckets):
        self.faiss.omp_set_num_threads(1)
        if pct not in self.selectors:
            self.selectors[pct] = self.faiss.IDSelectorBatch(np.flatnonzero(buckets < pct).astype(np.int64))
        params = self.faiss.SearchParametersHNSW(sel=self.selectors[pct], efSearch=self.ef)
        return self.index.search(self._prep(q[None, :]), k, params=params)[1][0]

    def save(self, path):
        self.faiss.write_index(self.index, path)


class FaissIvfPq(FaissHnsw):
    """IVF-PQ: the memory-saving reference (16 bytes per vector, re-ranked by nothing)."""

    def build(self, x, threads):
        import faiss
        self.faiss = faiss
        faiss.omp_set_num_threads(threads)
        mt = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
        nlist = max(16, int(4 * np.sqrt(len(x))))
        m = next(m for m in (16, 8, 4, 2, 1) if self.dim % m == 0)
        self.index = faiss.IndexIVFPQ(faiss.IndexFlat(self.dim, mt), self.dim, nlist, m, 8, mt)
        xs = self._prep(x)
        self.index.train(xs[np.random.default_rng(0).choice(len(xs), min(len(xs), 100 * nlist), replace=False)])
        self.index.add(xs)
        self.selectors = {}

    def sweep(self):
        return NPROBE_SWEEP

    def set_beam(self, nprobe):
        self.index.nprobe = nprobe
        self.ef = nprobe

    def query_filtered(self, q, k, pct, buckets):
        self.faiss.omp_set_num_threads(1)
        if pct not in self.selectors:
            self.selectors[pct] = self.faiss.IDSelectorBatch(np.flatnonzero(buckets < pct).astype(np.int64))
        params = self.faiss.SearchParametersIVF(sel=self.selectors[pct], nprobe=self.ef)
        return self.index.search(self._prep(q[None, :]), k, params=params)[1][0]


class USearch(Library):
    filtered = False  # no filtered search in its Python API

    def build(self, x, threads):
        from usearch.index import Index
        self.index = Index(ndim=self.dim, metric="cos" if self.metric == "cosine" else "l2sq", dtype="f32",
                           connectivity=self.args.M, expansion_add=self.args.ef_construction)
        self.index.add(np.arange(len(x)), x, threads=threads)

    def set_beam(self, ef):
        self.index.expansion_search = ef

    def query(self, q, k):
        return self.index.search(q, k).keys

    def batch(self, qs, k, threads):
        return self.index.search(qs, k, threads=threads).keys

    def save(self, path):
        self.index.save(path)


ADAPTERS = {"hnsw-lite": HnswLite, "hnswlib": Hnswlib, "faiss-hnsw": FaissHnsw, "usearch": USearch, "faiss-ivfpq": FaissIvfPq}


# ---------------------------------------------------------------- one library (a worker process)

def recall(found: np.ndarray, truth: np.ndarray) -> float:
    hits = sum(len(set(np.asarray(f).tolist()) & set(t.tolist())) for f, t in zip(found, truth))
    return hits / truth.size


def worker(args) -> dict:
    import psutil
    data = np.load(args.data)
    train, test, truth, metric, buckets = data["train"], data["test"], data["neighbors"], str(data["metric"]), data["buckets"]
    threads = args.threads or os.cpu_count() or 1
    proc = psutil.Process()
    lib = ADAPTERS[args.library](train.shape[1], metric, args)
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    lib.build(train, threads)
    build = time.perf_counter() - t0
    memory = proc.memory_info().rss - rss0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "index")
        try:
            lib.save(path)
            size = os.path.getsize(path)
        except NotImplementedError:
            size = None
    result = {"library": args.library, "build_s": build, "memory_bytes": memory, "index_bytes": size, "threads": threads, "curve": [], "filtered": []}
    for beam in lib.sweep():
        lib.set_beam(beam)
        for q in test[:20]:  # warm-up
            lib.query(q, K)
        singles, batches = [], []
        for _ in range(args.repeat):  # the median of several runs: timings on shared machines vary
            t0 = time.perf_counter()
            found = [lib.query(q, K) for q in test]
            singles.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            batch_found = lib.batch(test, K, threads)
            batches.append(time.perf_counter() - t0)
        single, batch = float(np.median(singles)), float(np.median(batches))
        r = recall(found, truth)
        result["curve"].append({"beam": beam, "recall": r, "qps": len(test) / single, "batch_qps": len(test) / batch,
                                "batch_recall": recall(batch_found, truth)})
        print(f"  {args.library:11s} beam {beam:4d}: recall {r:.4f}, {len(test) / single:9.0f} q/s single, {len(test) / batch:9.0f} q/s batch", flush=True)
        if r >= 0.999:
            break
    if args.with_filters and lib.filtered:
        lib.prepare_filters(train, threads)
        for pct in SELECTIVITIES:
            ftruth = data[f"filtered_{pct}"]
            fq = test[: len(ftruth)]
            for beam in [b for b in lib.sweep() if b >= 16][:6]:
                lib.set_beam(beam)
                times = []
                for _ in range(args.repeat):
                    t0 = time.perf_counter()
                    found = [lib.query_filtered(q, K, pct, buckets) for q in fq]
                    times.append(time.perf_counter() - t0)
                elapsed = float(np.median(times))
                r = recall(found, ftruth)
                result["filtered"].append({"selectivity": pct, "beam": beam, "recall": r, "qps": len(fq) / elapsed})
                print(f"  {args.library:11s} {pct:2d}% filter, beam {beam:4d}: recall {r:.4f}, {len(fq) / elapsed:9.0f} q/s", flush=True)
    return result


# ---------------------------------------------------------------- report

def best_qps(curve: list, target: float, key: str = "qps") -> float | None:
    ok = [p[key] for p in curve if p["recall"] >= target]
    return max(ok) if ok else None


def report(results: list, meta: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    name = meta["dataset"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for res in results:
        c = res["curve"]
        axes[0].plot([p["recall"] for p in c], [p["qps"] for p in c], marker="o", label=res["library"])
        axes[1].plot([p["batch_recall"] for p in c], [p["batch_qps"] for p in c], marker="o", label=res["library"])
    threads = f"{meta['threads']} thread" + ("s" if meta["threads"] != 1 else "")
    for ax, title in zip(axes, ["one query per call, one thread", f"batch, {threads}"]):
        ax.set_xlabel("recall@10 (points below 0.5 are in the JSON)"), ax.set_ylabel("queries per second"), ax.set_yscale("log")
        ax.set_xlim(0.5, 1.005), ax.grid(True, alpha=0.3)
        ax.set_title(f"{name}: {title}"), ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{name}-recall-qps.png", dpi=130)
    filtered = [r for r in results if r["filtered"]]
    if filtered:
        fig, axes = plt.subplots(1, len(SELECTIVITIES), figsize=(6 * len(SELECTIVITIES), 5))
        for ax, pct in zip(axes, SELECTIVITIES):
            for res in filtered:
                pts = [p for p in res["filtered"] if p["selectivity"] == pct]
                ax.plot([p["recall"] for p in pts], [p["qps"] for p in pts], marker="o", label=res["library"])
            ax.set_title(f"{name}: filter matching {pct}%"), ax.set_xlabel("recall@10"), ax.set_ylabel("queries per second")
            ax.set_xlim(0.5, 1.005)
            ax.set_yscale("log"), ax.grid(True, alpha=0.3), ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{name}-filtered.png", dpi=130)
    plt.close("all")

    def fmt(x, unit=""):
        return "—" if x is None else (f"{x:,.0f}{unit}" if x >= 100 else f"{x:.2f}{unit}")
    lines = [f"# {name}", "", f"{meta['n']:,} vectors of {meta['dim']} dimensions ({meta['metric']}), {meta['queries']:,} queries; "
                              f"M {meta['M']}, ef_construction {meta['ef_construction']}; {meta['cpu']}, {meta['threads']} threads; median of {meta['repeat']} runs; {meta['date']}.", "",
             "Queries per second at a recall@10 of at least 0.90 / 0.95 / 0.99 (one query per call, one thread), and build cost:", "",
             "| Library | q/s @ 0.90 | q/s @ 0.95 | q/s @ 0.99 | batch q/s @ 0.95 | build (s) | memory (MB) | index file (MB) |",
             "|---|---|---|---|---|---|---|---|"]
    for res in results:
        c = res["curve"]
        size = "—" if res["index_bytes"] is None else f"{res['index_bytes'] / 2**20:.0f}"
        lines.append(f"| {res['library']} | {fmt(best_qps(c, 0.90))} | {fmt(best_qps(c, 0.95))} | {fmt(best_qps(c, 0.99))} | "
                     f"{fmt(best_qps(c, 0.95, 'batch_qps'))} | {res['build_s']:.1f} | {res['memory_bytes'] / 2**20:.0f} | {size} |")
    if filtered:
        lines += ["", "Filtered search: queries per second at a recall@10 of at least 0.95 (one thread). The filter keeps "
                      "vectors whose bucket (id % 100) is below the percentage, so matches are spread evenly; hnsw-lite filters on "
                      "metadata stored in the index, hnswlib with a Python callback, FAISS with an ID selector:", "",
                  "| Library | " + " | ".join(f"{p}% matching" for p in SELECTIVITIES) + " |", "|---|" + "---|" * len(SELECTIVITIES)]
        for res in filtered:
            cells = [fmt(best_qps([p for p in res["filtered"] if p["selectivity"] == pct], 0.95)) for pct in SELECTIVITIES]
            lines.append(f"| {res['library']} | " + " | ".join(cells) + " |")
    lines += ["", "— : the library did not reach that recall in the sweep, or does not support it.", "",
              "Versions: " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items()) + f"; Python {meta['python']}.", ""]
    (out / f"{name}.md").write_text("\n".join(lines))


def library_versions() -> dict:
    out = {}
    for name, module in [("hnsw-lite", "hnsw_lite_lib"), ("hnswlib", "hnswlib"), ("faiss", "faiss"), ("usearch", "usearch"), ("numpy", "numpy")]:
        try:
            import importlib.metadata as md
            out[name] = md.version({"hnsw-lite": "hnsw-lite-lib", "faiss": "faiss-cpu"}.get(name, name))
        except Exception:
            out[name] = "not installed"
    return out


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="glove-100-angular", help=f"an ann-benchmarks dataset: {', '.join(DATASETS)}")
    ap.add_argument("--synthetic", help="N,DIM: clustered random data instead (no download)")
    ap.add_argument("--metric", default="cosine", choices=["cosine", "l2"], help="for --synthetic")
    ap.add_argument("--libraries", nargs="+", default=LIBRARIES, choices=LIBRARIES)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--filtered-queries", type=int, default=200)
    ap.add_argument("--no-filters", dest="with_filters", action="store_false", help="skip the filtered-search comparison")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0, help="for building and batches (0: every core)")
    ap.add_argument("--repeat", type=int, default=3, help="timed runs per point; the median is reported")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/hnsw-lite-bench"))
    ap.add_argument("--out", default="bench-results")
    ap.add_argument("--worker", dest="library", help=argparse.SUPPRESS)
    ap.add_argument("--data", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.library:  # inside a worker process
        print("RESULT " + json.dumps(worker(args)), flush=True)
        return
    data_path = load_dataset(args)
    data = np.load(data_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = data_path.stem.rsplit("-q", 1)[0]
    meta = {"dataset": name, "n": int(len(data["train"])), "dim": int(data["train"].shape[1]), "metric": str(data["metric"]),
            "queries": int(len(data["test"])), "M": args.M, "ef_construction": args.ef_construction,
            "threads": args.threads or os.cpu_count(), "cpu": cpu_name(), "python": platform.python_version(),
            "date": time.strftime("%Y-%m-%d"), "repeat": args.repeat,
            "versions": library_versions()}
    print(f"{name}: {meta['n']:,} x {meta['dim']} ({meta['metric']}), {meta['queries']} queries, {meta['threads']} threads, {meta['cpu']}", flush=True)
    results = []
    for library in args.libraries:
        cmd = [sys.executable, __file__, "--worker", library, "--data", str(data_path), "--M", str(args.M),
               "--ef-construction", str(args.ef_construction), "--threads", str(args.threads), "--repeat", str(args.repeat)] + ([] if args.with_filters else ["--no-filters"])
        print(f"{library}: building and searching...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                results.append(json.loads(line[7:]))
            else:
                print(line, flush=True)
        if proc.returncode != 0:
            last = (proc.stderr.strip().splitlines() or ["?"])[-1]
            print(f"{library}: skipped ({last})", flush=True)
    (out / f"{name}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    report(results, meta, out)
    print(f"\nwrote {out / (name + '.md')}, {out / (name + '-recall-qps.png')}, {out / (name + '.json')}", flush=True)
    print((out / f"{name}.md").read_text())


if __name__ == "__main__":
    main()