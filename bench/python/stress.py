#!/usr/bin/env python3
"""Stress and load test for hnsw-lite at scale (a million vectors by default).

    pip install hnsw-lite-lib numpy psutil
    python bench/python/stress.py                                  # 1,000,000 x 128, every phase
    python bench/python/stress.py --vectors 100000 --duration 20   # a quick run
    python bench/python/stress.py --phases build,recall,load       # some phases only
    python bench/python/stress.py --vectors auto --max-vectors 50000000 --max-hours 4   # as many as fit

Phases (each a test with checks, not only a timing):

  build    inserts --vectors clustered vectors in batches on every core: throughput
           per batch, memory per vector.
  recall   exact ground truth by brute force for --queries queries; recall@10 and
           single-thread queries per second at several beam widths (ef).
  load     concurrent search clients (1, 2, 4, ... --clients) for --duration
           seconds each: queries per second and latency percentiles.
  mixed    searchers and writers at once for --duration seconds (inserts, updates,
           removes), then: the exact final count, sampled ids present or absent as
           expected, sampled vectors identical to their last write, graph health,
           and recall against a fresh brute force.
  durable  a child process writes to a durable collection and is killed mid-write;
           every write it reported as done must survive recovery, vector included.

Writes stress-results/stress.json and stress-results/stress.md; exits 1 if any
check fails. Memory: about (4 x dim + 200) bytes per vector for the index, plus the
test's own copy of the data (4 x dim bytes per vector).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:  # memory figures are then left out
    psutil = None

import hnsw_lite_lib as hl

K = 10


# ---------------------------------------------------------------- data

# Every vector is a cluster center plus noise. The centers come from `seed` alone,
# so the data, the queries, inserts and updates all share one distribution (as real
# embeddings and their queries do); `stream` selects independent noise for each use.
STREAM_DATA, STREAM_QUERIES, STREAM_INSERTS, STREAM_UPDATES, STREAM_CRASH = range(5)


def centers_for(dim: int, clusters: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, (clusters, dim)).astype(np.float32)


def vectors_for(ids, dim: int, clusters: int, seed: int, stream: int, version: int = 0) -> np.ndarray:
    """Deterministic: the same (id, stream, version) gives the same vector in any
    process, so a crashed writer's data can be checked without storing it."""
    ids = np.asarray(ids, dtype=np.int64)
    centers = centers_for(dim, clusters, seed)
    out = np.empty((len(ids), dim), dtype=np.float32)
    for i, id in enumerate(ids):
        r = np.random.default_rng((seed, stream, version, int(id)))
        out[i] = centers[int(id) % clusters] + r.normal(0, 0.3, dim).astype(np.float32)
    return out


def vectors_block(start: int, count: int, dim: int, clusters: int, seed: int, stream: int) -> np.ndarray:
    """Vectors for ids start .. start + count - 1, generated fast for bulk loading."""
    ids = np.arange(start, start + count)
    noise = np.random.default_rng((seed, stream, start)).normal(0, 0.3, (count, dim)).astype(np.float32)
    return centers_for(dim, clusters, seed)[ids % clusters] + noise


def exact_top_k(data: np.ndarray, live: np.ndarray, queries: np.ndarray, k: int, extra=None) -> np.ndarray:
    """Exact nearest ids (L2), by brute force in blocks: among the rows of `data`
    where `live` is true (row i is id i), and the (ids, vectors) of `extra`
    (vectors written later). No copy of `data` is made, whatever its size."""
    best_d = np.full((len(queries), k), np.inf, dtype=np.float32)
    best_i = np.full((len(queries), k), -1, dtype=np.int64)
    qn = (queries ** 2).sum(1)[:, None]

    def merge(block, ids, mask):
        nonlocal best_d, best_i
        d = qn - 2 * queries @ block.T + (block ** 2).sum(1)[None, :]
        if mask is not None:
            d[:, ~mask] = np.inf
        cand_d = np.concatenate([best_d, d], axis=1)
        cand_i = np.concatenate([best_i, np.broadcast_to(ids, d.shape)], axis=1)
        top = np.argpartition(cand_d, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(cand_d, top, 1)
        best_i = np.take_along_axis(cand_i, top, 1)

    for s in range(0, len(data), 65536):
        block = data[s:s + 65536]
        merge(block, np.arange(s, s + len(block)), live[s:s + len(block)])
    if extra is not None and len(extra[0]):
        ids, vectors = extra
        for s in range(0, len(ids), 65536):
            merge(vectors[s:s + 65536], ids[s:s + 65536], None)
    return best_i


def recall(found, truth) -> float:
    return float(np.mean([len(set(map(int, f)) & set(map(int, t))) / len(t) for f, t in zip(found, truth)]))


# Memory per vector, measured at a million 128-dimensional vectors with M 16: 801
# bytes for the index (vector, links, ids); the test also keeps the data (4 x dim).
def bytes_per_vector(dim: int, M: int) -> float:
    return 4 * dim + 18 * M + 4 * dim + 64  # index + the test's copy + bookkeeping


def auto_size(args) -> int:
    """The largest run that fits in 70% of the available memory, at most
    --max-vectors, rounded down to 100,000."""
    if psutil is None:
        raise SystemExit("--vectors auto needs psutil: pip install psutil")
    available = psutil.virtual_memory().available
    n = int(available * 0.7 / bytes_per_vector(args.dim, args.M))
    n = min(n, args.max_vectors) // 100_000 * 100_000
    print(f"auto: {available / 2**30:.1f} GB available, {bytes_per_vector(args.dim, args.M):.0f} bytes per vector: "
          f"{n:,} vectors (at most {args.max_vectors:,})", flush=True)
    if n < 100_000:
        raise SystemExit("auto: not enough memory for 100,000 vectors")
    return n


def rss_mb() -> float | None:
    return psutil.Process().memory_info().rss / 2**20 if psutil else None


# ---------------------------------------------------------------- results

class Report:
    def __init__(self, args):
        self.args = args
        self.data = {"settings": vars(args), "machine": {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(),
                                                         "python": platform.python_version(), "os": platform.platform()},
                     "phases": {}, "checks": []}

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.data["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"    {'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}", flush=True)
        return ok

    def failed(self) -> int:
        return sum(not c["ok"] for c in self.data["checks"])

    def write(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        (out / "stress.json").write_text(json.dumps(self.data, indent=1, default=str))
        a, lines = self.args, []
        lines += [f"# hnsw-lite stress test", "", f"{a.vectors:,} vectors of {a.dim} dimensions ({a.clusters} clusters), "
                                                  f"M {a.M}, ef_construction {a.ef_construction}; {self.data['machine']['cores']} cores; Python {self.data['machine']['python']}.", ""]
        p = self.data["phases"]
        if "build" in p:
            b = p["build"]
            lines += ["## Build", "", f"{b['vectors']:,} vectors in {b['seconds']:.0f} s: **{b['per_second']:,.0f} vectors/s**"
                      + (f"; index memory {b['memory_mb']:,.0f} MB ({b['bytes_per_vector']:,.0f} bytes per vector)" if b.get("memory_mb") else "") + ".", ""]
        if "recall" in p:
            lines += ["## Recall", "", "| ef | recall@10 | queries/s (1 thread) |", "|---|---|---|"]
            lines += [f"| {r['ef']} | {r['recall']:.3f} | {r['qps']:,.0f} |" for r in p["recall"]] + [""]
        if "load" in p:
            lines += ["## Load", "", f"ef {p['load'][0]['ef']}, {a.duration} s per level.", "",
                      "| clients | queries/s | p50 ms | p95 ms | p99 ms | max ms |", "|---|---|---|---|---|---|"]
            lines += [f"| {r['clients']} | {r['qps']:,.0f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | {r['max_ms']:.2f} |" for r in p["load"]] + [""]
        if "mixed" in p:
            m = p["mixed"]
            lines += ["## Mixed (searches during changes)", "", f"{a.duration} s, {a.clients} searchers, {a.writers} writers: "
                                                                f"{m['searches']:,} searches ({m['search_qps']:,.0f}/s, p99 {m['search_p99_ms']:.2f} ms), "
                                                                f"{m['inserts']:,} inserts, {m['updates']:,} updates, {m['removes']:,} removes ({m['writes_per_second']:,.0f} changes/s). "
                                                                f"Recall@10 after: {m['recall_after']:.3f}; after upkeep: {m['recall_after_upkeep']:.3f}.", ""]
        if "durable" in p:
            d = p["durable"]
            lines += ["## Crash recovery", "", f"Writer killed after {d['acknowledged']:,} acknowledged writes; recovered {d['recovered']:,} "
                                               f"documents in {d['recovery_seconds']:.1f} s.", ""]
        lines += ["## Checks", "", "| Check | Result | Detail |", "|---|---|---|"]
        lines += [f"| {c['check']} | {'pass' if c['ok'] else '**FAIL**'} | {c['detail']} |" for c in self.data["checks"]]
        (out / "stress.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- phases

def phase_build(args, rep):
    print(f"build: {args.vectors:,} vectors of {args.dim} dimensions", flush=True)
    p = hl.HnswParams()
    p.M, p.ef_construction = args.M, args.ef_construction
    index = hl.HnswIndex(args.dim, hl.Metric.L2, p)
    data = np.empty((args.vectors, args.dim), dtype=np.float32)
    rss0 = rss_mb()
    t0 = time.perf_counter()
    batches = []
    budget = args.max_hours * 3600 * 0.5 if args.max_hours else None  # half the time for building, half for the rest
    for start in range(0, args.vectors, args.batch):
        if budget and time.perf_counter() - t0 > budget:
            print(f"  time budget reached: stopping at {start:,} vectors", flush=True)
            data = data[:start]
            args.vectors = start
            break
        n = min(args.batch, args.vectors - start)
        block = vectors_block(start, n, args.dim, args.clusters, args.seed, STREAM_DATA)
        data[start:start + n] = block
        tb = time.perf_counter()
        index.add(np.arange(start, start + n), block, threads=args.threads)
        dt = time.perf_counter() - tb
        batches.append({"start": start, "vectors": n, "seconds": dt, "per_second": n / dt})
        done = start + n
        print(f"  {done:>10,} vectors, {n / dt:>8,.0f}/s this batch, {time.perf_counter() - t0:7.0f} s", flush=True)
    seconds = time.perf_counter() - t0
    data_mb = data.nbytes / 2**20
    mem = (rss_mb() - rss0 - data_mb) if psutil else None
    rep.data["phases"]["build"] = {"vectors": args.vectors, "seconds": seconds, "per_second": args.vectors / seconds, "batches": batches,
                                   "memory_mb": mem, "bytes_per_vector": mem * 2**20 / args.vectors if mem else None}
    rep.check("every vector stored", len(index) == args.vectors, f"{len(index):,} of {args.vectors:,}")
    slowest = min(b["per_second"] for b in batches)
    fastest = max(b["per_second"] for b in batches)
    rep.check("insert speed holds up as the index grows", slowest > fastest / 20, f"slowest batch {slowest:,.0f}/s, fastest {fastest:,.0f}/s")
    return index, data


def phase_recall(args, rep, index, data, live, queries):
    print(f"recall: ground truth for {len(queries)} queries", flush=True)
    truth = exact_top_k(data, live, queries, K)
    rows = []
    for ef in (32, 64, 128, 256):
        t0 = time.perf_counter()
        found = [index.search(q, k=K, ef=ef)[0] for q in queries]
        qps = len(queries) / (time.perf_counter() - t0)
        rows.append({"ef": ef, "recall": recall(found, truth), "qps": qps})
        print(f"  ef {ef:4d}: recall@10 {rows[-1]['recall']:.3f}, {qps:,.0f} queries/s", flush=True)
    rep.data["phases"]["recall"] = rows
    rep.check("recall@10 at ef 128 is at least 0.9", rows[2]["recall"] >= 0.9, f"{rows[2]['recall']:.3f}")
    rep.check("recall grows with ef", rows[-1]["recall"] >= rows[0]["recall"], " -> ".join(f"{r['recall']:.3f}" for r in rows))
    return truth


def run_clients(index, queries, clients, seconds, ef, stop_flag=None):
    """Search threads for `seconds`; returns (count, latencies in ms, errors)."""
    latencies, errors = [[] for _ in range(clients)], []
    deadline = time.perf_counter() + seconds

    def client(c):
        rng = np.random.default_rng(c)
        try:
            while time.perf_counter() < deadline and not (stop_flag and stop_flag.is_set()):
                q = queries[rng.integers(len(queries))]
                t0 = time.perf_counter()
                index.search(q, k=K, ef=ef)
                latencies[c].append((time.perf_counter() - t0) * 1000)
        except Exception as e:  # recorded: a failing search fails the run
            errors.append(repr(e))

    threads = [threading.Thread(target=client, args=(c,)) for c in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    all_lat = np.concatenate([np.array(l) for l in latencies]) if any(latencies) else np.array([0.0])
    return len(all_lat), all_lat, errors


def phase_load(args, rep, index, queries):
    print(f"load: up to {args.clients} concurrent clients, {args.duration} s each", flush=True)
    rows, levels = [], sorted({1, *[2 ** i for i in range(1, 8) if 2 ** i < args.clients], args.clients})
    for clients in levels:
        n, lat, errors = run_clients(index, queries, clients, args.duration, args.ef)
        rows.append({"clients": clients, "ef": args.ef, "queries": n, "qps": n / args.duration, "p50_ms": float(np.percentile(lat, 50)),
                     "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99)), "max_ms": float(lat.max()),
                     "errors": errors})
        r = rows[-1]
        print(f"  {clients:3d} clients: {r['qps']:9,.0f} queries/s, p50 {r['p50_ms']:.2f} ms, p99 {r['p99_ms']:.2f} ms", flush=True)
        rep.check(f"no errors with {clients} clients", not errors, "; ".join(errors[:2]))
    rep.data["phases"]["load"] = rows
    if len(rows) > 1 and (os.cpu_count() or 1) > 1:
        rep.check("throughput grows with clients (searches run in parallel)", rows[-1]["qps"] > rows[0]["qps"] * 1.3,
                  f"{rows[0]['qps']:,.0f}/s with 1, {rows[-1]['qps']:,.0f}/s with {rows[-1]['clients']}")


def phase_mixed(args, rep, index, data, queries):
    print(f"mixed: {args.clients} searchers and {args.writers} writers for {args.duration} s", flush=True)
    n0 = len(data)
    capacity = n0 + 2_000_000
    live = np.zeros(capacity, dtype=bool)
    live[:n0] = True
    last = {}  # id -> the vector last written (inserts and updates)
    removed = set()  # ids removed during the phase
    counts = {"inserts": 0, "updates": 0, "removes": 0}
    errors, lock = [], threading.Lock()
    stop = threading.Event()
    deadline = time.perf_counter() + args.duration

    def writer(w):
        rng = np.random.default_rng(1000 + w)
        next_id = n0 + w  # new ids: n0 + w, n0 + w + writers, ...: each writer owns its ids
        mine = [i for i in range(w, n0, args.writers)]  # existing ids this writer may change
        try:
            while time.perf_counter() < deadline:
                op = rng.random()
                if op < 0.5 or not mine:
                    v = vectors_for([next_id], args.dim, args.clusters, args.seed, STREAM_INSERTS)[0]
                    index.add(next_id, v)
                    with lock:
                        live[next_id] = True
                        last[next_id] = v
                        counts["inserts"] += 1
                    mine.append(next_id)
                    next_id += args.writers
                elif op < 0.8:
                    id = mine[rng.integers(len(mine))]
                    v = vectors_for([id], args.dim, args.clusters, args.seed, STREAM_UPDATES, int(rng.integers(10**6)))[0]
                    if index.upsert(id, v):
                        raise AssertionError(f"upsert of stored id {id} reported an insert")
                    with lock:
                        last[id] = v
                        counts["updates"] += 1
                else:
                    id = mine.pop(int(rng.integers(len(mine))))
                    if index.remove(id) != 1:
                        raise AssertionError(f"removing stored id {id} removed nothing")
                    with lock:
                        live[id] = False
                        last.pop(id, None)
                        removed.add(id)
                        counts["removes"] += 1
        except Exception as e:
            errors.append(f"writer {w}: {e!r}")
            stop.set()

    writers = [threading.Thread(target=writer, args=(w,)) for w in range(args.writers)]
    t0 = time.perf_counter()
    for t in writers:
        t.start()
    n, lat, search_errors = run_clients(index, queries, args.clients, args.duration, args.ef, stop)
    for t in writers:
        t.join()
    seconds = time.perf_counter() - t0
    errors += search_errors
    changes = sum(counts.values())
    rep.check("no errors during searches and changes", not errors, "; ".join(errors[:3]))
    expected = int(live.sum())
    rep.check("the final count is exact", len(index) == expected, f"{len(index):,} stored, {expected:,} expected")
    rng = np.random.default_rng(99)
    touched = np.array(sorted(set(last) | removed), dtype=np.int64)  # written or removed during the phase
    sample = np.concatenate([rng.integers(0, n0, 3000), rng.choice(touched, min(len(touched), 2000), replace=False) if len(touched) else []]).astype(np.int64)
    wrong_presence = [int(i) for i in sample if (int(i) in index) != bool(live[i])]
    rep.check("sampled ids are present or absent as expected", not wrong_presence, f"{len(sample):,} sampled, wrong: {wrong_presence[:5]}")
    wrong_vectors = []
    for i in sample:
        i = int(i)
        if not live[i]:
            continue
        want = last[i] if i in last else data[i]
        got = index.get_vector(i)
        if got is None or not np.array_equal(got, want):
            wrong_vectors.append(i)
    rep.check("sampled vectors equal their last write", not wrong_vectors, f"wrong: {wrong_vectors[:5]}")
    # Recall against a fresh brute force over the live set: the base data where
    # unchanged, plus every vector written during the phase (no second copy).
    unchanged = live[:n0].copy()
    written = np.array(sorted(last), dtype=np.int64)
    unchanged[written[written < n0]] = False
    extra_vectors = np.stack([last[int(i)] for i in written]) if len(written) else np.zeros((0, args.dim), np.float32)
    truth = exact_top_k(data, unchanged, queries, K, (written, extra_vectors))
    after = recall([index.search(q, k=K, ef=128)[0] for q in queries], truth)
    health = index.health(recall_sample=0)
    t1 = time.perf_counter()
    upkeep = index.upkeep()
    upkeep_s = time.perf_counter() - t1
    health2 = index.health(recall_sample=0)
    after_upkeep = recall([index.search(q, k=K, ef=128)[0] for q in queries], truth)
    print(f"  {n:,} searches, {changes:,} changes; recall@10 {after:.3f}, after upkeep {after_upkeep:.3f}; "
          f"unreachable {health['unreachable']} -> {health2['unreachable']}", flush=True)
    rep.check("recall@10 stays at least 0.85 under churn (ef 128)", after >= 0.85, f"{after:.3f}")
    rep.check("upkeep leaves no vector unreachable", health2["unreachable"] == 0 and upkeep["complete"],
              f"{health['unreachable']} before, {health2['unreachable']} after ({upkeep_s:.1f} s)")
    rep.data["phases"]["mixed"] = {**counts, "searches": n, "search_qps": n / seconds, "search_p99_ms": float(np.percentile(lat, 99)),
                                   "writes_per_second": changes / seconds, "recall_after": after, "recall_after_upkeep": after_upkeep,
                                   "unreachable_before_upkeep": health["unreachable"], "upkeep_seconds": upkeep_s}


def durable_child(args):
    """A writer killed by its parent: upserts documents and prints each id after the
    call returns (an acknowledged write)."""
    c = hl.Collection(dim=args.dim, metric="l2", text=False, path=args.child_dir, maintenance="manual", durability="grouped")
    i = 0
    while True:
        c.upsert(i, vector=vectors_for([i], args.dim, args.clusters, args.seed, STREAM_CRASH)[0], metadata={"n": i})
        print(i, flush=True)
        i += 1


def phase_durable(args, rep):
    print(f"durable: a writer killed after {args.crash_after} s, then recovery", flush=True)
    work = Path(tempfile.mkdtemp(prefix="hnsw_lite_stress_"))
    directory = work / "collection.db"
    cmd = [sys.executable, __file__, "--child-dir", str(directory), "--dim", str(args.dim), "--clusters", str(args.clusters), "--seed", str(args.seed)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    acknowledged = []
    killer = threading.Timer(args.crash_after, proc.kill)  # SIGKILL on Unix, TerminateProcess on Windows
    killer.start()
    for line in proc.stdout:
        if line.strip().isdigit():
            acknowledged.append(int(line))
    proc.wait()
    killer.cancel()
    t0 = time.perf_counter()
    c = hl.Collection(dim=args.dim, metric="l2", text=False, path=str(directory), maintenance="manual")
    recovery = time.perf_counter() - t0
    missing = [i for i in acknowledged if i not in c]
    sample = acknowledged[:: max(1, len(acknowledged) // 500)]
    wrong = [i for i in sample if not np.array_equal(c.get(i)["vector"], vectors_for([i], args.dim, args.clusters, args.seed, STREAM_CRASH)[0])]
    recovered = len(c)
    c.close()
    shutil.rmtree(work, ignore_errors=True)
    print(f"  {len(acknowledged):,} acknowledged writes, {recovered:,} recovered in {recovery:.1f} s", flush=True)
    rep.check("the writer was killed mid-write", proc.returncode != 0 and len(acknowledged) > 0, f"exit code {proc.returncode}, {len(acknowledged):,} writes")
    rep.check("every acknowledged write survived the crash", not missing, f"missing: {missing[:5]}")
    rep.check("recovered vectors are the ones written", not wrong, f"{len(sample)} checked, wrong: {wrong[:5]}")
    rep.data["phases"]["durable"] = {"acknowledged": len(acknowledged), "recovered": recovered, "recovery_seconds": recovery}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vectors", default="1000000", help="a number, or auto: as many as fit in memory (see --max-vectors)")
    ap.add_argument("--max-vectors", type=int, default=50_000_000, help="the most --vectors auto may choose")
    ap.add_argument("--max-hours", type=float, default=0, help="time budget: building stops at half of it (0: none)")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--clusters", type=int, default=1000, help="the data's natural clusters (real embeddings are clustered)")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef", type=int, default=128, help="beam width of the load and mixed phases")
    ap.add_argument("--batch", type=int, default=20_000, help="vectors per insert batch")
    ap.add_argument("--threads", type=int, default=0, help="for inserts (0: every core)")
    ap.add_argument("--queries", type=int, default=200, help="queries with exact ground truth")
    ap.add_argument("--duration", type=float, default=30, help="seconds per load level and for the mixed phase")
    ap.add_argument("--clients", type=int, default=max(2, os.cpu_count() or 2), help="concurrent search clients")
    ap.add_argument("--writers", type=int, default=2, help="concurrent writers in the mixed phase")
    ap.add_argument("--crash-after", type=float, default=10, help="seconds before the durable phase's writer is killed")
    ap.add_argument("--phases", default="build,recall,load,mixed,durable")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="stress-results")
    ap.add_argument("--child-dir", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child_dir:
        durable_child(args)
        return
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = set(phases) - {"build", "recall", "load", "mixed", "durable"}
    if unknown:
        ap.error(f"unknown phases: {sorted(unknown)}")
    if any(p in phases for p in ("recall", "load", "mixed")) and "build" not in phases:
        ap.error("recall, load and mixed need the build phase")
    args.vectors = auto_size(args) if str(args.vectors) == "auto" else int(args.vectors)
    rep = Report(args)
    started = time.perf_counter()
    index = data = None
    queries = vectors_block(0, args.queries, args.dim, args.clusters, args.seed, STREAM_QUERIES)  # same clusters, independent noise
    if "build" in phases:
        index, data = phase_build(args, rep)
    if "recall" in phases:
        phase_recall(args, rep, index, data, np.ones(len(data), dtype=bool), queries)
    if "load" in phases:
        phase_load(args, rep, index, queries)
    if "mixed" in phases:
        phase_mixed(args, rep, index, data, queries)
    if "durable" in phases:
        phase_durable(args, rep)
    rep.data["seconds"] = time.perf_counter() - started
    out = Path(args.out)
    rep.write(out)
    failed = rep.failed()
    print(f"\n{len(rep.data['checks']) - failed} of {len(rep.data['checks'])} checks passed in {rep.data['seconds']:.0f} s; "
          f"see {out / 'stress.md'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()