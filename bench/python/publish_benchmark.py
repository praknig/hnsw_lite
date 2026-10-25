#!/usr/bin/env python3
"""Publishes stress.py results: archives the run and draws charts, for the
`benchmarks` branch the nightly workflow (.github/workflows/benchmark.yml) commits.

    python bench/python/publish_benchmark.py --results stress-results --site site \\
        --commit "$GITHUB_SHA" --run-url "$RUN_URL"

In --site it writes:
  runs/<UTC time>-<commit>.json   every run's full results (the history)
  latest.json                     the newest run
  latest-recall.png               recall@10 and queries per second against ef
  latest-load.png                 queries per second and latency against clients
  latest-build.png                insert speed as the index grows
  history.png                     trends across runs: insert speed, queries per
                                  second, recall and p99 latency
  README.md                       the page GitHub shows for the branch
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HISTORY_ROWS = 30


def load_runs(site: Path) -> list[dict]:
    runs = []
    for f in sorted((site / "runs").glob("*.json")):
        try:
            r = json.loads(f.read_text())
            r["_file"] = f.name
            runs.append(r)
        except (json.JSONDecodeError, OSError):  # a damaged file is skipped, never fatal
            print(f"skipping unreadable {f}")
    return runs


def figure(title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def latest_charts(run: dict, site: Path) -> list[str]:
    made = []
    p = run["phases"]
    size = f"{run['settings']['vectors']:,} x {run['settings']['dim']}"
    if "recall" in p:
        rows = p["recall"]
        fig, ax = figure(f"Recall and speed against ef ({size})")
        ax.plot([r["ef"] for r in rows], [r["recall"] for r in rows], "o-", color="tab:blue", label="recall@10")
        ax.set_xlabel("ef (search beam)")
        ax.set_ylabel("recall@10", color="tab:blue")
        ax.set_xscale("log", base=2)
        ax2 = ax.twinx()
        ax2.plot([r["ef"] for r in rows], [r["qps"] for r in rows], "s--", color="tab:orange", label="queries/s")
        ax2.set_ylabel("queries per second (1 thread)", color="tab:orange")
        save(fig, site / "latest-recall.png")
        made.append("latest-recall.png")
    if "load" in p:
        rows = p["load"]
        fig, ax = figure(f"Concurrent load ({size}, ef {rows[0]['ef']})")
        ax.plot([r["clients"] for r in rows], [r["qps"] for r in rows], "o-", color="tab:green")
        ax.set_xlabel("concurrent clients")
        ax.set_ylabel("queries per second", color="tab:green")
        ax.set_ylim(bottom=0)
        ax2 = ax.twinx()
        for key, style in (("p50_ms", ":"), ("p99_ms", "--")):
            ax2.plot([r["clients"] for r in rows], [r[key] for r in rows], style, color="tab:red", marker="x", label=key.replace("_ms", ""))
        ax2.set_ylabel("latency (ms)", color="tab:red")
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper left")
        save(fig, site / "latest-load.png")
        made.append("latest-load.png")
    if "build" in p and p["build"].get("batches"):
        b = p["build"]["batches"]
        fig, ax = figure(f"Insert speed as the index grows ({size})")
        ax.plot([x["start"] + x["vectors"] for x in b], [x["per_second"] for x in b], color="tab:purple")
        ax.set_xlabel("vectors in the index")
        ax.set_ylabel("inserts per second")
        ax.set_ylim(bottom=0)
        save(fig, site / "latest-build.png")
        made.append("latest-build.png")
    return made


def metric(run: dict, name: str):
    p = run.get("phases", {})
    try:
        if name == "build":
            return p["build"]["per_second"]
        if name == "qps":
            return next(r["qps"] for r in p["recall"] if r["ef"] == 128)
        if name == "recall":
            return next(r["recall"] for r in p["recall"] if r["ef"] == 128)
        if name == "p99":
            return p["load"][-1]["p99_ms"]
    except (KeyError, StopIteration, IndexError, TypeError):
        return None
    return None


def short_size(n: int) -> str:
    return f"{n / 1e6:.3g}M" if n >= 1_000_000 else f"{n / 1e3:.3g}k" if n >= 1000 else str(n)


def history_chart(runs: list[dict], site: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    when = [dt.datetime.fromisoformat(r["published_at"]) for r in runs]
    panels = [("build", "inserts per second"), ("qps", "queries/s at ef 128 (1 thread)"),
              ("recall", "recall@10 at ef 128"), ("p99", "p99 latency, most clients (ms)")]
    for ax, (name, label) in zip(axes.flat, panels):
        pts = [(w, metric(r, name), r["settings"]["vectors"]) for w, r in zip(when, runs) if metric(r, name) is not None]
        if pts:
            ax.plot([x[0] for x in pts], [x[1] for x in pts], "o-")
            sizes = {x[2] for x in pts}
            if len(sizes) > 1:  # runs of different sizes are not directly comparable: say so on the chart
                for w, v, n in pts:
                    ax.annotate(short_size(n), (w, v), fontsize=7, textcoords="offset points", xytext=(0, 5), ha="center")
            # Honest scales: rates and latencies from zero (a 1% wobble must look like one);
            # recall over a fixed range near its top.
            if name == "recall":
                ax.set_ylim(max(0.0, min(x[1] for x in pts) - 0.05), 1.005)
            else:
                ax.set_ylim(0, max(x[1] for x in pts) * 1.15)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.suptitle("hnsw-lite nightly benchmark: trends")
    save(fig, site / "history.png")


def page(runs: list[dict], charts: list[str]) -> str:
    r = runs[-1]
    a, m = r["settings"], r["machine"]
    checks = r["checks"]
    passed = sum(c["ok"] for c in checks)
    lines = ["# hnsw-lite benchmarks", "",
             "Published by the nightly workflow (`.github/workflows/benchmark.yml`) from `bench/python/stress.py`. "
             "Every run is kept in `runs/`; this page shows the newest and the trends.", "",
             f"## Latest: {r['published_at'][:16].replace('T', ' ')} UTC, commit "
             + (f"[`{r['commit'][:7]}`]({r['commit_url']})" if r.get("commit_url") else f"`{r['commit'][:7]}`"), "",
             f"**{a['vectors']:,} vectors of {a['dim']} dimensions**, M {a['M']}, ef_construction {a['ef_construction']}; "
             f"{m['cores']} cores; Python {m['python']}. **{passed} of {len(checks)} checks passed**"
             + (f" ([run]({r['run_url']}))" if r.get("run_url") else "") + ".", ""]
    for c in charts:
        lines += [f"![{c}]({c})", ""]
    lines += ["## Trends", "", "![history](history.png)", "",
              "Runs on different machines or of different sizes are not directly comparable (sizes are marked on the chart when they vary).", "",
              "| Date (UTC) | Commit | Vectors | Inserts/s | Recall@10 (ef 128) | Queries/s (ef 128) | p99 ms | Checks |", "|---|---|---|---|---|---|---|---|"]
    for run in reversed(runs[-HISTORY_ROWS:]):
        def fmt(v, f):
            return "—" if v is None else f.format(v)
        ok = sum(c["ok"] for c in run["checks"])
        lines.append(f"| {run['published_at'][:16].replace('T', ' ')} | `{run['commit'][:7]}` | {run['settings']['vectors']:,} | "
                     f"{fmt(metric(run, 'build'), '{:,.0f}')} | {fmt(metric(run, 'recall'), '{:.3f}')} | {fmt(metric(run, 'qps'), '{:,.0f}')} | "
                     f"{fmt(metric(run, 'p99'), '{:.2f}')} | {ok}/{len(run['checks'])}{'' if ok == len(run['checks']) else ' **FAIL**'} |")
    failed = [c for c in checks if not c["ok"]]
    if failed:
        lines += ["", "## Failed checks in the latest run", ""] + [f"- {c['check']}: {c['detail']}" for c in failed]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="stress-results", help="stress.py's output directory")
    ap.add_argument("--site", required=True, help="the benchmarks branch's checkout")
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--commit-url", default="")
    ap.add_argument("--run-url", default="")
    args = ap.parse_args()
    run = json.loads((Path(args.results) / "stress.json").read_text())
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    run.update({"published_at": now.isoformat(), "commit": args.commit, "commit_url": args.commit_url, "run_url": args.run_url})
    site = Path(args.site)
    (site / "runs").mkdir(parents=True, exist_ok=True)
    name = f"{now.strftime('%Y-%m-%dT%H%M%SZ')}-{args.commit[:7]}.json"
    (site / "runs" / name).write_text(json.dumps(run, indent=1))
    shutil.copy(site / "runs" / name, site / "latest.json")
    runs = load_runs(site)
    runs.sort(key=lambda r: r["published_at"])
    charts = latest_charts(run, site)
    history_chart(runs, site)
    (site / "README.md").write_text(page(runs, charts))
    print(f"published {name}: {len(runs)} runs in the history; charts: {', '.join(charts + ['history.png'])}")


if __name__ == "__main__":
    main()