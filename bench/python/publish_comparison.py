#!/usr/bin/env python3
"""Publishes real-dataset comparisons (bench/python/compare.py) to the benchmarks branch.

    python bench/python/publish_comparison.py --results real-results --site site --commit SHA \\
        [--commit-url URL] [--run-url URL] [--repo-url URL]

--results is a folder written by compare.py --out (one or more datasets: NAME.json,
NAME.md and NAME-*.png each). Under SITE/real-datasets/ this writes:

  NAME/                   the latest run of each dataset: summary, charts, raw numbers
  history/DATE-SHA/NAME/  every run, kept
  README.md               the page GitHub shows: every dataset's latest summary and charts

Datasets not in this run keep their latest published results, so one dataset can be
rerun on its own. The branch's front page gets a link here if it lacks one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
from pathlib import Path

LINK = "Comparisons on real datasets (SIFT1M, GloVe) against hnswlib, FAISS and USearch: [real-datasets](real-datasets/README.md)."


def publish(results: Path, site: Path, meta: dict) -> list[str]:
    root = site / "real-datasets"
    stamp = f"{meta['published_at'][:10]}-{meta['commit'][:7]}"
    names = sorted(p.stem for p in results.glob("*.json"))
    if not names:
        raise SystemExit(f"no results in {results}")
    for name in names:
        files = [p for p in results.iterdir() if p.name == f"{name}.json" or p.name == f"{name}.md" or p.name.startswith(f"{name}-")]
        latest, kept = root / name, root / "history" / stamp / name
        if latest.exists():
            shutil.rmtree(latest)
        for folder in (latest, kept):
            folder.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, folder / f.name)
        (latest / "published.json").write_text(json.dumps(meta, indent=2) + "\n")
    return names


def index_page(root: Path, repo_url: str | None) -> str:
    method = f"[methodology]({repo_url}/blob/master/docs/benchmarks.md)" if repo_url else "methodology: `docs/benchmarks.md`"
    lines = ["# Real-dataset comparisons", "",
             "hnsw-lite against hnswlib, FAISS (HNSW and IVF-PQ) and USearch on standard ann-benchmarks datasets, "
             "whose files include the exact neighbors of every query. Published by `.github/workflows/real-benchmarks.yml` "
             f"from `bench/python/compare.py` ({method}). Graph libraries share M and ef_construction; each library's "
             "search beam is swept, so curves compare at equal recall, never at equal settings. Every run is kept in "
             "`history/`.", "", "[Nightly benchmark](../README.md)", ""]
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "history"):
        name = folder.name
        meta = json.loads((folder / "published.json").read_text()) if (folder / "published.json").exists() else {}
        summary = (folder / f"{name}.md").read_text() if (folder / f"{name}.md").exists() else ""
        summary = re.sub(r"^# .*\n+", "", summary)  # the dataset heading comes from here instead
        commit = meta.get("commit", "")[:7]
        when = meta.get("published_at", "")[:16].replace("T", " ")
        source = f"[`{commit}`]({meta['commit_url']})" if meta.get("commit_url") else f"`{commit}`"
        run = f" ([run]({meta['run_url']}))" if meta.get("run_url") else ""
        lines += [f"## {name}", "", f"Published {when} UTC from commit {source}{run}.", "", summary.rstrip(), ""]
        for chart in sorted(folder.glob(f"{name}-*.png")):
            lines += [f"![{chart.stem}]({name}/{chart.name})", ""]
    return "\n".join(lines).rstrip() + "\n"


def link_front_page(site: Path) -> None:
    page = site / "README.md"
    if not page.exists():
        page.write_text("# hnsw-lite benchmarks\n\n" + LINK + "\n")
        return
    text = page.read_text()
    if "real-datasets/README.md" in text:
        return
    first, sep, rest = text.partition("\n\n")  # after the title
    second, sep2, rest2 = rest.partition("\n\n")  # after the introduction
    page.write_text(first + sep + second + "\n\n" + LINK + (sep2 + rest2 if sep2 else "\n"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--site", required=True, type=Path)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--commit-url")
    ap.add_argument("--run-url")
    ap.add_argument("--repo-url")
    args = ap.parse_args()
    meta = {"commit": args.commit, "commit_url": args.commit_url, "run_url": args.run_url,
            "published_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    names = publish(args.results, args.site, meta)
    root = args.site / "real-datasets"
    (root / "README.md").write_text(index_page(root, args.repo_url))
    link_front_page(args.site)
    print(f"published {', '.join(names)} to {root}")


if __name__ == "__main__":
    main()