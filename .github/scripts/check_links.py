#!/usr/bin/env python3
"""Checks every relative link in the repository's Markdown files: the target file
must exist, and a #anchor must match a heading in the target Markdown file
(GitHub's rule: lower case, punctuation removed, spaces become hyphens).
Links to websites are not checked. Exits with 1 if anything is broken.

Usage (from the repository root): python3 .github/scripts/check_links.py
"""
import pathlib
import re
import subprocess
import sys


def markdown_files():
    out = subprocess.run(["git", "ls-files", "*.md"], capture_output=True, text=True, check=True).stdout
    return [pathlib.Path(line) for line in out.splitlines() if line]


def without_code(text):
    return re.sub(r"(?s)```.*?```", "", text)


def slug(heading):
    h = heading.strip().lower().replace("`", "")
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def anchors(path, cache={}):
    if path not in cache:
        text = without_code(path.read_text(encoding="utf-8"))
        cache[path] = {slug(m.group(1)) for m in re.finditer(r"(?m)^#{1,6} (.+)$", text)}
    return cache[path]


def main():
    broken, checked = [], 0
    for md in markdown_files():
        text = without_code(md.read_text(encoding="utf-8"))
        for match in re.finditer(r"\]\(([^)\s]+)\)", text):
            link = match.group(1)
            if link.startswith(("http://", "https://", "mailto:", "ftp://")):
                continue
            checked += 1
            path, _, anchor = link.partition("#")
            target = (md.parent / path).resolve() if path else md.resolve()
            if not target.exists():
                broken.append(f"{md}: {link} (no such file)")
            elif anchor and target.suffix == ".md" and anchor not in anchors(target):
                broken.append(f"{md}: {link} (no heading for #{anchor})")
    for b in broken:
        print("BROKEN", b)
    print(f"{checked} links checked, {len(broken)} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())