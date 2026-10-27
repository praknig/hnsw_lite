#!/usr/bin/env python3
"""Generates the developer reference's signature appendix from the compiler's view
of the headers (clang's JSON AST), so every type, default and parameter is exact.

Usage: python3 .github/scripts/gen_signatures.py   (from the repository root; needs clang++)
Rewrites the part of docs/developer-reference.md between the GENERATED markers.
"""
import json, pathlib, re, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
HEADERS = ["scoring.h", "collection.h", "hnsw_index.h", "flat_index.h", "disk_index.h", "vamana.h", "sparse.h", "metadata.h", "filter.h",
           "search_result.h", "health.h", "codec.h", "half.h", "facets.h", "inspect.h", "sparse_vector.h", "durable.h", "wal.h", "distance.h"]
# Public API by header, in reference order. Records whose public members are listed in full.
GROUPS = [
    ("Collections (collection.h)", ["Preset", "Maintenance", "MaintenancePolicy", "MaintenanceReport", "MaintenanceStatus", "TtlOptions", "CollectionOptions",
                                    "FacetRequest", "FacetValue", "FacetBucket", "FacetResult", "StoredInfo", "Document", "DocumentRecord", "CollectionScanOptions", "CollectionPage", "CollectionQuery", "CollectionSearchOptions", "CollectionHit", "Collection"]),
    ("Metadata and filters (metadata.h, filter.h)", ["FieldType", "Value", "Metadata", "FieldSpec", "SchemaMode", "Schema", "CompareOp", "Filter",
                                                     "Strategy", "PlanReason", "PlannerParams", "DecayShape", "RecencyDecay", "FieldMultiplier", "MatchBoost",
                                                     "ScoreModifiers", "SearchStats", "SearchOptions", "OperationOptions",
                                                     "Record", "ScanOptions", "Page", "PlanDecision"]),
    ("Results and distances (search_result.h, distance.h)", ["Metric", "SearchResult", "CompactStats"]),
    ("HnswIndex (hnsw_index.h, codec.h)", ["CodecKind", "QuantizationParams", "HnswParams", "HnswIndex"]),
    ("FlatIndex (flat_index.h)", ["IvfParams", "FlatIndex"]),
    ("DiskIndex (disk_index.h, vamana.h)", ["VamanaParams", "DiskVectorFormat", "DiskIndexParams", "LargeBuildParams", "LargeBuildStats", "DiskOpenOptions",
                                            "DiskSearchParams", "DiskSearchStats", "DiskIndex", "BlockFile"]),
    ("Sparse vectors, BM25 and hybrid search (sparse.h)", ["SparseVector", "SparseRecord", "SparseIndex", "Bm25Params", "Bm25", "Fusion",
                                                           "HybridOptions", "HybridResult"]),
    ("Health, upkeep and rebuilds (health.h)", ["HealthOptions", "LevelHealth", "HealthReport", "UpkeepOptions", "UpkeepStats", "RebuildOptions",
                                                "RebuildStats"]),
    ("Durability (durable.h, wal.h)", ["Durability", "DurableOptions", "RecoveryStats", "WalRecord", "WalScan", "WalWriter", "FileLock"]),
]
FUNCTIONS = ["hybrid_search", "fuse", "sparse_dot", "l2_radius", "write_vectors_file", "read_segment", "sync_path", "sync_directory", "plan_search",
             "validate", "parse_date", "format_date", "now_ms", "relevance_from_distance", "rescored", "mmr_order",
             "float_to_f16", "f16_to_float", "float_to_bf16", "bf16_to_float", "identify", "stored_kind_name"]

def dump():
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write("".join(f'#include "{h}"\n' for h in HEADERS))
    out = subprocess.run(["clang++", "-std=c++20", f"-I{ROOT/'include'}", f"-I{ROOT/'src'}", "-Xclang", "-ast-dump=json",
                          "-Xclang", "-ast-dump-filter=vecdb", "-fsyntax-only", f.name], capture_output=True, text=True, check=True).stdout
    dec, i, roots = json.JSONDecoder(), 0, []
    while i < len(out):
        while i < len(out) and out[i].isspace(): i += 1
        if i >= len(out): break
        obj, i = dec.raw_decode(out, i)
        roots.append(obj)
    return roots

class Source:
    """Clang's JSON names a file only when it changes: follow it in document order."""
    def __init__(self):
        self.file, self.cache = None, {}
    def see(self, loc):
        for part in (loc, loc.get("spellingLoc", {}), loc.get("expansionLoc", {})):
            if isinstance(part, dict) and "file" in part: self.file = part["file"]
    def text(self, rng):
        b, e = rng.get("begin", {}), rng.get("end", {})
        b = b.get("spellingLoc", b); e = e.get("spellingLoc", e)
        if "offset" not in b or "offset" not in e or not self.file: return None
        path = pathlib.Path(self.file if pathlib.Path(self.file).is_absolute() else ROOT / self.file)
        data = self.cache.setdefault(path, path.read_bytes() if path.exists() else b"")
        return data[b["offset"]: e["offset"] + e.get("tokLen", 0)].decode(errors="replace")

def walk(node, src, visit, ns=()):
    for key in ("loc", "range"):
        if key in node:
            src.see(node[key]) if key == "loc" else (src.see(node[key].get("begin", {})), src.see(node[key].get("end", {})))
    visit(node, ns)
    sub = ns + (node.get("name", ""),) if node.get("kind") == "NamespaceDecl" else ns
    for child in node.get("inner", []): walk(child, src, visit, sub)

def one_line(s): return re.sub(r"\s+", " ", s or "").strip()

def head(text):
    """A declaration without its body: up to the parameter list's closing
    parenthesis (matched, so defaults such as `= {}` stay) and the qualifiers after it."""
    depth, opened = 0, False
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
            opened = True
        elif ch == ")":
            depth -= 1
            if opened and depth == 0:
                rest = re.match(r"[\s]*((?:const|noexcept|override|final|\s)*)", text[i + 1:]).group(1)
                return one_line(text[: i + 1] + " " + rest)
    return one_line(text.split(";")[0])

def main():
    roots = dump()
    records, enums, functions = {}, {}, {}
    src = Source()
    def visit(n, ns):
        if ns[:1] != ("vecdb",) or (len(ns) > 1 and ns[1] not in ("wal",)): return
        k, name = n.get("kind"), n.get("name", "")
        if k == "CXXRecordDecl" and n.get("completeDefinition") and name and name not in records:
            records[name] = (n, src.file)
        elif k == "EnumDecl" and name and name not in enums:
            enums[name] = [c.get("name") for c in n.get("inner", []) if c.get("kind") == "EnumConstantDecl"]
        elif k in ("FunctionDecl", "FunctionTemplateDecl") and name in FUNCTIONS:
            text = src.text(n.get("range", {}))
            if text:
                sig = head(text)
                if len(ns) > 1: sig = re.sub(rf"\b{name}\(", f"{ns[1]}::{name}(", sig, count=1)  # wal::read_segment
                functions.setdefault(name, [])
                # A template's function appears twice (the template, and the function in it): keep the template.
                if not any(other.endswith(sig) or sig.endswith(other) for other in functions[name]):
                    functions[name].append(sig)
    for r in roots: walk(r, src, visit)
    # Second pass per record: members with their source text.
    def members(rec):
        node, _ = rec
        src2 = Source()
        public = node.get("tagUsed") != "class"
        out = []
        def emit(m, prefix=""):
            k = m.get("kind")
            if m.get("isImplicit") or k in ("AccessSpecDecl",): return
            if k == "FieldDecl" and public:
                t = src2.text(m.get("range", {}))
                if t: out.append(one_line(t) + ";")
            elif k in ("CXXMethodDecl", "CXXConstructorDecl", "FunctionTemplateDecl") and public:
                t = src2.text(m.get("range", {}))
                if not t or "= delete" in t or m.get("name", "").startswith("operator="): return
                sig = head(t)
                out.append(prefix + sig + ";")
        def rec_walk(n, depth):
            nonlocal public
            for key in ("loc", "range"):
                if key in n:
                    src2.see(n[key]) if key == "loc" else (src2.see(n[key].get("begin", {})), src2.see(n[key].get("end", {})))
            if depth == 1:
                if n.get("kind") == "AccessSpecDecl": public = n.get("access") == "public"
                elif n.get("kind") in ("FieldDecl", "CXXMethodDecl", "CXXConstructorDecl", "FunctionTemplateDecl"): emit(n)
                elif n.get("kind") == "FriendDecl" and public:  # operators such as Filter's && and ==
                    for f in n.get("inner", []):
                        t = src2.text(f.get("range", {})) if f.get("kind") == "FunctionDecl" else None
                        if t:
                            sig = head(t)
                            out.append(sig if sig.startswith("friend") else "friend " + sig)
                            if not out[-1].endswith(";"): out[-1] += ";"
            if depth == 0 or n.get("kind") in ("FunctionTemplateDecl",) and depth == 1:
                for c in n.get("inner", []): rec_walk(c, depth + 1) if depth == 0 else None
        # Walk the whole tree once to keep the file tracking right, then only depth-1 children are emitted.
        src2.file = rec[1]
        rec_walk(node, 0)
        return out
    lines = ["<!-- GENERATED by .github/scripts/gen_signatures.py from the headers: do not edit by hand -->", ""]
    for title, names in GROUPS:
        lines += [f"### {title}", ""]
        for name in names:
            if name in enums:
                lines += [f"**`{name}`** (enumeration)", "", "```cpp", f"enum class {name} {{ {', '.join(enums[name])} }};", "```", ""]
            elif name in records:
                body = members(records[name])
                kind = records[name][0].get("tagUsed", "struct")
                lines += [f"**`{name}`**", "", "```cpp", f"{kind} {name} {{"] + [f"    {b}" for b in body] + ["};", "```", ""]
            else:
                sys.exit(f"gen_signatures: {name} not found in the headers")
    lines += ["### Free functions", "", "```cpp"]
    for f in FUNCTIONS:
        for sig in functions.get(f, []): lines.append(sig + ";")
    lines += ["```", "", "<!-- END GENERATED -->"]
    doc = ROOT / "docs" / "developer-reference.md"
    text = doc.read_text()
    start, end = "<!-- GENERATED by .github/scripts/gen_signatures.py", "<!-- END GENERATED -->"
    block = "\n".join(lines)
    if start in text:
        a = text.index(start); b = text.index(end) + len(end)
        text = text[:a] + block + text[b:]
    else:
        text = text.rstrip("\n") + "\n\n---\n\n## Appendix: complete signatures\n\nEvery public type, field and method exactly as declared in the headers: parameter types, names and default values, and every field's type and default. Generated from the compiler's view of the code, so it cannot drift from it; the sections above explain what each one does.\n\n" + block + "\n"
    doc.write_text(text)
    print(f"gen_signatures: {sum(len(n) for _, n in GROUPS)} types, {sum(len(v) for v in functions.values())} functions")

if __name__ == "__main__":
    main()
