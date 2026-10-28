"""hnsw_lite_lib.importers: hnswlib files, FAISS indexes and Chroma collections.

hnswlib files are written here in hnswlib's layout, so the reader is tested without
hnswlib; when hnswlib is installed, it must load those files too (the layout is right,
not only consistent), and its own files must read as its get_items returns them.
"""

import math
import struct

import numpy as np
import pytest

import hnsw_lite_lib as hl
from hnsw_lite_lib import importers

RNG = np.random.default_rng(3)
DATA = RNG.normal(size=(300, 8)).astype(np.float32)
QUERIES = RNG.normal(size=(15, 8)).astype(np.float32)


def write_hnswlib(path, labels, vectors, deleted=(), M=4, ef_construction=100):
    """An hnswlib 0.8 index file: header, level-0 records, then empty upper link lists."""
    n, dim = vectors.shape
    max_m0 = 2 * M
    offset_data = 4 + 4 * max_m0
    label_offset = offset_data + 4 * dim
    size_per = label_offset + 8
    header = struct.pack("<QQQQQQiIQQQdQ", 0, n, n, size_per, label_offset, offset_data, 0, 0, M, max_m0, M,
                         1 / math.log(M), ef_construction)
    records = bytearray()
    for label, vector in zip(labels, vectors):
        flag = 0x01 if int(label) in set(deleted) else 0
        records += struct.pack("<HBB", 0, flag, 0) + bytes(4 * max_m0) + vector.astype("<f4").tobytes() + struct.pack("<Q", int(label))
    with open(path, "wb") as f:
        f.write(header + bytes(records) + b"".join(struct.pack("<I", 0) for _ in range(n)))


def exact_top(vectors, labels, q, k):
    return labels[np.argsort(((vectors - q) ** 2).sum(axis=1), kind="stable")[:k]]


# ---------------------------------------------------------------- hnswlib files


def test_reads_hnswlib_files(tmp_path):
    path = str(tmp_path / "index.bin")
    labels = np.arange(300, dtype=np.uint64) * 7 + 3
    write_hnswlib(path, labels, DATA, deleted={10, 24}, M=6, ef_construction=120)
    got_labels, got_vectors, info = importers.read_hnswlib(path)
    keep = ~np.isin(labels, [10, 24])
    assert np.array_equal(got_labels, labels[keep]) and np.array_equal(got_vectors, DATA[keep])
    assert info == {"M": 6, "ef_construction": 120, "max_elements": 300, "deleted": 2}
    index = importers.from_hnswlib(path, "l2")
    for q in QUERIES:  # the imported index against exact search over the same vectors
        assert list(index.search(q, 5, 300)[0]) == list(exact_top(DATA[keep], labels[keep], q, 5))
    with pytest.raises(ValueError):
        importers.from_hnswlib(path, "manhattan")


def test_hnswlib_files_against_hnswlib(tmp_path):
    hnswlib = pytest.importorskip("hnswlib")
    ours = str(tmp_path / "ours.bin")
    labels = np.arange(300, dtype=np.uint64) + 1000
    write_hnswlib(ours, labels, DATA)
    p = hnswlib.Index(space="l2", dim=8)
    p.load_index(ours)                       # hnswlib reads the layout written here
    assert np.array_equal(np.array(p.get_items(labels)), DATA)
    for space in ("l2", "cosine"):
        real = hnswlib.Index(space=space, dim=8)
        real.init_index(400, M=10, ef_construction=150)
        real.add_items(DATA, labels)
        real.mark_deleted(1005)
        path = str(tmp_path / f"real_{space}.bin")
        real.save_index(path)
        got_labels, got_vectors, info = importers.read_hnswlib(path)
        assert 1005 not in got_labels and len(got_labels) == 299 and info["M"] == 10
        assert np.array_equal(got_vectors, np.array(real.get_items(got_labels)))  # cosine: stored normalized, as hnswlib keeps them


@pytest.mark.parametrize("damage", ["truncate", "flip", "garbage", "huge"])
def test_damaged_hnswlib_files(tmp_path, damage):
    # A binary reader meets damaged and hostile files: every outcome is a clean ValueError
    # or correct data (truncating only the upper link lists, which it does not need).
    good = tmp_path / "good.bin"
    write_hnswlib(str(good), np.arange(12, dtype=np.uint64), DATA[:12])
    raw = good.read_bytes()
    level0_end = 96 + 12 * (4 + 4 * 8 + 4 * 8 + 8)
    bad = tmp_path / "bad.bin"
    cases = []
    if damage == "truncate":
        cases = [raw[:cut] for cut in range(len(raw))]
    elif damage == "flip":
        cases = [raw[:i] + bytes([raw[i] ^ 0xFF]) + raw[i + 1:] for i in range(96)]  # every header byte
    elif damage == "garbage":
        cases = [b"", b"x", bytes(96), RNG.integers(0, 256, 5000, dtype=np.uint8).tobytes()]
    else:  # a header claiming 2^40 elements of 1 MB: refused before reading anything
        cases = [struct.pack("<QQQQQQiIQQQdQ", 0, 1 << 40, 1 << 40, 1 << 20, 100, 36, 0, 0, 4, 8, 4, 0.7, 100) + raw[96:]]
    outcomes = {"refused": 0, "read": 0}
    for content in cases:
        bad.write_bytes(content)
        try:
            labels, vectors, _ = importers.read_hnswlib(str(bad))
        except ValueError:
            outcomes["refused"] += 1
            continue
        assert vectors.ndim == 2 and len(labels) == len(vectors)
        if damage == "truncate":
            assert len(content) >= level0_end and np.array_equal(vectors, DATA[:12])  # only link lists cut
        outcomes["read"] += 1
    assert sum(outcomes.values()) == len(cases) and outcomes["refused"] > 0


# ---------------------------------------------------------------- FAISS


def test_from_faiss(tmp_path):
    faiss = pytest.importorskip("faiss")
    labels = np.arange(300)
    flat = faiss.IndexFlatL2(8)
    flat.add(DATA)
    path = str(tmp_path / "flat.faiss")
    faiss.write_index(flat, path)
    idmap = faiss.IndexIDMap(faiss.IndexFlatL2(8))
    idmap.add_with_ids(DATA, labels.astype(np.int64) * 3 + 7)
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(8), 8, 8)
    ivf.train(DATA)
    ivf.add(DATA)
    hnsw = faiss.IndexHNSWFlat(8, 16)
    hnsw.add(DATA)
    for source, ids in ((path, labels), (idmap, labels * 3 + 7), (ivf, labels), (hnsw, labels)):
        index = importers.from_faiss(source)
        for q in QUERIES:
            assert list(index.search(q, 5, 300)[0]) == list(exact_top(DATA, ids, q, 5))
    ip = faiss.IndexFlatIP(8)
    ip.add(DATA)
    assert importers.from_faiss(ip).metric == hl.Metric.INNER_PRODUCT
    assert importers.from_faiss(ip, metric=hl.Metric.COSINE).metric == hl.Metric.COSINE
    negative = faiss.IndexIDMap(faiss.IndexFlatL2(8))
    negative.add_with_ids(DATA[:2], np.array([-5, 1], dtype=np.int64))
    with pytest.raises(ValueError):
        importers.from_faiss(negative)
    assert len(importers.from_faiss(faiss.IndexFlatL2(8)).search(QUERIES[0], 3)[0]) == 0  # empty


# ---------------------------------------------------------------- Chroma


class FakeChroma:
    """Chroma's collection interface: get(include, limit, offset) and metadata."""

    def __init__(self, n, space="cosine", metadatas=True, documents=True):
        self.metadata = {"hnsw:space": space}
        self.ids = [f"c{i}" for i in range(n)]
        self.metadatas, self.documents = metadatas, documents

    def get(self, include, limit, offset):
        ids = self.ids[offset:offset + limit]
        rows = range(offset, offset + len(ids))
        return {"ids": ids, "embeddings": DATA[list(rows)].tolist() if ids else [],
                "documents": [f"text {i}" for i in rows] if self.documents else None,
                "metadatas": [{"n": i, "lang": "en"} for i in rows] if self.metadatas else None}


def test_from_chroma():
    col = importers.from_chroma(FakeChroma(25), batch=10)  # three pages
    assert len(col) == 25 and col._core.options.metric == hl.Metric.COSINE and col._core.options.string_ids
    doc = col.get("c7")
    assert doc["text"] == "text 7" and doc["metadata"] == {"n": 7, "lang": "en"}
    assert col.search(vector=DATA[7], k=1)[0]["key"] == "c7"
    bare = importers.from_chroma(FakeChroma(5, space="l2", metadatas=False, documents=False))
    assert len(bare) == 5 and not bare._core.options.text
    with pytest.raises(ValueError):
        importers.from_chroma(FakeChroma(0))                # empty: its dimension is unknown
    with pytest.raises(ValueError):
        importers.from_chroma(FakeChroma(3, space="manhattan"))
    with pytest.raises(ValueError):
        importers.from_chroma(FakeChroma(3), batch=0)
