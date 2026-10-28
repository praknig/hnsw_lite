"""Moving existing data into hnsw_lite.

* ``from_hnswlib(path, space)``: an hnswlib index file, read directly (no hnswlib needed).
* ``from_faiss(index_or_path)``: a FAISS index (needs the faiss package).
* ``from_chroma(collection)``: a Chroma collection, with its ids, documents and metadata.

Each copies the stored vectors into a new hnsw_lite index or collection, which builds its
own graph: results match the source closely (exactly, for exact indexes), not bit for bit.
"""

from __future__ import annotations

import os
import struct
from typing import Optional

import numpy as np

from . import Collection, HnswIndex, HnswParams, Metric

__all__ = ["from_hnswlib", "from_faiss", "from_chroma"]

_SPACES = {"l2": Metric.L2, "ip": Metric.INNER_PRODUCT, "cosine": Metric.COSINE}


# ---------------------------------------------------------------- hnswlib

# hnswlib's saveIndex (hnswalg.h): offsetLevel0, max_elements, cur_element_count,
# size_data_per_element, label_offset, offsetData (size_t each), maxlevel (int),
# enterpoint_node (uint32), maxM, maxM0, M (size_t), mult (double), ef_construction
# (size_t); then cur_element_count records of size_data_per_element bytes: a link-list
# header (uint32: the count, and the deleted mark 0x01 in its third byte), the links,
# the vector (float32) at offsetData, the label (size_t) at label_offset.
_HEADER = struct.Struct("<QQQQQQiIQQQdQ")


def read_hnswlib(path: str):
    """The labels, vectors and settings of an hnswlib index file: (labels uint64,
    vectors float32 of shape (n, dim), {"M", "ef_construction", "max_elements",
    "deleted"}). Elements marked deleted are left out (counted in "deleted")."""
    with open(path, "rb") as f:
        head = f.read(_HEADER.size)
        if len(head) != _HEADER.size:
            raise ValueError("not an hnswlib index file: shorter than its header")
        (offset_level0, max_elements, count, size_per, label_offset, offset_data, _maxlevel, _entry,
         _max_m, _max_m0, m, _mult, ef_construction) = _HEADER.unpack(head)
        dim_bytes = label_offset - offset_data
        if (offset_level0 != 0 or count > max_elements or size_per < label_offset + 8 or offset_data < 4
                or dim_bytes <= 0 or dim_bytes % 4 or size_per > (1 << 32)):
            raise ValueError("not an hnswlib index file (or one of an unsupported version)")
        # Checked against the file before reading: a damaged header must not make it
        # allocate what the file does not hold.
        if count * size_per > os.path.getsize(path) - _HEADER.size:
            raise ValueError("hnswlib index file truncated (or its header damaged)")
        data = f.read(count * size_per)
        if len(data) != count * size_per:
            raise ValueError("hnswlib index file truncated")
    records = np.frombuffer(data, dtype=np.uint8).reshape(count, size_per)
    deleted = (records[:, 2] & 0x01).astype(bool)
    dim = dim_bytes // 4
    vectors = records[:, offset_data:label_offset].copy().view(np.float32).reshape(count, dim)
    labels = records[:, label_offset:label_offset + 8].copy().view(np.uint64).reshape(count)
    keep = ~deleted
    return labels[keep], np.ascontiguousarray(vectors[keep]), {
        "M": int(m), "ef_construction": int(ef_construction), "max_elements": int(max_elements), "deleted": int(deleted.sum())}


def from_hnswlib(path: str, space: str, *, M: Optional[int] = None, ef_construction: Optional[int] = None) -> HnswIndex:
    """An HnswIndex with the vectors and labels of an hnswlib index file. `space` is the
    one the file was built with ("l2", "ip" or "cosine"; hnswlib does not store it). M
    and ef_construction default to the file's. hnswlib stores cosine vectors normalized,
    which cosine search is unaffected by."""
    if space not in _SPACES:
        raise ValueError("space must be 'l2', 'ip' or 'cosine'")
    labels, vectors, info = read_hnswlib(path)
    params = HnswParams()
    params.M = int(M or info["M"])
    params.ef_construction = int(ef_construction or info["ef_construction"])
    index = HnswIndex(vectors.shape[1], _SPACES[space], params)
    if len(labels):
        index.add(labels, vectors)
    return index


# ---------------------------------------------------------------- FAISS


def from_faiss(index_or_path, *, metric: Optional[Metric] = None, M: int = 16, ef_construction: int = 200) -> HnswIndex:
    """An HnswIndex with the vectors and IDs of a FAISS index (an object, or a file to
    read with faiss). The metric follows the index (L2, or inner product); pass
    metric=Metric.COSINE for vectors normalized for cosine. Indexes that store
    compressed codes (PQ, SQ) give their reconstructions, not the original vectors."""
    try:
        import faiss
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("from_faiss needs faiss: pip install faiss-cpu") from e
    index = faiss.read_index(str(index_or_path)) if isinstance(index_or_path, (str, bytes)) or hasattr(index_or_path, "__fspath__") \
        else index_or_path
    ids = None
    inner = index
    if hasattr(index, "id_map"):  # IndexIDMap / IndexIDMap2: custom IDs over an inner index
        ids = faiss.vector_to_array(index.id_map).astype(np.int64)
        inner = faiss.downcast_index(index.index)
    n, d = int(index.ntotal), int(index.d)
    if hasattr(inner, "make_direct_map"):  # IVF indexes reconstruct only with a direct map
        inner.make_direct_map()
    vectors = inner.reconstruct_n(0, n) if n else np.zeros((0, d), dtype=np.float32)
    if ids is None:
        ids = np.arange(n, dtype=np.int64)
    if (ids < 0).any():
        raise ValueError("from_faiss: negative IDs cannot be stored")
    if metric is None:
        metric = Metric.INNER_PRODUCT if index.metric_type == faiss.METRIC_INNER_PRODUCT else Metric.L2
        if index.metric_type not in (faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2):
            raise ValueError("from_faiss: only L2 and inner-product indexes are supported")
    params = HnswParams()
    params.M, params.ef_construction = int(M), int(ef_construction)
    out = HnswIndex(d, metric, params)
    if n:
        out.add(ids.astype(np.uint64), np.ascontiguousarray(vectors, dtype=np.float32))
    return out


# ---------------------------------------------------------------- Chroma

_CHROMA_SPACES = {"l2": "l2", "ip": "ip", "cosine": "cosine"}


def from_chroma(collection, *, batch: int = 1000, **options) -> Collection:
    """A Collection with a Chroma collection's IDs (as string keys), embeddings, documents
    (as text) and metadata, read `batch` at a time. The metric follows the collection's
    "hnsw:space" setting (Chroma's default is "l2"); other Collection options may be passed."""
    if batch <= 0:
        raise ValueError("from_chroma: batch must be positive")
    space = (getattr(collection, "metadata", None) or {}).get("hnsw:space", "l2")
    if space not in _CHROMA_SPACES:
        raise ValueError(f"from_chroma: unsupported space {space!r}")
    out = None
    offset = 0
    while True:
        page = collection.get(include=["embeddings", "documents", "metadatas"], limit=batch, offset=offset)
        ids = list(page.get("ids") or [])
        if not ids:
            break
        embeddings = page.get("embeddings")
        if embeddings is None or len(embeddings) != len(ids):
            raise ValueError("from_chroma: the collection returned no embeddings")
        vectors = np.asarray(embeddings, dtype=np.float32)
        documents = page.get("documents") or [None] * len(ids)
        metadatas = page.get("metadatas") or [None] * len(ids)
        if out is None:
            settings = dict(dim=vectors.shape[1], metric=_CHROMA_SPACES[space], string_ids=True,
                            text=any(d is not None for d in documents))
            settings.update(options)
            out = Collection(**settings)
        out.upsert_many([str(i) for i in ids], vectors, documents if out._core.options.text else None, None,
                        [dict(m) if m else {} for m in metadatas])
        offset += len(ids)
    if out is None:
        raise ValueError("from_chroma: the collection is empty (its dimension is unknown)")
    return out
