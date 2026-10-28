"""A drop-in for hnswlib's Python API, backed by hnsw_lite.

Change ``import hnswlib`` to ``from hnsw_lite_lib import hnswlib``: the same classes
(``Index``, ``BFIndex``), methods, arguments, return shapes, errors (``RuntimeError``)
and distance conventions (squared L2; ``1 - dot`` for "ip"; ``1 - cosine`` for
"cosine"). What differs:

* Files: ``save_index`` writes hnsw_lite's format (one file), not hnswlib's; files
  are not interchangeable between the two libraries.
* ``random_seed`` and ``ef_construction`` are honored, but graphs are not identical
  to hnswlib's, so approximate results can differ slightly (as between two hnswlib
  builds with different seeds).
* ``BFIndex.knn_query`` with ``k`` beyond the stored vectors raises ``RuntimeError``,
  as ``Index`` does (hnswlib's ``BFIndex`` reads past its results there: undefined).
* Not provided: ``index_file_size``, ``get_neighbors``-style internals, and pickling.

Kept on purpose, to change no numbers when switching: hnswlib's ``BFIndex`` with
"cosine" normalizes the stored vectors but not the query, so its distances are
``1 - |q| * cos`` (the ranking is cosine's). This module returns the same.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable, Optional

import numpy as np

from . import FlatIndex, HnswIndex, HnswParams, Metric

__all__ = ["Index", "BFIndex"]

_SPACES = {"l2": Metric.L2, "ip": Metric.INNER_PRODUCT, "cosine": Metric.COSINE}
_MAGIC = b"HNSWLIB-COMPAT-1"


def _space(space: str):
    if space not in _SPACES:
        raise RuntimeError("Space name must be one of l2, ip, or cosine.")
    return _SPACES[space]


def _as_2d(data, dim: int) -> np.ndarray:
    a = np.asarray(data, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2 or a.shape[1] != dim:
        raise RuntimeError(f"Wrong dimensionality of the vectors: expected {dim}")
    return np.ascontiguousarray(a)


def _labels(ids, count: int, start: int) -> np.ndarray:
    if ids is None:
        return np.arange(start, start + count, dtype=np.uint64)
    labels = np.atleast_1d(np.asarray(ids)).astype(np.uint64)
    if labels.shape[0] != count:
        raise RuntimeError("The number of labels must match the number of vectors")
    return labels


def _to_hnswlib(space: str, distances: np.ndarray) -> np.ndarray:
    # hnsw_lite: inner product is -dot; hnswlib reports 1 - dot. L2 (squared) and
    # cosine (1 - cosine) already agree.
    return (distances + np.float32(1)).astype(np.float32) if space == "ip" else distances.astype(np.float32)


class _Base:
    def __init__(self, space: str, dim: int):
        self.space = space
        self._metric = _space(space)
        if int(dim) <= 0:
            raise RuntimeError("dim must be positive")
        self.dim = int(dim)
        self._index = None
        self._max = 0
        self._live: set[int] = set()

    def _check_init(self):
        if self._index is None:
            raise RuntimeError("Index is not initialized: call init_index or load_index first")

    def _search_one(self, query: np.ndarray, k: int, ef: int, keep: Callable[[int], bool]):
        # A filter is a Python callable: widen until k labels pass it, or none are left.
        n = len(self._live)
        fetch = min(n, max(k * 2, 16))
        while True:
            ids, dists = self._raw_search(query, fetch, max(ef, fetch))
            chosen = [(int(i), float(d)) for i, d in zip(ids, dists) if keep(int(i))]
            if len(chosen) >= k or fetch >= n:
                return chosen[:k]
            fetch = min(n, fetch * 4)

    def knn_query(self, data, k: int = 1, num_threads: int = -1, filter: Optional[Callable[[int], bool]] = None):
        """The k nearest labels for each query: (labels uint64, distances float32), each
        shaped (queries, k). Raises RuntimeError when fewer than k are available."""
        self._check_init()
        queries = _as_2d(data, self.dim)
        k = int(k)
        if k < 0:
            raise RuntimeError("k must be >= 0")
        labels = np.zeros((queries.shape[0], k), dtype=np.uint64)
        dists = np.zeros((queries.shape[0], k), dtype=np.float32)
        if k == 0:
            return labels, dists
        too_few = RuntimeError("Cannot return the results in a contiguous 2D array. Probably ef or M is too small")
        if filter is None:
            ids, d = self._raw_batch(queries, k, num_threads)
            if (ids == np.iinfo(np.uint64).max).any():  # hnsw_lite pads a short result with this ID (and infinity)
                raise too_few
            return ids.astype(np.uint64), self._finish(queries, d)
        for row, q in enumerate(queries):
            chosen = self._search_one(q, k, self._ef(), filter)
            if len(chosen) < k:
                raise too_few
            labels[row] = [c[0] for c in chosen]
            dists[row] = [c[1] for c in chosen]
        return labels, self._finish(queries, dists)

    def _finish(self, queries: np.ndarray, distances: np.ndarray) -> np.ndarray:
        return _to_hnswlib(self.space, distances)

    def get_max_elements(self) -> int:
        return self._max

    # One file: hnsw_lite's index, plus what hnswlib keeps beside it (the labels, the
    # marked-deleted vectors, max_elements and the search settings).
    _kind = None  # HnswIndex or FlatIndex

    def save_index(self, path: str):
        self._check_init()
        with tempfile.TemporaryDirectory() as tmp:
            inner = os.path.join(tmp, "index.idx")
            self._index.save(inner)
            with open(inner, "rb") as f:
                body = f.read()
        deleted_map = getattr(self, "_deleted", {})
        deleted = np.array(sorted(deleted_map), dtype=np.uint64)
        vectors = np.array([deleted_map[int(x)] for x in deleted], dtype=np.float32).reshape(len(deleted), self.dim)
        header = np.array([self._max, getattr(self, "ef", 0), getattr(self, "M", 0), getattr(self, "ef_construction", 0)], dtype=np.int64)
        with open(path, "wb") as f:
            f.write(_MAGIC)
            np.savez(f, body=np.frombuffer(body, dtype=np.uint8), header=header, live=np.array(sorted(self._live), dtype=np.uint64),
                     deleted=deleted, vectors=vectors, space=np.array(self.space))

    def load_index(self, path: str, max_elements: int = 0, allow_replace_deleted: bool = False):
        with open(path, "rb") as f:
            if f.read(len(_MAGIC)) != _MAGIC:
                raise RuntimeError("Not an index saved by hnsw_lite_lib.hnswlib (hnswlib's own files are not readable)")
            data = np.load(f)
            body, header, live = data["body"].tobytes(), data["header"], data["live"]
            deleted, vectors, space = data["deleted"], data["vectors"], str(data["space"])
        if space != self.space:
            raise RuntimeError(f"The index was saved with space {space!r}, not {self.space!r}")
        with tempfile.TemporaryDirectory() as tmp:
            inner = os.path.join(tmp, "index.idx")
            with open(inner, "wb") as f:
                f.write(body)
            index = self._kind.load(inner)
        if index.dim != self.dim:
            raise RuntimeError(f"The index has dimension {index.dim}, not {self.dim}")
        self._index = index
        self._max = max(int(header[0]), int(max_elements))
        if hasattr(self, "_deleted"):
            self.ef, self.M, self.ef_construction = int(header[1]), int(header[2]), int(header[3])
            self._allow_replace = bool(allow_replace_deleted)
            self._deleted = {int(label): vectors[i] for i, label in enumerate(deleted)}
        self._live = set(int(x) for x in live)

    @property
    def max_elements(self) -> int:
        return self._max

    def _ef(self) -> int:
        return 0


class Index(_Base):
    """hnswlib.Index: an HNSW graph (hnsw_lite's HnswIndex)."""

    _kind = HnswIndex

    def __init__(self, space: str, dim: int):
        super().__init__(space, dim)
        self._deleted: dict[int, np.ndarray] = {}  # marked deleted: their vectors, so unmark_deleted can restore them
        self._allow_replace = False                # init_index(allow_replace_deleted=): add_items may reuse their places
        self.ef = 10
        self.num_threads = -1
        self.M = 16
        self.ef_construction = 200

    def _ef(self) -> int:
        return self.ef

    def init_index(self, max_elements: int, M: int = 16, ef_construction: int = 200, random_seed: int = 100,
                   allow_replace_deleted: bool = False):
        params = HnswParams()
        params.M = int(M)
        params.ef_construction = int(ef_construction)
        params.seed = int(random_seed)
        self._index = HnswIndex(self.dim, self._metric, params)
        self._max = int(max_elements)
        self.M, self.ef_construction = int(M), int(ef_construction)
        self._live.clear()
        self._deleted.clear()
        self._allow_replace = bool(allow_replace_deleted)

    def set_ef(self, ef: int):
        self.ef = int(ef)

    def set_num_threads(self, num_threads: int):
        self.num_threads = int(num_threads)

    def add_items(self, data, ids=None, num_threads: int = -1, replace_deleted: bool = False):
        """Adds vectors; an existing label is replaced (and undeleted, as in hnswlib). With
        replace_deleted (allowed by init_index), each new label takes a deleted element's
        place while any remain, as hnswlib does: the count does not grow, and the replaced
        labels are gone."""
        self._check_init()
        if replace_deleted and not self._allow_replace:
            raise RuntimeError("Replacement of deleted elements is disabled in constructor")
        vectors = _as_2d(data, self.dim)
        labels = _labels(ids, vectors.shape[0], self.get_current_count())
        fresh = [i for i, label in enumerate(labels) if int(label) not in self._live and int(label) not in self._deleted]
        if len(set(int(x) for x in labels)) != len(labels):
            raise RuntimeError("Duplicate labels in one call")
        reused = min(len(fresh), len(self._deleted)) if replace_deleted else 0
        if self.get_current_count() - reused + len(fresh) > self._max:
            raise RuntimeError("The number of elements exceeds the specified limit")
        for label in sorted(self._deleted)[:reused]:  # their places taken (hnswlib takes whichever it finds first)
            del self._deleted[label]
        if fresh:
            self._index.add(labels[fresh], vectors[fresh])
            self._live.update(int(labels[i]) for i in fresh)
        for i, label in enumerate(labels):
            label = int(label)
            if label in self._live and i not in fresh:
                self._index.upsert(label, vectors[i])
            elif label in self._deleted:
                del self._deleted[label]
                self._index.add(label, vectors[i])
                self._live.add(label)

    def _raw_search(self, query, k, ef):
        return self._index.search(query, k, ef)

    def _raw_batch(self, queries, k, num_threads):
        return self._index.search_batch(queries, k, max(self.ef, k), threads=max(0, int(num_threads)))

    def mark_deleted(self, label: int):
        self._check_init()
        label = int(label)
        if label in self._deleted:
            raise RuntimeError("The requested to delete element is already deleted")
        if label not in self._live:
            raise RuntimeError("Label not found")
        self._deleted[label] = np.asarray(self._index.get_vector(label), dtype=np.float32)
        self._index.remove(label)
        self._live.discard(label)

    def unmark_deleted(self, label: int):
        self._check_init()
        label = int(label)
        if label not in self._deleted:
            raise RuntimeError("The requested to undelete element is not deleted" if label in self._live else "Label not found")
        self._index.add(label, self._deleted.pop(label))
        self._live.add(label)

    def resize_index(self, new_size: int):
        if int(new_size) < self.get_current_count():
            raise RuntimeError("Cannot resize, max element is less than the current number of elements")
        self._max = int(new_size)

    def get_items(self, ids=None, return_type: str = "numpy"):
        self._check_init()
        labels = self.get_ids_list() if ids is None else [int(x) for x in np.atleast_1d(ids)]
        rows = []
        for label in labels:
            if label in self._deleted:
                rows.append(self._deleted[label])
            elif label in self._live:
                rows.append(np.asarray(self._index.get_vector(label), dtype=np.float32))
            else:
                raise RuntimeError("Label not found")
        if return_type == "numpy":
            return np.array(rows, dtype=np.float32).reshape(len(rows), self.dim)
        if return_type == "list":
            return [r.tolist() for r in rows]
        raise RuntimeError("return_type must be 'numpy' or 'list'")

    def get_ids_list(self) -> list:
        return sorted(self._live | set(self._deleted))  # hnswlib lists deleted labels too

    def get_current_count(self) -> int:
        return len(self._live) + len(self._deleted)  # as hnswlib counts: deleted elements included

    @property
    def element_count(self) -> int:
        return self.get_current_count()


class BFIndex(_Base):
    """hnswlib.BFIndex: exact search (hnsw_lite's FlatIndex)."""

    _kind = FlatIndex

    def init_index(self, max_elements: int):
        self._index = FlatIndex(self.dim, self._metric)
        self._max = int(max_elements)
        self._live.clear()

    def add_items(self, data, ids=None):
        self._check_init()
        vectors = _as_2d(data, self.dim)
        labels = _labels(ids, vectors.shape[0], len(self._live))
        fresh = [i for i, label in enumerate(labels) if int(label) not in self._live]
        if len(self._live) + len(fresh) > self._max:
            raise RuntimeError("The number of elements exceeds the specified limit")
        for i, label in enumerate(labels):
            self._index.upsert(int(label), vectors[i])
            self._live.add(int(label))

    def _finish(self, queries: np.ndarray, distances: np.ndarray) -> np.ndarray:
        # hnswlib's BFIndex normalizes stored vectors for cosine but not the query: its
        # distance is 1 - |q| cos (hnsw_lite's is 1 - cos). Reproduced, so numbers match.
        if self.space != "cosine":
            return _to_hnswlib(self.space, distances)
        norms = np.linalg.norm(queries.astype(np.float64), axis=1, keepdims=True)
        return (1.0 - norms * (1.0 - distances.astype(np.float64))).astype(np.float32)

    def delete_vector(self, label: int):
        self._check_init()
        if not self._index.remove(int(label)):
            raise RuntimeError("Label not found")
        self._live.discard(int(label))

    def _raw_search(self, query, k, ef):
        return self._index.search(query, k)

    def _raw_batch(self, queries, k, num_threads):
        return self._index.search_batch(queries, k, threads=max(0, int(num_threads)))

    def get_current_count(self) -> int:
        return len(self._live)
