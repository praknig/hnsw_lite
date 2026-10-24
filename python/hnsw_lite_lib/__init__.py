"""hnsw_lite_lib: the Python package of hnsw-lite, vector search with metadata filters, hybrid (dense + BM25) search,
durability and automatic maintenance.

    import hnsw_lite_lib as hl

    docs = hl.Collection(dim=384, path="docs.db")           # durable; omit path for in memory
    docs.upsert(1, vector=embedding, text="the text", metadata={"lang": "en"})
    hits = docs.search(vector=query_embedding, text="what the user typed", k=10, filter={"lang": "en"})

Vectors are NumPy arrays (any float dtype; converted to float32) or anything
NumPy accepts. Metadata values are int, float, bool, str, or lists of str (tags).
pandas is optional: upsert_dataframe() and search(..., as_dataframe=True) use it.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from . import _core
from ._core import format_date, now_ms, parse_date  # noqa: F401  (re-exported)
from ._core import (  # noqa: F401  (re-exported)
    Cancelled,
    Codec,
    CollectionOptions,
    Durability,
    DurableOptions,
    FieldType,
    Filter,
    FlatIndex,
    Fusion,
    HnswIndex,
    HnswParams,
    IvfParams,
    Maintenance,
    MaintenancePolicy,
    Metric,
    MISSING_ID,
    Preset,
    QuantizationParams,
    Strategy,
    FormatError,
)

__all__ = [
    "to_ms", "parse_date", "format_date", "now_ms",
    "Collection", "HnswIndex", "FlatIndex", "Filter", "Metric", "Preset", "Fusion", "Durability", "Maintenance", "Strategy",
    "FieldType", "Codec", "HnswParams", "QuantizationParams", "IvfParams", "DurableOptions", "MaintenancePolicy",
    "CollectionOptions", "FormatError", "Cancelled", "MISSING_ID",
]
__version__ = "0.1.0"

# ---------------------------------------------------------------- names

_METRICS = {"l2": Metric.L2, "euclidean": Metric.L2, "ip": Metric.INNER_PRODUCT, "inner_product": Metric.INNER_PRODUCT,
            "dot": Metric.INNER_PRODUCT, "cosine": Metric.COSINE}
_PRESETS = {"balanced": Preset.BALANCED, "high_recall": Preset.HIGH_RECALL, "low_memory": Preset.LOW_MEMORY}
_DURABILITY = {"buffered": Durability.BUFFERED, "grouped": Durability.GROUPED, "synced": Durability.SYNCED}
_MAINTENANCE = {"manual": Maintenance.MANUAL, "background": Maintenance.BACKGROUND}
_FUSION = {"rrf": Fusion.RRF, "weighted": Fusion.WEIGHTED}
_FIELD_TYPES = {"int": FieldType.INT, int: FieldType.INT, "float": FieldType.FLOAT, float: FieldType.FLOAT, "bool": FieldType.BOOL,
                bool: FieldType.BOOL, "keyword": FieldType.KEYWORD, "str": FieldType.KEYWORD, str: FieldType.KEYWORD, "tags": FieldType.TAGS,
                list: FieldType.TAGS}


def _choose(value, table: Mapping, what: str):
    if isinstance(value, str):
        try:
            return table[value.lower()]
        except KeyError:
            raise ValueError(f"unknown {what} {value!r}; expected one of {sorted(k for k in table if isinstance(k, str))}") from None
    return value


def _schema(schema: Optional[Mapping[str, Any]]):
    if schema is None:
        return None
    return {name: (_FIELD_TYPES[t] if not isinstance(t, FieldType) else t) for name, t in schema.items()}


def to_ms(value) -> int:
    """A date as milliseconds since 1970 (UTC), the form dates are stored and
    compared in: from a datetime (naive ones are taken as UTC), a date (midnight
    UTC), an ISO 8601 string, a pandas Timestamp, a NumPy datetime64, or a number
    (returned as an int)."""
    import datetime as _dt
    if isinstance(value, bool):
        raise TypeError("to_ms: expected a date, not a bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(round(float(value)))
    if isinstance(value, str):
        return parse_date(value)
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[ms]").astype(np.int64))
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp
        value = value.to_pydatetime()
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        return int(round(value.timestamp() * 1000))
    if isinstance(value, _dt.date):
        return int(_dt.datetime(value.year, value.month, value.day, tzinfo=_dt.timezone.utc).timestamp() * 1000)
    raise TypeError(f"to_ms: expected a date, got {type(value).__name__}")


def _is_date(value) -> bool:
    import datetime as _dt
    return isinstance(value, (_dt.date, np.datetime64)) or hasattr(value, "to_pydatetime")


_HALF_LIFE_UNITS = {"half_life": 1, "half_life_seconds": 1000, "half_life_minutes": 60000, "half_life_hours": 3600000, "half_life_days": 86400000}


def _modifiers(spec):
    """The canonical form of a modifiers specification (see Collection.search)."""
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise TypeError("modifiers: expected a dict")
    unknown = set(spec) - {"decay", "multiply", "boost", "pool"}
    if unknown:
        raise ValueError(f"modifiers: unknown keys {sorted(unknown)}; expected decay, multiply, boost, pool")
    def many(x, one):  # one rule, or a list of them
        return [] if x is None else [x] if one(x) else list(x)
    def one_boost(b):
        return isinstance(b, Mapping) or (isinstance(b, tuple) and len(b) == 2 and isinstance(b[1], (int, float)))
    out = {"decays": [], "multipliers": [], "boosts": []}
    for d in many(spec.get("decay"), lambda x: isinstance(x, Mapping)):
        units = [k for k in _HALF_LIFE_UNITS if k in d]
        if len(units) != 1:
            raise ValueError("modifiers: a decay needs exactly one of " + ", ".join(_HALF_LIFE_UNITS))
        unknown = set(d) - {"field", "origin", "shape", "missing", "floor"} - set(_HALF_LIFE_UNITS)
        if unknown or "field" not in d:
            raise ValueError(f"modifiers: a decay takes field, origin, shape, missing, floor and a half-life (unknown: {sorted(unknown)})")
        shape = d.get("shape", "exponential")
        if shape not in ("exponential", "linear"):
            raise ValueError("modifiers: a decay's shape is 'exponential' or 'linear'")
        out["decays"].append({"field": d["field"], "origin": float(to_ms(d["origin"])) if "origin" in d else float(now_ms()),
                              "half_life": float(d[units[0]]) * _HALF_LIFE_UNITS[units[0]], "shape": shape,
                              "missing": float(d.get("missing", 1.0)), "floor": float(d.get("floor", 0.0))})
    for m in many(spec.get("multiply"), lambda x: isinstance(x, (str, Mapping))):
        m = {"field": m} if isinstance(m, str) else dict(m)
        unknown = set(m) - {"field", "missing", "min", "max"}
        if unknown or "field" not in m:
            raise ValueError(f"modifiers: a multiplier takes field, missing, min, max (unknown: {sorted(unknown)})")
        out["multipliers"].append({"field": m["field"], "missing": float(m.get("missing", 1.0)), "min": float(m.get("min", 0.0)),
                                   "max": float(m.get("max", float("inf")))})
    for b in many(spec.get("boost"), one_boost):
        if isinstance(b, Mapping):
            when, bonus = b.get("when"), b.get("bonus")
        else:
            when, bonus = b
        out["boosts"].append({"when": as_filter(when), "bonus": float(bonus)})
    if "pool" in spec:
        out["pool"] = int(spec["pool"])
    return out


def as_filter(f: Union[None, Filter, Mapping[str, Any]]) -> Optional[Filter]:
    """A Filter from a Filter, None, or a dict: {"lang": "en", "year": [2023, 2024]}
    means lang == "en" and year in (2023, 2024)."""
    if f is None or isinstance(f, Filter):
        return f
    if isinstance(f, Mapping):
        out = None
        for field, value in f.items():
            if isinstance(value, (list, tuple, set, frozenset)):
                part = Filter.in_(field, [to_ms(v) if _is_date(v) else v for v in value])
            else:
                part = Filter.eq(field, to_ms(value) if _is_date(value) else value)
            out = part if out is None else out & part
        return out if out is not None else Filter.all()
    raise TypeError("filter: expected a Filter, a dict or None")


def _missing(value) -> bool:
    """None, NaN, pd.NA and NaT: no value (dropped from metadata)."""
    if value is None:
        return True
    try:
        import pandas as pd  # noqa: F401
        if value is pd.NA or value is pd.NaT:
            return True
    except ImportError:
        pass
    return isinstance(value, float) and math.isnan(value) or (isinstance(value, np.floating) and np.isnan(value))


def _ids(ids) -> np.ndarray:
    """IDs as uint64, checked first: casting a negative NumPy integer to an unsigned
    type would wrap it around silently."""
    a = np.asarray(ids)
    if a.size == 0:
        return np.zeros(0, dtype=np.uint64)
    if a.dtype.kind not in "iu" or a.dtype == np.bool_:
        raise TypeError(f"ids: expected integers, got {a.dtype}")
    if a.dtype.kind == "i" and (a < 0).any():
        raise ValueError("ids must be non-negative")
    return a.astype(np.uint64).reshape(-1)


def _id(id) -> int:
    if isinstance(id, (bool, np.bool_)) or not isinstance(id, (int, np.integer)):
        raise TypeError(f"id: expected an integer, got {type(id).__name__}")
    if id < 0:
        raise ValueError("ids must be non-negative")
    return int(id)


def _clean(md: Optional[Mapping[str, Any]]) -> Optional[dict]:
    """Metadata from a pandas row: NumPy scalars as Python values, missing values dropped."""
    if md is None:
        return None
    out = {}
    for k, v in md.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if _is_date(v) and not _missing(v):
            out[str(k)] = to_ms(v)
            continue
        if isinstance(v, (list, tuple, set)):
            out[str(k)] = [str(x) for x in v]
        elif not _missing(v):
            out[str(k)] = v.item() if isinstance(v, np.generic) else v
    return out


# ---------------------------------------------------------------- indexes

def _index_add(self, ids, vectors, metadata=None, threads: int = 0) -> None:
    """Adds one vector (an int id and a 1-D vector) or many (ids and a 2-D array).
    Everything is checked first: on any error nothing is added."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if np.isscalar(ids) or isinstance(ids, (np.integer,)):
        ids = [_id(ids)]
        if metadata is not None and isinstance(metadata, Mapping):
            metadata = [metadata]
    if metadata is not None:
        metadata = [_clean(m) for m in metadata]
    self._add(_ids(ids), vectors, metadata, threads)


HnswIndex.add = _index_add
FlatIndex.add = _index_add


def _hnsw_search(self, query, k: int = 10, ef: int = 64, filter=None, strategy=Strategy.AUTO, modifiers=None, diversity=None):
    """(ids, distances), closest first; with `modifiers` or `diversity` (see
    Collection.search), (ids, distances, scores), in the re-ranked order."""
    return self._search(query, k, ef, as_filter(filter), strategy, _modifiers(modifiers), diversity)


def _flat_search(self, query, k: int = 10, filter=None, modifiers=None, diversity=None):
    """(ids, distances), closest first; with `modifiers` or `diversity`, (ids, distances, scores)."""
    return self._search(query, k, as_filter(filter), _modifiers(modifiers), diversity)


HnswIndex.search = _hnsw_search
FlatIndex.search = _flat_search


# ---------------------------------------------------------------- Collection

def _ttl_ms(default_ttl) -> int:
    """A default lifetime (seconds, or a timedelta) in milliseconds; 0 for none."""
    if default_ttl is None:
        return 0
    seconds = default_ttl.total_seconds() if hasattr(default_ttl, "total_seconds") else float(default_ttl)
    if seconds <= 0:
        raise ValueError("default_ttl must be positive")
    return int(round(seconds * 1000))


class Collection:
    """Documents with any of a dense vector, text (BM25) and a sparse vector, plus
    metadata, under one ID space; hybrid search by default.

    Collection(dim)                     in memory
    Collection(dim, path="docs.db")     durable: created, or recovered after a crash
    Collection.load("docs.col")         from a file written by save()

    Close a durable collection with close() or a with-statement: it flushes its log
    and releases the directory's lock.
    """

    def __init__(self, dim: int = 0, *, path: Optional[str] = None, metric: Union[str, Metric] = "cosine",
                 preset: Union[str, Preset] = "balanced", expected_size: int = 100_000, text: bool = True, sparse: bool = False,
                 store_text: bool = True, schema: Optional[Mapping[str, Any]] = None, strict: bool = False,
                 durability: Union[str, Durability] = "grouped", maintenance: Union[str, Maintenance, None] = None,
                 ef: Optional[int] = None, bm25_avg_doc_length: Optional[float] = None, ttl: bool = False, default_ttl=None,
                 string_ids: bool = False, options: Optional[CollectionOptions] = None) -> None:
        if options is None:
            options = CollectionOptions.preset(dim, _choose(preset, _PRESETS, "preset"), expected_size)
            options.metric = _choose(metric, _METRICS, "metric")
            options.text = text
            options.sparse = sparse
            options.store_text = store_text
            if schema is not None or strict:
                options.set_schema(_schema(schema), strict)
            durable = options.durability
            durable.durability = _choose(durability, _DURABILITY, "durability")
            options.durability = durable
            if maintenance is not None:
                policy = options.maintenance
                policy.mode = _choose(maintenance, _MAINTENANCE, "maintenance")
                options.maintenance = policy
            if ef is not None:
                options.ef = ef
            if bm25_avg_doc_length is not None:
                bm25 = options.bm25
                bm25.avg_doc_length = bm25_avg_doc_length
                options.bm25 = bm25
            options.ttl_enabled = bool(ttl or default_ttl is not None)
            options.ttl_default_ms = _ttl_ms(default_ttl)
            options.string_ids = bool(string_ids)
        self._c = _core.Collection.open(str(path), options) if path is not None else _core.Collection(options)
        self.last_search_stopped_early = False

    @classmethod
    def load(cls, path: str, *, ttl: bool = False, default_ttl=None, string_ids: bool = False) -> "Collection":
        """A collection from a file written by save(). Maintenance starts manual;
        expiry and string keys are off unless given here (as for the constructor)."""
        self = cls.__new__(cls)
        self._c = _core.Collection.load(str(path))
        self.last_search_stopped_early = False
        self._c._set_runtime(bool(ttl or default_ttl is not None), _ttl_ms(default_ttl), bool(string_ids))
        return self

    # -- lifetime
    def close(self) -> None:
        """Closes the collection: stops background maintenance, flushes the log and
        releases the directory's lock (durable collections). Further use raises."""
        self._c = None

    def __enter__(self) -> "Collection":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def _core(self):
        if self._c is None:
            raise ValueError("the collection is closed")
        return self._c

    def __len__(self) -> int:
        return len(self._core)

    def __contains__(self, id) -> bool:
        return id in self._core

    def __repr__(self) -> str:
        if self._c is None:
            return "<hnsw_lite_lib.Collection (closed)>"
        o = self._c.options
        return f"<hnsw_lite_lib.Collection: {len(self._c)} documents, dim {o.dim}, text {o.text}, sparse {o.sparse}, durable {self._c.durable}>"

    @property
    def options(self) -> CollectionOptions:
        return self._core.options

    @property
    def durable(self) -> bool:
        return self._core.durable

    # -- changes
    @staticmethod
    def _key(id):
        return id if isinstance(id, str) else _id(id)

    def add(self, id, vector=None, text: Optional[str] = None, sparse=None, metadata: Optional[Mapping[str, Any]] = None,
            expires_at=None) -> None:
        """Adds a document; ValueError if the id (or string key) exists or a part is
        invalid. expires_at (with ttl): a date, ISO string or milliseconds since 1970."""
        self._core.add(self._key(id), vector, text, sparse, _clean(metadata), None if expires_at is None else to_ms(expires_at))

    def upsert(self, id, vector=None, text: Optional[str] = None, sparse=None, metadata: Optional[Mapping[str, Any]] = None,
               expires_at=None) -> bool:
        """Adds or replaces the whole document (parts left out are removed); True if added."""
        return self._core.upsert(self._key(id), vector, text, sparse, _clean(metadata), None if expires_at is None else to_ms(expires_at))

    def upsert_many(self, ids: Sequence, vectors=None, texts: Optional[Iterable[Optional[str]]] = None,
                    sparse: Optional[Iterable] = None, metadata: Optional[Iterable[Mapping[str, Any]]] = None, expires_at=None) -> int:
        """Upserts many documents: ids, a 2-D array of vectors, and lists of texts,
        sparse vectors and metadata dicts (each optional, one entry per id).
        Returns how many were added."""
        if metadata is not None:
            metadata = [_clean(m) for m in metadata]
        if vectors is not None:
            vectors = np.asarray(vectors, dtype=np.float32)
        ids = list(ids) if not isinstance(ids, np.ndarray) else ids
        expires = None if expires_at is None else [None if _missing(e) else to_ms(e) for e in expires_at]
        if len(ids) and isinstance(ids[0], str):  # string keys
            return self._core._upsert_many_keyed([str(i) for i in ids], vectors, texts, sparse, metadata, expires)
        if expires is not None:  # expiry times with numeric ids: one document at a time
            texts = list(texts) if texts is not None else None
            sparse = list(sparse) if sparse is not None else None
            added = 0
            for i, id in enumerate(_ids(ids)):
                added += self._core.upsert(int(id), None if vectors is None else vectors[i], None if texts is None else texts[i],
                                           None if sparse is None else sparse[i], None if metadata is None else metadata[i], expires[i])
            return added
        return self._core._upsert_many(_ids(ids), vectors, texts, sparse, metadata)

    def upsert_dataframe(self, df, *, id_column: Optional[str] = None, vector_column: Optional[str] = None,
                         text_column: Optional[str] = None, sparse_column: Optional[str] = None,
                         metadata_columns: Optional[Sequence[str]] = None, expires_column: Optional[str] = None) -> int:
        """Upserts the rows of a pandas DataFrame. IDs come from id_column, or the
        index; vector_column holds arrays or lists (one vector per row); every other
        column becomes metadata unless metadata_columns says which. Missing values
        (NaN, None, pd.NA) are left out of a row's metadata. Returns how many were added."""
        ids = df[id_column].to_numpy() if id_column is not None else df.index.to_numpy()
        used = {c for c in (id_column, vector_column, text_column, sparse_column, expires_column) if c is not None}
        cols = list(metadata_columns) if metadata_columns is not None else [c for c in df.columns if c not in used]
        vectors = np.stack([np.asarray(v, dtype=np.float32) for v in df[vector_column]]) if vector_column is not None and len(df) else None
        texts = [None if _missing(t) else str(t) for t in df[text_column]] if text_column is not None else None
        sparse = list(df[sparse_column]) if sparse_column is not None else None
        metadata = df[cols].to_dict("records") if cols else None
        expires = list(df[expires_column]) if expires_column is not None else None
        if ids.dtype == object:
            ids = [str(i) for i in ids]
        return self.upsert_many(ids, vectors, texts, sparse, metadata, expires)

    def remove(self, ids) -> int:
        """Removes one id (or string key) or many; returns how many were stored."""
        if isinstance(ids, str):
            return int(self._core._remove_key(ids))
        if not np.isscalar(ids) and len(ids) and isinstance(next(iter(ids)), str):
            return sum(int(self._core._remove_key(k)) for k in ids)
        return self._core.remove(_ids(ids) if not np.isscalar(ids) else _id(ids))

    def remove_expired(self) -> int:
        """With ttl: removes expired documents (already hidden from searches); returns how many."""
        return self._core.remove_expired()

    @staticmethod
    def id_of(key: str) -> int:
        """The numeric id of a string key (a 64-bit hash)."""
        return _core.Collection.id_of(key)

    def remove_where(self, filter) -> int:
        return self._core.remove_where(as_filter(filter))

    def count_where(self, filter) -> int:
        return self._core.count_where(as_filter(filter))

    def set_metadata(self, id, metadata: Mapping[str, Any]) -> bool:
        """Changes only the given fields of the document with this id (or string key);
        a value of None clears a field."""
        if isinstance(id, str):
            return self._core._set_metadata_key(id, _clean(metadata) or {})
        return self._core.set_metadata(_id(id), dict(metadata))

    def set_metadata_where(self, filter, changes: Mapping[str, Any]) -> int:
        return self._core.set_metadata_where(as_filter(filter), dict(changes))

    def get(self, id) -> Optional[dict]:
        """{"id", "vector" (NumPy array or None), "text", "sparse" (dict or None), "metadata"},
        plus "key" (string ids) and "expires_at" (ttl) when set; None if absent or expired."""
        return self._core.get(self._key(id))

    # -- search
    def search(self, vector=None, text: Optional[str] = None, sparse=None, *, k: int = 10, filter=None,
               fusion: Union[str, Fusion] = "rrf", rrf_k: float = 60.0, vector_weight: float = 1.0, text_weight: float = 1.0,
               sparse_weight: float = 1.0, candidates: int = 0, ef: int = 0, with_metadata: bool = False, with_text: bool = False,
               timeout: Optional[float] = None, as_dataframe: bool = False, modifiers=None, diversity: Optional[float] = None):
        """The k best documents for any combination of a vector, text and a sparse
        vector, fused by Reciprocal Rank Fusion (or weighted scores). filter is a
        Filter or a dict. timeout (seconds) stops the search early: the hits so far are
        returned and last_search_stopped_early is set. Returns a list of dicts, or a
        DataFrame with metadata expanded into columns (as_dataframe=True).

        modifiers (opt-in) re-score the fused hits by metadata, in C++:
            {"decay": {"field": "published", "half_life_days": 30},      # recency; origin: now
             "multiply": "provenance_confidence",                         # times a field
             "boost": [({"edge_type": "verified_decision"}, 0.5)],        # + bonus on a match
             "pool": 100}                                                 # candidates re-scored
        Each of decay, multiply and boost may be one item or a list. Hits then
        carry base_score (before the modifiers) and score (after).

        diversity (opt-in, 0 to 1): maximal marginal relevance over the candidates;
        1 keeps the ranking, lower values favor variety (needs vectors)."""
        hits, stopped = self._core._search(vector, text, sparse, k, as_filter(filter), _choose(fusion, _FUSION, "fusion"), rrf_k,
                                           vector_weight, text_weight, sparse_weight, candidates, ef, with_metadata or as_dataframe,
                                           with_text, timeout, _modifiers(modifiers), diversity)
        self.last_search_stopped_early = stopped
        if not as_dataframe:
            return hits
        return _hits_dataframe(hits)

    # -- durability and maintenance
    def checkpoint(self) -> int:
        return self._core.checkpoint()

    def maintain(self, budget_ms: int = 0) -> dict:
        """Runs the maintenance that is due; returns what it did and why ("notes")."""
        return self._core.maintain(budget_ms)

    def maintenance_status(self) -> dict:
        return self._core.maintenance_status()

    def set_maintenance(self, mode: Union[str, Maintenance] = "background", **settings) -> None:
        """Changes the maintenance policy: mode, and any MaintenancePolicy field
        (interval_ms, budget_ms, checkpoint_records, upkeep_churn, ...)."""
        policy = self._core.options.maintenance
        policy.mode = _choose(mode, _MAINTENANCE, "maintenance")
        for name, value in settings.items():
            if not hasattr(policy, name):
                raise ValueError(f"unknown maintenance setting {name!r}")
            setattr(policy, name, value)
        self._core.set_maintenance(policy)

    def save(self, path: str) -> None:
        self._core.save(str(path))


def _hits_dataframe(hits: list):
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("as_dataframe=True needs pandas: pip install pandas") from e
    base = [{k: v for k, v in h.items() if k != "metadata"} for h in hits]
    extra = [c for c in ("key", "base_score", "text") if hits and c in hits[0]]
    df = pd.DataFrame(base, columns=["id", "score", "vector_rank", "text_rank", "sparse_rank", "vector_distance", "text_score",
                                     "sparse_score"] + extra)
    for col in ("vector_rank", "text_rank", "sparse_rank"):  # integers with missing values
        df[col] = df[col].astype("Int64")
    if hits and "metadata" in hits[0]:
        meta = pd.DataFrame([h.get("metadata") or {} for h in hits])
        meta.columns = [c if c not in df.columns else f"metadata_{c}" for c in meta.columns]
        df = pd.concat([df, meta], axis=1)
    return df