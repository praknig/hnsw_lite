"""Tests of the Python bindings: conversions, indexes, filters, collections, pandas."""
import os
import threading

import numpy as np
import pytest

import hnsw_lite_lib as hl

pd = pytest.importorskip("pandas")
rng = np.random.default_rng(1)


def brute_force(data, query, k):
    d = ((data - query) ** 2).sum(axis=1)
    return np.argsort(d, kind="stable")[:k]


# ---------------------------------------------------------------- conversions

def test_any_float_dtype_and_layout_is_accepted():
    idx = hl.HnswIndex(8, hl.Metric.L2)
    data = rng.random((40, 8))
    idx.add(np.arange(0, 10), data[:10])                           # float64
    idx.add(np.arange(10, 20), data[10:20].astype(np.float16))     # float16
    idx.add(np.arange(20, 30), np.asfortranarray(data[20:30]))     # Fortran order
    idx.add(np.arange(30, 40), data[30:40][:, ::-1][:, ::-1])      # a non-contiguous view
    idx.add(np.uint64(40), list(data[0]))                          # one vector as a list, a NumPy id
    assert len(idx) == 41
    np.testing.assert_allclose(idx.get_vector(25), data[25].astype(np.float32))


def test_shapes_and_values_are_checked():
    idx = hl.HnswIndex(4, hl.Metric.L2)
    with pytest.raises(ValueError, match="dimensions"):
        idx.add([1, 2], np.zeros((2, 5)))
    with pytest.raises(ValueError):
        idx.add([1, 2, 3], np.zeros((2, 4)))                        # ids and vectors differ
    with pytest.raises(ValueError):
        idx.add([1], np.zeros((1, 2, 4)))                           # 3-D
    with pytest.raises(ValueError):
        idx.add([1], np.array([[1, np.nan, 0, 0]]))
    with pytest.raises(ValueError):
        idx.search(np.zeros((2, 4)), k=1)                           # a 2-D query to search()
    with pytest.raises(ValueError):                                 # as NumPy reports it
        idx.add([1], [["a", "b", "c", "d"]])
    assert len(idx) == 0                                            # nothing added by failed calls


def test_ids_are_checked_never_wrapped():
    # Found while documenting: NumPy's cast to uint64 turned -3 into 2**64 - 3.
    idx = hl.HnswIndex(2, hl.Metric.L2)
    for bad in (-1, [-1, 2], np.array([-3]), np.array([5, -1], dtype=np.int32)):
        with pytest.raises(ValueError, match="non-negative"):
            idx.add(bad, np.zeros((np.atleast_1d(bad).size, 2)))
    for bad in (1.5, np.array([1.0, 2.0]), ["a"], True):
        with pytest.raises(TypeError):
            idx.add(bad, np.zeros((np.atleast_1d(np.asarray(bad, dtype=object)).size, 2)))
    with pytest.raises(ValueError):
        idx.remove(np.array([-1]))
    assert len(idx) == 0
    big = np.array([2**63 + 5], dtype=np.uint64)                  # the full unsigned range works
    idx.add(big, np.zeros((1, 2)))
    assert int(big[0]) in idx and idx.search([0, 0], k=1)[0][0] == big[0]
    c = hl.Collection(dim=2, metric="l2", maintenance="manual")
    for call in (lambda: c.add(-1, vector=[0, 0]), lambda: c.upsert_many(np.array([-2]), np.zeros((1, 2))),
                 lambda: c.get(-1), lambda: c.remove([-4])):
        with pytest.raises(ValueError):
            call()
    with pytest.raises(TypeError):
        c.add(1.0, vector=[0, 0])


def test_metadata_types_round_trip():
    idx = hl.FlatIndex(2, hl.Metric.L2)
    md = {"i": 7, "f": 2.5, "b": True, "s": "word", "t": ["a", "b"], "n": np.int32(3), "g": np.float64(1.5), "bb": np.bool_(False)}
    idx.add(1, [0, 0], metadata=md)
    got = idx.get_metadata(1)
    assert got == {"i": 7, "f": 2.5, "b": True, "s": "word", "t": ["a", "b"], "n": 3, "g": 1.5, "bb": False}
    assert type(got["b"]) is bool and type(got["i"]) is int
    assert idx.set_metadata(1, {"i": None, "s": "new"}) and idx.get_metadata(1).get("i") is None and idx.get_metadata(1)["s"] == "new"
    with pytest.raises(TypeError):
        idx.set_metadata(1, {"bad": object()})
    with pytest.raises(TypeError):
        idx.set_metadata(1, {"tags": ["ok", 3]})
    with pytest.raises(ValueError):
        idx.set_metadata(1, {"f": "now a string"})                  # a field keeps its type


# ---------------------------------------------------------------- filters

def test_filters():
    idx = hl.FlatIndex(2, hl.Metric.L2)
    for i in range(100):
        idx.add(i, [i, 0], metadata={"n": i, "x": i / 10, "even": i % 2 == 0, "c": "abc"[i % 3], "t": ["t%d" % (i % 4)]})
    F = hl.Filter
    count = idx.count_where
    assert count(F.eq("n", 5)) == 1 and count(F.ne("c", "a")) == 66 and count(F.lt("n", 10)) == 10 and count(F.le("n", 10)) == 11
    assert count(F.gt("x", 9.5)) == 4 and count(F.ge("n", np.int64(95))) == 5 and count(F.between("n", 10, 19)) == 10
    assert count(F.between("x", 1.0, 1.95)) == 10 and count(F.in_("c", ["a", "b"])) == 67 and count(F.in_("n", [1, 2, 300])) == 2
    assert count(F.in_("x", [0.5, 1])) == 2 and count(F.in_("even", [True])) == 50 and count(F.in_("n", [])) == 0
    assert count(F.has_tag("t", "t1")) == 25 and count(F.has_any_tag("t", ["t1", "t2"])) == 50 and count(F.has_all_tags("t", ["t1"])) == 25
    assert count(F.exists("n")) == 100 and count(F.all()) == 100 and F.all().is_all()
    assert count(F.eq("even", True) & F.lt("n", 10)) == 5 and count(F.eq("n", 1) | F.eq("n", 2)) == 2 and count(~F.lt("n", 90)) == 10
    assert count(hl.as_filter({"c": "a", "n": [0, 3, 4]})) == 2 and hl.as_filter(None) is None
    # NumPy arrays as value lists, NumPy booleans included (a NumPy bool_ was taken for a number)
    assert count(F.in_("even", np.array([True]))) == 50 and count(F.in_("n", np.array([1, 2], dtype=np.int32))) == 2
    assert count(F.in_("c", np.array(["a"]))) == 34 and count(F.in_("x", np.array([0.5]))) == 1
    with pytest.raises(TypeError):
        F.eq("n", [1, 2])
    with pytest.raises(TypeError):
        F.between("n", True, 3)
    with pytest.raises(TypeError):
        hl.as_filter(42)


# ---------------------------------------------------------------- indexes

def test_hnsw_search_matches_brute_force():
    data = rng.random((2000, 16)).astype(np.float32)
    idx = hl.HnswIndex(16, hl.Metric.L2)
    idx.add(np.arange(2000), data, threads=2)
    recall = 0
    for q in data[:50]:
        ids, dist = idx.search(q, k=10, ef=128)
        recall += len(set(ids) & set(brute_force(data, q, 10)))
        assert np.all(np.diff(dist) >= 0) and ids.dtype == np.uint64 and dist.dtype == np.float32
    assert recall / 500 >= 0.98
    I, D = idx.search_batch(data[:20], k=10, ef=128, threads=2)
    for row, q in enumerate(data[:20]):
        np.testing.assert_array_equal(I[row], idx.search(q, k=10, ef=128)[0])


def test_flat_is_exact_and_pads_missing_results():
    data = rng.random((300, 8)).astype(np.float32)
    idx = hl.FlatIndex(8, hl.Metric.L2)
    idx.add(np.arange(300), data, metadata=[{"g": i % 3} for i in range(300)])
    for q in data[:10]:
        np.testing.assert_array_equal(idx.search(q, k=7)[0], brute_force(data, q, 7))
    I, D = idx.search_batch(data[:2], k=150, filter=hl.Filter.eq("g", 0))    # 100 match: 50 padded
    assert I.shape == (2, 150) and np.all(I[:, 100:] == hl.MISSING_ID) and np.all(np.isinf(D[:, 100:])) and np.all(I[:, :100] % 3 == 0)


def test_changes_save_load_and_durability(tmp_path):
    data = rng.random((200, 8)).astype(np.float32)
    idx = hl.HnswIndex(8, hl.Metric.COSINE)
    idx.add(np.arange(200), data, metadata=[{"g": i % 4} for i in range(200)])
    assert idx.remove(5) == 1 and idx.remove([6, 7, 9999]) == 2 and 5 not in idx and len(idx) == 197
    assert idx.update(8, data[0]) and not idx.update(9999, data[0]) and idx.upsert(500, data[1]) and not idx.upsert(500, data[2])
    threes = idx.count_where(hl.Filter.eq("g", 3))
    assert threes > 0 and idx.remove_where(hl.Filter.eq("g", 3)) == threes and idx.count_where(hl.Filter.eq("g", 3)) == 0
    ids, _ = idx.search_by_id(10, k=5)
    assert 10 not in ids and idx.search_by_id(99999) is None
    h = idx.health(recall_sample=20)
    assert h["live"] == len(idx) and 0 <= h["recall"] <= 1
    # Removals can leave a vector unreachable (found on a CI runner whose SIMD kernel
    # built a slightly different graph): upkeep's guarantee is that none stays so.
    up = idx.upkeep()
    assert set(up) == {"nodes_repaired", "reconnected", "complete"} and up["complete"]
    assert idx.health(recall_sample=0)["unreachable"] == 0
    path = str(tmp_path / "i.idx")
    idx.save(path)
    loaded = hl.HnswIndex.load(path)
    assert len(loaded) == len(idx) and np.array_equal(loaded.search(data[3], k=5)[0], idx.search(data[3], k=5)[0])
    with open(path, "r+b") as f:
        f.seek(100)
        f.write(b"\xff\xff\xff")
    with pytest.raises(hl.FormatError):
        hl.HnswIndex.load(path)
    with pytest.raises(RuntimeError):
        hl.HnswIndex.load(str(tmp_path / "missing.idx"))
    d = hl.HnswIndex.open_durable(str(tmp_path / "d"), 8, hl.Metric.L2)
    d.add(np.arange(10), data[:10])
    assert d.durable and d.checkpoint() > 0
    del d
    assert len(hl.HnswIndex.open_durable(str(tmp_path / "d"), 8, hl.Metric.L2)) == 10


def test_searches_run_in_parallel_threads():
    data = rng.random((3000, 32)).astype(np.float32)
    idx = hl.HnswIndex(32, hl.Metric.L2)
    idx.add(np.arange(3000), data)
    expected = [idx.search(q, k=5)[0] for q in data[:40]]
    errors = []

    def worker():
        for q, want in zip(data[:40], expected):
            if not np.array_equal(idx.search(q, k=5)[0], want):
                errors.append(1)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# ---------------------------------------------------------------- collections

def make_collection(**kw):
    kw.setdefault("maintenance", "manual")
    return hl.Collection(dim=8, metric="l2", **kw)


def test_collection_parts_search_and_fusion():
    data = rng.random((300, 8)).astype(np.float32)
    c = make_collection(sparse=True)
    c.upsert_many(np.arange(300), data, texts=[f"doc {i} {'rare' if i == 42 else 'common'}" for i in range(300)],
                  sparse=[{i % 10: 1.0} for i in range(300)], metadata=[{"g": i % 3} for i in range(300)])
    assert len(c) == 300 and 42 in c
    by_text = c.search(text="rare", k=3)
    assert by_text[0]["id"] == 42 and by_text[0]["text_rank"] == 1 and by_text[0]["vector_rank"] is None
    by_vector = c.search(vector=data[5], k=3)
    assert by_vector[0]["id"] == 5 and by_vector[0]["vector_distance"] == pytest.approx(0.0, abs=1e-5)
    by_sparse = c.search(sparse=([4], [1.0]), k=50)
    assert all(h["id"] % 10 == 4 for h in by_sparse)
    hybrid = c.search(vector=data[42], text="rare", k=5, filter={"g": 0}, with_metadata=True, with_text=True)
    assert hybrid[0]["id"] == 42 and hybrid[0]["metadata"] == {"g": 0} and "rare" in hybrid[0]["text"]
    # Sparse terms are checked, never wrapped around (a NumPy cast turned -1 into 4294967295).
    for bad in ({-1: 1.0}, {2**32: 1.0}, (np.array([-1]), [1.0]), (np.array([2**32], dtype=np.uint64), [1.0])):
        with pytest.raises(ValueError, match=r"\[0, 2\*\*32\)"):
            c.upsert(900, sparse=bad)
    for bad in ({1.5: 1.0}, (np.array([1.0]), [1.0])):
        with pytest.raises(TypeError):
            c.upsert(900, sparse=bad)
    assert 900 not in c and c.upsert(900, sparse=(np.array([2**32 - 1], dtype=np.uint64), [1.0])) and c.upsert(901, sparse={np.int64(3): 0.5})
    weighted = c.search(vector=data[1], text="rare", k=3, fusion="weighted", text_weight=0)
    assert weighted[0]["id"] == 1
    with pytest.raises(ValueError):
        c.search(k=3)                                     # no query part
    with pytest.raises(ValueError):
        c.search(vector=data[0], fusion="nope")
    with pytest.raises(ValueError):
        c.add(1, vector=data[1])                          # duplicate


def test_collection_dataframes():
    n = 120
    df = pd.DataFrame({
        "emb": list(rng.random((n, 8)).astype(np.float32)),
        "body": [f"text {i}" if i % 10 else None for i in range(n)],
        "lang": ["en" if i % 2 else "de" for i in range(n)],
        "year": [2020 + i % 5 for i in range(n)],
        "rating": [i / 10 if i % 4 else np.nan for i in range(n)],
        "tags": [["a", "b"] if i % 3 == 0 else ["c"] for i in range(n)],
        "flag": pd.array([True if i % 5 else pd.NA for i in range(n)], dtype="boolean"),
        "score": np.arange(n, dtype=np.float64),        # collides with the hit's score column
    }, index=pd.Index(np.arange(1000, 1000 + n), name="doc_id"))
    c = make_collection()
    assert c.upsert_dataframe(df, vector_column="emb", text_column="body") == n
    r = c.get(1004)                                       # rating NaN and flag NA left out
    assert r["metadata"] == {"lang": "de", "year": 2024, "tags": ["c"], "flag": True, "score": 4.0} and r["text"] == "text 4"
    assert "flag" not in c.get(1005)["metadata"] and "rating" not in c.get(1004)["metadata"]  # pd.NA and NaN left out
    assert c.get(1000)["text"] == "" and c.get(1003)["metadata"]["tags"] == ["a", "b"] and c.get(1001)["metadata"]["flag"] is True
    out = c.search(vector=df["emb"].iloc[7], text="text", k=5, filter={"lang": "en"}, as_dataframe=True)
    assert list(out.columns[:8]) == ["id", "score", "vector_rank", "text_rank", "sparse_rank", "vector_distance", "text_score", "sparse_score"]
    assert "metadata_score" in out.columns and "lang" in out.columns and (out["lang"] == "en").all()
    assert out["vector_rank"].dtype == "Int64" and out["id"].iloc[0] == 1007
    empty = c.search(vector=df["emb"].iloc[0], k=5, filter={"lang": "fr"}, as_dataframe=True)
    assert len(empty) == 0 and "id" in empty.columns
    assert c.upsert_dataframe(df.iloc[:0], vector_column="emb") == 0


def test_collection_lifetime_durability_and_maintenance(tmp_path):
    path = str(tmp_path / "c.db")
    data = rng.random((50, 8)).astype(np.float32)
    with make_collection(path=path) as c:
        c.upsert_many(np.arange(50), data)
        with pytest.raises(RuntimeError):
            make_collection(path=path)                      # locked while open
        assert c.durable and c.remove_where({"x": 1}) == 0 and c.set_metadata_where({"x": 1}, {"y": 2}) == 0
        assert c.set_metadata(3, {"note": "hi"}) and c.get(3)["metadata"] == {"note": "hi"}
        st = c.maintenance_status()
        assert st["log_records"] > 0
        report = c.maintain()
        assert isinstance(report["notes"], list) and report["error"] == ""
        c.set_maintenance("manual", checkpoint_records=10)
        assert c.maintain()["checkpointed"]
        c.set_maintenance("background", interval_ms=5)
        with pytest.raises(ValueError):
            c.set_maintenance("manual", nonsense=1)
    with pytest.raises(ValueError):
        len(c)                                              # closed
    assert repr(c) == "<hnsw_lite_lib.Collection (closed)>"
    with make_collection(path=path) as again:              # the lock was released on close
        assert len(again) == 50 and again.get(3)["metadata"] == {"note": "hi"}
        again.save(str(tmp_path / "copy.col"))
    copy = hl.Collection.load(str(tmp_path / "copy.col"))
    assert len(copy) == 50 and not copy.durable
    with pytest.raises(ValueError):
        hl.Collection(dim=8, metric="nope")


def test_collection_timeout_and_presets():
    c = hl.Collection(dim=8, preset="low_memory", maintenance="manual")
    assert c.options.hnsw.quantization.codec == hl.Codec.SQ8 and c.options.ef == 128
    data = rng.random((500, 8)).astype(np.float32)
    c.upsert_many(np.arange(500), data, texts=["word"] * 500)
    c.search(vector=data[0], text="word", k=5, timeout=-1.0)  # already expired
    assert c.last_search_stopped_early
    c.search(vector=data[0], k=5)
    assert not c.last_search_stopped_early
    text_only = hl.Collection(dim=0, maintenance="manual", schema={"lang": "keyword"}, strict=True)
    text_only.add(1, text="hello world", metadata={"lang": "en"})
    with pytest.raises(ValueError):
        text_only.add(2, text="x", metadata={"other": 1})     # strict schema
    with pytest.raises(ValueError):
        text_only.add(3, vector=[1.0])                         # no dense side
    assert text_only.search(text="hello")[0]["id"] == 1


# ---------------------------------------------------------------- scoring and dates

def test_dates_in_metadata_and_filters():
    import datetime as dt
    assert hl.to_ms("2024-05-17T09:30:00+02:00") == hl.to_ms(dt.datetime(2024, 5, 17, 7, 30, tzinfo=dt.timezone.utc))
    assert hl.to_ms(dt.date(2024, 5, 17)) == hl.parse_date("2024-05-17") == hl.to_ms(dt.datetime(2024, 5, 17))  # naive: UTC
    assert hl.to_ms(np.datetime64("2024-05-17T00:00:00")) == hl.parse_date("2024-05-17") == hl.to_ms(pd.Timestamp("2024-05-17"))
    assert hl.format_date(hl.parse_date("2024-05-17T09:30:15.250Z")) == "2024-05-17T09:30:15.250Z" and hl.now_ms() > hl.parse_date("2024-01-01")
    with pytest.raises(ValueError):
        hl.parse_date("2024-02-30")
    with pytest.raises(TypeError):
        hl.to_ms(True)
    c = hl.Collection(dim=2, metric="l2", maintenance="manual")
    for i in range(10):
        c.add(i, vector=[i, 0], metadata={"published": dt.date(2024, 1, 1 + i)})
    assert c.get(3)["metadata"]["published"] == hl.parse_date("2024-01-04")
    assert c.count_where({"published": dt.date(2024, 1, 5)}) == 1
    assert c.count_where(hl.Filter.between("published", hl.to_ms("2024-01-03"), hl.to_ms(dt.date(2024, 1, 6)))) == 4


def test_score_modifiers():
    day = 86_400_000
    now = hl.parse_date("2024-06-01")
    rng2 = np.random.default_rng(5)
    data = rng2.random((300, 8)).astype(np.float32)
    idx = hl.HnswIndex(8, hl.Metric.L2)
    idx.add(np.arange(300), data, metadata=[{"t": now - i * day, "conf": (i % 10) / 10, "kind": "gold" if i == 250 else "x"} for i in range(300)])
    ids, dist = idx.search(data[0], k=5)
    assert 250 not in ids
    ids, dist, scores = idx.search(data[0], k=5, modifiers={"boost": ({"kind": "gold"}, 10.0), "pool": 300})
    assert ids[0] == 250 and scores[0] > 10 and np.all(np.diff(scores) <= 0)
    ids, dist, scores = idx.search(data[0], k=5, modifiers={"decay": {"field": "t", "half_life_days": 1, "origin": now}, "pool": 300})
    assert ids[0] in (0, 1, 2)  # one day halves the score: the newest win
    ids, dist, scores = idx.search(data[0], k=10, modifiers={"multiply": "conf"})
    assert np.allclose(scores, (1 / (1 + dist)) * (ids % 10) / 10, atol=1e-5)
    flat = hl.FlatIndex(8, hl.Metric.L2)
    flat.add(np.arange(300), data, metadata=[{"kind": "gold" if i == 250 else "x"} for i in range(300)])
    assert flat.search(data[0], k=3, modifiers={"boost": [{"when": {"kind": "gold"}, "bonus": 5}], "pool": 300})[0][0] == 250
    c = hl.Collection(dim=8, metric="l2", maintenance="manual")
    c.upsert_many(np.arange(300), data, texts=["word"] * 300, metadata=[{"conf": 1.0 if i == 200 else 0.1} for i in range(300)])
    hits = c.search(vector=data[0], text="word", k=5, modifiers={"multiply": {"field": "conf"}, "pool": 300})
    assert hits[0]["id"] == 200 and all(abs(h["score"] - h["base_score"] * (1.0 if h["id"] == 200 else 0.1)) < 1e-9 for h in hits)
    assert "base_score" not in c.search(vector=data[0], k=3)[0]
    table = c.search(vector=data[0], k=3, as_dataframe=True, modifiers={"multiply": "conf", "pool": 300})
    assert table["id"].iloc[0] == 200
    for bad in ({"decay": {"field": "t"}}, {"decay": {"field": "t", "half_life": 1, "half_life_days": 1}}, {"nonsense": 1},
                {"decay": {"field": "t", "half_life": 1, "shape": "cubic"}}, {"multiply": {"field": "c", "weight": 2}},
                {"decay": {"field": "t", "half_life": -1}}):
        with pytest.raises(ValueError):
            idx.search(data[0], k=3, modifiers=bad)


# ---------------------------------------------------------------- expiry, string keys, diversity

def test_expiry_keys_and_diversity(tmp_path):
    import datetime as dt
    import time
    data = rng.random((40, 8)).astype(np.float32)
    c = hl.Collection(dim=8, metric="l2", maintenance="manual", ttl=True)
    c.add(1, vector=data[1], expires_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=5))
    c.add(2, vector=data[2], expires_at="2999-01-01")
    c.add(3, vector=data[3])
    assert 1 not in c and c.get(1) is None and c.get(2)["expires_at"] == hl.parse_date("2999-01-01") and "expires_at" not in c.get(3)
    assert c.count_where(hl.Filter.all()) == 2 and len(c) == 3 and c.remove_expired() == 1 and len(c) == 2
    assert all(h["id"] != 1 for h in c.search(vector=data[1], k=5))
    short = hl.Collection(dim=8, metric="l2", maintenance="manual", default_ttl=dt.timedelta(milliseconds=40))
    short.add(1, vector=data[0])
    assert 1 in short
    time.sleep(0.08)
    assert 1 not in short
    with pytest.raises(ValueError):
        hl.Collection(dim=8, metric="l2", maintenance="manual").add(9, vector=data[0], expires_at="2999-01-01")
    with pytest.raises(ValueError):
        c.add(9, vector=data[0], metadata={"_key": "x"})
    keys = hl.Collection(dim=8, metric="l2", maintenance="manual", string_ids=True, path=str(tmp_path / "k.db"))
    keys.add("doc-a", vector=data[0], metadata={"n": 1})
    assert keys.upsert_many(["doc-b", "123e4567-e89b-12d3-a456-426614174000"], data[1:3]) == 2
    assert "doc-a" in keys and keys.get("doc-b")["key"] == "doc-b" and keys.get("doc-a")["metadata"] == {"n": 1}
    hits = keys.search(vector=data[0], k=3)
    assert hits[0]["key"] == "doc-a" and hits[0]["id"] == hl.Collection.id_of("doc-a")
    assert keys.search(vector=data[0], k=3, as_dataframe=True)["key"].iloc[0] == "doc-a"
    with pytest.raises(ValueError):
        keys.add("doc-a", vector=data[0])
    assert keys.remove("doc-b") == 1 and "doc-b" not in keys and keys.remove(["doc-a", "nope"]) == 1
    keys.close()
    again = hl.Collection(dim=8, metric="l2", maintenance="manual", string_ids=True, path=str(tmp_path / "k.db"))
    assert "123e4567-e89b-12d3-a456-426614174000" in again
    again.close()
    with pytest.raises(RuntimeError):
        hl.Collection(dim=8, metric="l2", maintenance="manual").add("x", vector=data[0])
    # Diversity: near-duplicate groups; a plain search returns one group.
    groups = []
    for g in range(10):
        for i in range(6):
            v = np.zeros(16, dtype=np.float32)
            v[0], v[1 + g] = 1.0, 0.6
            groups.append(v + rng.normal(0, 0.005, 16).astype(np.float32))
    groups = np.array(groups)
    idx = hl.HnswIndex(16, hl.Metric.COSINE)
    idx.add(np.arange(60), groups)
    plain, _ = idx.search(groups[0], k=5)
    mixed, _, _ = idx.search(groups[0], k=5, diversity=0.3)
    assert len({int(i) // 6 for i in plain}) == 1 and len({int(i) // 6 for i in mixed}) >= 3
    col = hl.Collection(dim=16, metric="cosine", maintenance="manual", text=False)
    col.upsert_many(np.arange(60), groups)
    assert len({h["id"] // 6 for h in col.search(vector=groups[0], k=5, diversity=0.3)}) >= 3
    with pytest.raises(ValueError):
        idx.search(groups[0], k=5, diversity=2)


# ---------------------------------------------------------------- every binding, and loading with runtime settings

def test_index_maintenance_bindings_and_option_classes():
    data = rng.random((1200, 8)).astype(np.float32)
    p = hl.HnswParams()
    p.M, p.ef_construction = 12, 100
    p.quantization.codec = hl.Codec.SQ8
    p.quantization.train_after = 0  # trained by hand below
    idx = hl.HnswIndex(8, hl.Metric.L2, p)
    assert idx.dim == 8 and idx.metric == hl.Metric.L2
    idx.add(np.arange(1200), data, metadata=[{"tag": ["a"] if i % 3 == 0 else ["b"], "k": "x"} for i in range(1200)])
    idx.create_payload_index("tag")
    idx.train_codec()
    before = idx.search(data[7], k=5)[0]
    assert idx.reorder() == 1200 and np.array_equal(idx.search(data[7], k=5)[0], before)  # same results after reordering
    assert idx.remove(np.arange(0, 300)) == 300
    assert idx.compact() == 300 and len(idx) == 900
    assert idx.rebuild() == 900 and idx.search(data[700], k=1)[0][0] == 700
    assert idx.search(data[7], k=5, strategy=hl.Strategy.EXACT)[0][0] == 7 if 7 in idx else True
    ivf = hl.IvfParams()
    ivf.lists, ivf.probes, ivf.train_after = 8, 8, 0
    q = hl.QuantizationParams()
    flat = hl.FlatIndex(8, hl.Metric.COSINE, quantization=q, ivf=ivf)
    assert flat.dim == 8 and flat.metric == hl.Metric.COSINE
    flat.add(np.arange(1200), data, metadata=[{"k": "x"}] * 1200)
    flat.create_payload_index("k")
    flat.train_lists()
    assert flat.search(data[3], k=1)[0][0] == 3
    with pytest.raises(RuntimeError):
        flat.train_codec()  # no codec configured
    opts = hl.CollectionOptions.preset(8, hl.Preset.HIGH_RECALL)
    pol = hl.MaintenancePolicy()
    pol.mode, pol.interval_ms = hl.Maintenance.MANUAL, 50
    opts.maintenance = pol
    dur = hl.DurableOptions()
    dur.durability = hl.Durability.SYNCED
    opts.durability = dur
    c = hl.Collection(options=opts)
    c.add(1, vector=data[1], text="hello world")
    hits = c.search(vector=data[1], text="hello", k=1, fusion=hl.Fusion.WEIGHTED)
    assert hits[0]["id"] == 1 and c.options.maintenance.mode == hl.Maintenance.MANUAL
    strict = hl.Collection(dim=8, metric="l2", maintenance="manual", schema={"n": hl.FieldType.INT}, strict=True)
    with pytest.raises(ValueError):
        strict.add(2, vector=data[2], metadata={"other": 1})
    assert issubclass(hl.Cancelled, RuntimeError) and issubclass(hl.FormatError, RuntimeError)


def test_load_with_runtime_settings_and_keyed_metadata(tmp_path):
    data = rng.random((5, 4)).astype(np.float32)
    c = hl.Collection(dim=4, metric="l2", maintenance="manual", ttl=True, string_ids=True)
    c.add("old", vector=data[0], expires_at="2000-01-01")
    c.add("kept", vector=data[1])
    assert c.set_metadata("kept", {"n": 1}) and c.get("kept")["metadata"] == {"n": 1}
    assert not c.set_metadata("old", {"n": 2}) and not c.set_metadata("absent", {"n": 3})
    path = str(tmp_path / "c.col")
    c.save(path)
    plain = hl.Collection.load(path)  # both off: the expired document shows, keys are refused
    assert hl.Collection.id_of("old") in plain
    with pytest.raises(RuntimeError):
        plain.get("kept")
    loaded = hl.Collection.load(path, ttl=True, string_ids=True)
    assert "old" not in loaded and loaded.get("kept")["key"] == "kept"
    with pytest.raises(ValueError):
        hl.Collection.load(path, default_ttl=-1)


# ---------------------------------------------------------------- threads and processes

def test_python_threads_run_in_parallel_safely():
    """The bindings release the GIL in searches and writes, so these Python threads
    really run at once in the C++ code: results must stay consistent."""
    import concurrent.futures
    import time
    data = rng.random((600, 8)).astype(np.float32)
    idx = hl.HnswIndex(8, hl.Metric.L2)
    idx.add(np.arange(300), data[:300])
    col = hl.Collection(dim=8, metric="l2", maintenance="manual")
    col.upsert_many(np.arange(300), data[:300], texts=["word"] * 300)
    stop = time.monotonic() + 1.0
    def worker(w):
        r = np.random.default_rng(w)
        n = 0
        while time.monotonic() < stop:
            i = int(r.integers(600))
            op = w % 4
            if op == 0:
                idx.upsert(i, data[i])
            elif op == 1:
                idx.remove(i)
            elif op == 2:
                ids, dist = idx.search(data[i], k=5)
                assert len(ids) == len(dist) and np.all(np.diff(dist) >= 0)
            else:
                col.upsert(i, vector=data[i], text="word")
                hits = col.search(vector=data[i], text="word", k=3)
                assert all(h["id"] < 600 for h in hits)
            n += 1
        return n
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        done = sum(ex.map(worker, range(8)))
    assert done > 0
    idx.upkeep()  # heavy concurrent churn: repair the graph before judging it, as the C++ test does
    stored = [i for i in range(600) if i in idx]
    found = sum(idx.search(data[i], k=1, ef=128)[0][0] == i for i in stored)
    assert found >= 0.99 * len(stored)  # approximate search: nearly all, not necessarily every one
    assert col.count_where(hl.Filter.all()) == len(col)


def test_durable_directory_is_locked_across_processes(tmp_path):
    """A second process cannot open a durable collection another process holds, and
    the lock dies with its holder (the OS releases flock and LockFileEx), so a
    crashed writer never leaves the directory locked."""
    import subprocess
    import sys
    path = str(tmp_path / "locked.db")
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, numpy as np, hnsw_lite_lib as hl\n"
         f"c = hl.Collection(dim=4, metric='l2', maintenance='manual', path={path!r})\n"
         "c.upsert(42, vector=np.ones(4, dtype=np.float32))\n"
         "print('ready', flush=True)\n"
         "sys.stdin.read()\n"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(Exception, match="in use"):
            hl.Collection(dim=4, metric="l2", maintenance="manual", path=path)
    finally:
        child.kill()  # a crash: SIGKILL on POSIX, TerminateProcess on Windows
        child.wait()
    # The lock died with the child. Windows releases a killed process's LockFileEx lock
    # "depending on available system resources", not always at once: retry briefly.
    import time
    deadline = time.monotonic() + 10
    while True:
        try:
            c = hl.Collection(dim=4, metric="l2", maintenance="manual", path=path)
            break
        except Exception as e:
            if "in use" not in str(e) or time.monotonic() > deadline:
                raise
            time.sleep(0.1)
    assert 42 in c  # and its acknowledged write survived
    c.close()