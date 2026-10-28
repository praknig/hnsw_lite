"""hnsw_lite_lib.hnswlib: hnswlib's API, backed by hnsw_lite.

The first tests always run: behaviour with the expectations established by running the
same code against hnswlib 0.8.0. The differential tests at the end run that comparison
again when hnswlib is installed (pip install hnswlib), and are skipped otherwise.
"""

import numpy as np
import pytest

from hnsw_lite_lib import hnswlib as ours

RNG = np.random.default_rng(0)
DATA = RNG.normal(size=(1500, 12)).astype(np.float32)
QUERIES = RNG.normal(size=(30, 12)).astype(np.float32)


def build(lib, space, cls="Index", n=1500):
    if cls == "Index":
        p = lib.Index(space=space, dim=12)
        p.init_index(max_elements=n, ef_construction=200, M=16, random_seed=100)
        p.add_items(DATA[:n], np.arange(n))
        p.set_ef(100)
    else:
        p = lib.BFIndex(space=space, dim=12)
        p.init_index(n)
        p.add_items(DATA[:n], np.arange(n))
    return p


# ---------------------------------------------------------------- always run


@pytest.mark.parametrize("space", ["l2", "ip", "cosine"])
def test_distances_follow_hnswlib_conventions(space):
    p = build(ours, space, "BFIndex", 200)
    labels, dists = p.knn_query(QUERIES[:5], k=4)
    assert labels.shape == (5, 4) and labels.dtype == np.uint64 and dists.dtype == np.float32
    for row, q in enumerate(QUERIES[:5]):
        x = DATA[labels[row]].astype(np.float64)
        if space == "l2":
            expected = ((x - q) ** 2).sum(axis=1)                       # squared
        elif space == "ip":
            expected = 1.0 - x @ q                                       # 1 - dot
        else:
            expected = 1.0 - (x / np.linalg.norm(x, axis=1, keepdims=True)) @ q  # hnswlib's BFIndex: query not normalized
        np.testing.assert_allclose(dists[row], expected, rtol=1e-4, atol=1e-4)
    if space == "cosine":  # Index normalizes both: 1 - cosine
        i = build(ours, space, "Index", 200)
        labels, dists = i.knn_query(QUERIES[0], k=3)
        x = DATA[labels[0]].astype(np.float64)
        cos = (x @ QUERIES[0]) / (np.linalg.norm(x, axis=1) * np.linalg.norm(QUERIES[0]))
        np.testing.assert_allclose(dists[0], 1.0 - cos, rtol=1e-4, atol=1e-5)


def test_index_lifecycle():
    p = ours.Index(space="l2", dim=4)
    p.init_index(max_elements=4, M=8, ef_construction=50)
    eye = np.eye(4, dtype=np.float32)
    p.add_items(eye[:3])                                   # labels 0, 1, 2 (sequential)
    assert p.get_ids_list() == [0, 1, 2] and p.get_current_count() == 3 and p.element_count == 3
    with pytest.raises(RuntimeError):
        p.add_items(np.ones((2, 4), np.float32))           # 3 + 2 > 4
    p.add_items(eye[3] * 5, ids=[2])                       # an existing label: replaced
    np.testing.assert_allclose(p.get_items([2])[0], eye[3] * 5)
    p.mark_deleted(1)
    assert 1 not in p.knn_query(eye[1], k=2)[0][0].tolist() and p.get_current_count() == 3
    with pytest.raises(RuntimeError):
        p.mark_deleted(1)                                  # already deleted
    with pytest.raises(RuntimeError):
        p.mark_deleted(99)                                 # unknown
    np.testing.assert_allclose(p.get_items([1])[0], eye[1])  # still readable, as in hnswlib
    p.unmark_deleted(1)
    assert int(p.knn_query(eye[1], k=1)[0][0][0]) == 1
    with pytest.raises(RuntimeError):
        p.unmark_deleted(1)                                # not deleted
    with pytest.raises(RuntimeError):
        p.knn_query(eye[0], k=10)                          # fewer than k
    with pytest.raises(RuntimeError):
        p.resize_index(2)                                  # below the count
    p.resize_index(10)
    assert p.get_max_elements() == 10 == p.max_elements
    assert p.knn_query(eye[0], k=0)[0].shape == (1, 0)
    assert p.get_items(return_type="list")[0] == [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(RuntimeError):
        p.get_items([42])
    with pytest.raises(RuntimeError):
        p.add_items(np.ones((1, 5), np.float32))           # wrong dimension
    with pytest.raises(RuntimeError):
        ours.Index(space="manhattan", dim=4)


def test_filters_and_threads():
    p = build(ours, "l2", "Index", 500)
    labels, _ = p.knn_query(QUERIES[:3], k=5, num_threads=2, filter=lambda label: label % 2 == 0)
    assert labels.shape == (3, 5) and all(int(x) % 2 == 0 for x in labels.ravel())
    b = build(ours, "l2", "BFIndex", 500)
    exact, _ = b.knn_query(QUERIES[:3], k=5, filter=lambda label: label % 3 == 0)
    assert all(int(x) % 3 == 0 for x in exact.ravel())
    with pytest.raises(RuntimeError):
        b.knn_query(QUERIES[0], k=5, filter=lambda label: label == 7)  # one match for five


def test_save_and_load(tmp_path):
    p = build(ours, "cosine", "Index", 300)
    p.mark_deleted(5)
    path = str(tmp_path / "index.bin")
    p.save_index(path)
    r = ours.Index(space="cosine", dim=12)
    r.load_index(path, max_elements=400)
    assert r.get_ids_list() == p.get_ids_list() and r.get_max_elements() == 400 and r.ef == p.ef
    np.testing.assert_array_equal(r.knn_query(QUERIES, k=5)[0], p.knn_query(QUERIES, k=5)[0])
    r.unmark_deleted(5)                                    # the deleted vector was saved too
    with pytest.raises(RuntimeError):
        ours.Index(space="l2", dim=12).load_index(path)    # another space
    with pytest.raises(RuntimeError):
        ours.Index(space="cosine", dim=8).load_index(path) # another dimension
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not an index")
    with pytest.raises(RuntimeError):
        ours.Index(space="cosine", dim=12).load_index(str(junk))
    b = build(ours, "ip", "BFIndex", 100)
    b.delete_vector(3)
    b.save_index(path)
    c = ours.BFIndex(space="ip", dim=12)
    c.load_index(path)
    assert c.get_current_count() == 99
    np.testing.assert_array_equal(c.knn_query(QUERIES, k=4)[0], b.knn_query(QUERIES, k=4)[0])
    with pytest.raises(RuntimeError):
        c.delete_vector(3)


# ---------------------------------------------------------------- against hnswlib


@pytest.mark.parametrize("space", ["l2", "ip", "cosine"])
def test_same_results_as_hnswlib(space):
    real = pytest.importorskip("hnswlib")
    ri, oi = build(real, space), build(ours, space)
    rl, rd = ri.knn_query(QUERIES, k=10)
    ol, od = oi.knn_query(QUERIES, k=10)
    overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(rl, ol)])
    assert overlap >= 0.98                                 # approximate: graphs differ, results barely
    for a, b, da, db in zip(rl, ol, rd, od):
        if np.array_equal(a, b):
            np.testing.assert_allclose(da, db, rtol=1e-4, atol=1e-5)
    rb, ob = build(real, space, "BFIndex"), build(ours, space, "BFIndex")
    (rl, rd), (ol, od) = rb.knn_query(QUERIES, k=10), ob.knn_query(QUERIES, k=10)
    np.testing.assert_array_equal(rl, ol)                  # exact: identical
    np.testing.assert_allclose(rd, od, rtol=1e-4, atol=1e-5)
    keep = lambda label: label % 4 == 1
    np.testing.assert_array_equal(rb.knn_query(QUERIES, k=5, filter=keep)[0], ob.knn_query(QUERIES, k=5, filter=keep)[0])


def test_same_lifecycle_as_hnswlib():
    real = pytest.importorskip("hnswlib")
    eye = np.eye(4, dtype=np.float32)
    seen = {}
    for name, lib in (("real", real), ("ours", ours)):
        p = lib.Index(space="l2", dim=4)
        p.init_index(max_elements=4)
        p.add_items(eye[:3])
        p.add_items(eye[3] * 2, ids=[0])
        p.mark_deleted(2)
        result = [sorted(p.get_ids_list()), p.get_current_count(), p.get_items([0]).tolist(),
                  p.knn_query(eye[1], k=2)[0].tolist()]
        for call in (lambda: p.add_items(np.ones((2, 4), np.float32)), lambda: p.mark_deleted(2), lambda: p.knn_query(eye[0], k=4)):
            try:
                call()
                result.append("no error")
            except RuntimeError:
                result.append("RuntimeError")
        p.unmark_deleted(2)
        result.append(p.knn_query(eye[2], k=1)[0].tolist())
        seen[name] = result
    assert seen["real"] == seen["ours"]


def test_replace_deleted_as_hnswlib():
    real = pytest.importorskip("hnswlib")
    eye = np.eye(4, dtype=np.float32)
    seen = {}
    for name, lib in (("real", real), ("ours", ours)):
        p = lib.Index(space="l2", dim=4)
        p.init_index(max_elements=4, allow_replace_deleted=True)
        p.add_items(eye, [0, 1, 2, 3])
        p.mark_deleted(1)
        p.mark_deleted(2)
        p.add_items(eye[:2] * 3, [10, 11], replace_deleted=True)   # both take deleted places: still 4
        result = [p.get_current_count(), sorted(p.get_ids_list())]
        try:
            p.add_items(eye[:1], [12], replace_deleted=True)        # none left to replace, and full
            result.append("no error")
        except RuntimeError:
            result.append("RuntimeError")
        q = lib.Index(space="l2", dim=4)
        q.init_index(max_elements=4)
        try:
            q.add_items(eye[:1], [0], replace_deleted=True)         # not allowed by init_index
            result.append("no error")
        except RuntimeError:
            result.append("RuntimeError")
        seen[name] = result
    assert seen["real"] == seen["ours"] == [4, [0, 3, 10, 11], "RuntimeError", "RuntimeError"]
