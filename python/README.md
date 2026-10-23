# hnsw-lite for Python

```bash
pip install hnsw-lite-lib
```

Vector search with metadata filters, hybrid (dense + BM25 keyword) search, crash-safe durability and automatic maintenance: a C++ library with Python bindings and NumPy and pandas integration.

```python
import numpy as np
import hnsw_lite_lib as hl

docs = hl.Collection(dim=384, path="docs.db")          # durable; omit path for in memory
docs.upsert(1, vector=np.random.rand(384), text="the quick brown fox", metadata={"lang": "en", "year": 2024})
hits = docs.search(vector=np.random.rand(384), text="fox", k=5, filter={"lang": "en"})

# pandas
docs.upsert_dataframe(df, id_column="id", vector_column="embedding", text_column="body")
results = docs.search(text="fox", k=10, as_dataframe=True)

# A plain vector index
index = hl.HnswIndex(dim=128, metric=hl.Metric.L2)
index.add(np.arange(10_000), np.random.rand(10_000, 128))
ids, distances = index.search_batch(np.random.rand(5, 128), k=10)   # arrays of shape (5, 10)
docs.close()
```

Wheels for Linux (x86-64, ARM64), macOS (Intel, Apple Silicon) and Windows, for CPython 3.9 to 3.13. See [docs/python.md](https://github.com/praknig/hnsw_lite/blob/master/docs/python.md) for the full guide.
