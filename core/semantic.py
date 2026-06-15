"""Layer 2 — Semantic (LSA via truncated SVD).

We build a term-document matrix ``M`` where ``M[i, j]`` is the TF-IDF score
of term ``i`` in dump ``j`` and factorise it::

    M = U Σ Vᵀ            (truncated to the top k singular values)

* ``U`` (terms × k)  — term vectors
* ``Σ`` (k,)         — singular values: the "strength" of each latent topic
* ``V`` (dumps × k)  — dump vectors in latent semantic space

A dump's coordinates are the rows of ``V``. A free-text query is *folded in*
with the same transform::

    v_query = Σ⁻¹ Uᵀ q          (q = the query's TF-IDF term vector)

Relatedness between any two vectors is their cosine similarity. This is pure
linear algebra (``scipy.sparse.linalg.svds``) — no trained model.

The factorisation is rebuilt from the persisted per-dump TF-IDF outputs
whenever the corpus changes, and the results are cached as ``.npy``/JSON.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from core.lexical import load_lexical
from core.storage import SEMANTIC_DIR, Storage

MAX_K = 50

DOC_VECTORS_PATH = f"{SEMANTIC_DIR}/doc_vectors.npy"      # V  (dumps × k)
TERM_VECTORS_PATH = f"{SEMANTIC_DIR}/term_vectors.npy"    # U  (terms × k)
SINGULAR_VALUES_PATH = f"{SEMANTIC_DIR}/singular_values.npy"  # Σ (k,)
VOCAB_PATH = f"{SEMANTIC_DIR}/vocab.json"                 # term -> row index
DUMP_IDS_PATH = f"{SEMANTIC_DIR}/dump_ids.json"           # ordered dump ids


def _build_term_doc_matrix(storage: Storage, dump_ids: list[str]):
    """Assemble the sparse TF-IDF term-document matrix and its vocabulary."""
    vocab: dict[str, int] = {}
    rows, cols, vals = [], [], []
    for j, dump_id in enumerate(dump_ids):
        payload = load_lexical(storage, dump_id)
        if not payload:
            continue
        for term, score in payload.get("tfidf", {}).items():
            i = vocab.setdefault(term, len(vocab))
            rows.append(i)
            cols.append(j)
            vals.append(float(score))
    if not vocab:
        return None, vocab
    matrix = csr_matrix(
        (vals, (rows, cols)), shape=(len(vocab), len(dump_ids)), dtype=np.float64
    )
    return matrix, vocab


def rebuild(storage: Storage, dump_ids: list[str]) -> dict:
    """Recompute and persist the LSA factorisation across all dumps.

    Returns a small status dict. SVD needs at least 2 dumps and 2 terms;
    below that the layer is a no-op and similarity falls back to lexical.
    """
    matrix, vocab = _build_term_doc_matrix(storage, dump_ids)
    n_terms = len(vocab)
    n_docs = len(dump_ids)
    if matrix is None or n_docs < 2 or n_terms < 2:
        return {"ok": False, "reason": "not enough data", "k": 0,
                "n_terms": n_terms, "n_docs": n_docs}

    k = min(MAX_K, min(matrix.shape) - 1)
    # svds returns singular values in ascending order; reverse to descending.
    u, s, vt = svds(matrix, k=k)
    order = np.argsort(s)[::-1]
    u, s, vt = u[:, order], s[order], vt[order, :]

    storage.write_npy(TERM_VECTORS_PATH, u)               # terms × k
    storage.write_npy(SINGULAR_VALUES_PATH, s)            # k
    storage.write_npy(DOC_VECTORS_PATH, vt.T)             # dumps × k
    storage.write_json(VOCAB_PATH, vocab)
    storage.write_json(DUMP_IDS_PATH, dump_ids)
    return {"ok": True, "k": int(k), "n_terms": n_terms, "n_docs": n_docs}


def _load_model(storage: Storage):
    """Load (U, Σ, V, vocab, dump_ids) or None if not yet built."""
    u = storage.read_npy(TERM_VECTORS_PATH)
    s = storage.read_npy(SINGULAR_VALUES_PATH)
    v = storage.read_npy(DOC_VECTORS_PATH)
    vocab = storage.read_json(VOCAB_PATH)
    dump_ids = storage.read_json(DUMP_IDS_PATH)
    if any(x is None for x in (u, s, v, vocab, dump_ids)):
        return None
    return u, s, v, vocab, dump_ids


def _cosine(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Cosine similarity between a vector and every row of a matrix."""
    vn = np.linalg.norm(vec)
    mn = np.linalg.norm(mat, axis=1)
    denom = vn * mn
    denom[denom == 0] = 1e-12
    return (mat @ vec) / denom


def fold_in(query_tfidf: dict[str, float], u, s, vocab) -> np.ndarray | None:
    """Project a query's TF-IDF term vector into latent space: Σ⁻¹ Uᵀ q."""
    q = np.zeros(len(vocab))
    hit = False
    for term, score in query_tfidf.items():
        idx = vocab.get(term)
        if idx is not None:
            q[idx] = score
            hit = True
    if not hit:
        return None
    return (u.T @ q) / np.where(s == 0, 1e-12, s)


def similar_to_dump(storage: Storage, dump_id: str, top_n: int = 10) -> list[dict]:
    """Rank other dumps by cosine similarity to ``dump_id`` in latent space."""
    model = _load_model(storage)
    if model is None:
        return []
    _u, _s, v, _vocab, dump_ids = model
    if dump_id not in dump_ids:
        return []
    idx = dump_ids.index(dump_id)
    sims = _cosine(v[idx], v)
    ranked = sorted(
        (
            {"id": dump_ids[j], "similarity": round(float(sims[j]), 4)}
            for j in range(len(dump_ids))
            if j != idx
        ),
        key=lambda d: d["similarity"],
        reverse=True,
    )
    return ranked[:top_n]


def search(storage: Storage, query_tokens: list[str], top_n: int = 10) -> list[dict]:
    """Rank all dumps by cosine similarity to a free-text query.

    The query is converted to a raw TF vector and folded into the same
    semantic space as the dumps before comparison.
    """
    model = _load_model(storage)
    if model is None:
        return []
    u, s, v, vocab, dump_ids = model
    from core.lexical import term_frequencies

    q_vec = fold_in(term_frequencies(query_tokens), u, s, vocab)
    if q_vec is None:
        return []
    sims = _cosine(q_vec, v)
    ranked = sorted(
        (
            {"id": dump_ids[j], "similarity": round(float(sims[j]), 4)}
            for j in range(len(dump_ids))
        ),
        key=lambda d: d["similarity"],
        reverse=True,
    )
    return ranked[:top_n]
