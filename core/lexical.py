"""Layer 1 — Lexical (TF-IDF).

For a term ``t`` in dump ``d`` drawn from a corpus of ``N`` dumps::

    TF(t, d)    = count(t in d) / total_terms(d)
    IDF(t)      = log((1 + N) / (1 + df(t))) + 1     df(t) = #dumps with t
    TFIDF(t, d) = TF(t, d) * IDF(t)

We use **smoothed** IDF (the ``+1`` smoothing borrowed from scikit-learn)
rather than the raw ``log(N / df)``. Raw IDF is exactly 0 whenever a term
appears in every dump — and on your *first* dump every term appears in every
(i.e. the one) dump, so every score collapses to 0 and ranking is impossible.
Smoothing keeps IDF strictly positive: with a single dump it degenerates to a
constant, so TF-IDF cleanly falls back to ranking by term frequency, and as
the corpus grows, common terms are still down-weighted relative to rare ones.

A *global* IDF table is persisted across all dumps (``_system/global_idf.json``)
so a brand-new dump is scored against the entire corpus, not just itself.
The per-dump output is the top-20 keywords by TF-IDF, stored as JSON.
"""

from __future__ import annotations

import math
from collections import Counter

from core.storage import GLOBAL_IDF_PATH, Storage, keywords_path

TOP_K = 20


def term_frequencies(tokens: list[str]) -> dict[str, float]:
    """TF(t, d) = count(t) / total terms in d."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {t: c / total for t, c in counts.items()}


def load_global_idf(storage: Storage) -> dict:
    """Return the persisted ``{"N": int, "df": {term: count}}`` table."""
    return storage.read_json(GLOBAL_IDF_PATH, default={"N": 0, "df": {}})


def update_global_idf(storage: Storage, dump_id: str, tokens: list[str]) -> dict:
    """Fold a new dump's vocabulary into the global df table.

    Document frequency counts each term once per dump. Re-ingesting the same
    dump id is idempotent because we track which dumps have been counted.
    """
    table = load_global_idf(storage)
    counted = set(table.get("counted_dumps", []))
    if dump_id in counted:
        return table
    table["N"] = table.get("N", 0) + 1
    df = table.setdefault("df", {})
    for term in set(tokens):
        df[term] = df.get(term, 0) + 1
    counted.add(dump_id)
    table["counted_dumps"] = sorted(counted)
    storage.write_json(GLOBAL_IDF_PATH, table, message=f"IDF update {dump_id}")
    return table


def idf_for(table: dict, term: str) -> float:
    """Smoothed IDF: log((1 + N) / (1 + df(t))) + 1.

    Always strictly positive, so scores never collapse to zero (the bug with
    raw ``log(N/df)`` on a single dump). With an empty/one-dump corpus this is
    a constant, making TF-IDF degrade gracefully to term-frequency ranking.
    """
    n = table.get("N", 0)
    df = table.get("df", {}).get(term, 0)
    return math.log((1 + n) / (1 + df)) + 1.0


def compute_tfidf(tokens: list[str], idf_table: dict) -> dict[str, float]:
    """Return TF-IDF scores for every distinct term in ``tokens``."""
    tf = term_frequencies(tokens)
    return {t: weight * idf_for(idf_table, t) for t, weight in tf.items()}


def top_keywords(scores: dict[str, float], k: int = TOP_K) -> list[dict]:
    """Top-k terms as ``[{"term": .., "score": ..}, ...]`` (desc)."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"term": t, "score": round(s, 6)} for t, s in ranked]


def process_lexical(storage: Storage, dump_id: str, tokens: list[str]) -> dict:
    """Update the global IDF table and write this dump's TF-IDF output.

    Returns the stored payload, including the top-20 keywords.
    """
    table = update_global_idf(storage, dump_id, tokens)
    scores = compute_tfidf(tokens, table)
    payload = {
        "id": dump_id,
        "n_tokens": len(tokens),
        "n_unique_terms": len(set(tokens)),
        "tfidf": {t: round(s, 6) for t, s in scores.items()},
        "top_keywords": top_keywords(scores),
    }
    storage.write_json(keywords_path(dump_id), payload, message=f"Lexical {dump_id}")
    return payload


def load_lexical(storage: Storage, dump_id: str) -> dict | None:
    return storage.read_json(keywords_path(dump_id))
