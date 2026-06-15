"""Layer 1 — Lexical (TF-IDF).

For a term ``t`` in dump ``d`` drawn from a corpus of ``N`` dumps::

    TF(t, d)    = count(t in d) / total_terms(d)
    IDF(t)      = log(N / df(t))          df(t) = #dumps containing t
    TFIDF(t, d) = TF(t, d) * IDF(t)

A *global* IDF table is persisted across all dumps (``lexical/_global_idf.json``)
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
    """IDF(t) = log(N / df(t)); 0 when the term is in every dump."""
    n = max(table.get("N", 0), 1)
    df = table.get("df", {}).get(term, 0)
    if df <= 0:
        return math.log(n)  # unseen term -> maximally specific
    return math.log(n / df)


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
