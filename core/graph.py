"""Layer 3 — Graph (positive PMI concept graph).

Pointwise mutual information measures how much more often two concepts
co-occur than chance would predict::

    PMI(x, y) = log( P(x, y) / (P(x) · P(y)) )

Probabilities are estimated over a sliding window of ``WINDOW`` words:

* ``P(x)``     = (#windows containing x) / (#windows)
* ``P(x, y)``  = (#windows containing both x and y) / (#windows)

Only *positive* PMI is kept (negatives clipped to 0): we care about
concepts that attract, not ones that merely avoid each other. Nodes are
concepts/keywords and edge weights are PMI scores.

Two graphs are produced and stored as node-link JSON:

* a *per-dump* graph capturing the structure of a single dump, and
* a *global* graph accumulated across the whole corpus.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import networkx as nx

from core.storage import GLOBAL_GRAPH_PATH, Storage, dump_graph_path

WINDOW = 5


def _cooccurrence_counts(tokens: list[str], window: int = WINDOW):
    """Count single- and pair-window occurrences over a sliding window.

    A term/pair is counted once per window in which it appears (presence,
    not multiplicity) so a repeated word inside one window does not inflate
    its probability.
    """
    single = Counter()
    pair = Counter()
    n_windows = 0
    if len(tokens) < 2:
        return single, pair, n_windows

    # Sliding windows of fixed size; the tail yields progressively shorter
    # windows so short inputs still produce co-occurrences.
    for start in range(max(1, len(tokens) - window + 1)):
        win = set(tokens[start:start + window])
        if len(win) < 1:
            continue
        n_windows += 1
        for term in win:
            single[term] += 1
        for a, b in combinations(sorted(win), 2):
            pair[(a, b)] += 1
    return single, pair, n_windows


def build_pmi_graph(tokens: list[str], window: int = WINDOW,
                    min_count: int = 1) -> nx.Graph:
    """Build a positive-PMI graph from a token stream."""
    single, pair, n_windows = _cooccurrence_counts(tokens, window)
    g = nx.Graph()
    if n_windows == 0:
        return g

    for term, count in single.items():
        if count >= min_count:
            g.add_node(term, count=count)

    for (a, b), c_xy in pair.items():
        if c_xy < min_count or a not in g or b not in g:
            continue
        p_xy = c_xy / n_windows
        p_x = single[a] / n_windows
        p_y = single[b] / n_windows
        pmi = math.log(p_xy / (p_x * p_y))
        if pmi > 0:  # positive PMI only
            g.add_edge(a, b, weight=round(pmi, 6), cooccur=c_xy)

    # Drop isolated nodes so the graph reflects actual relationships.
    g.remove_nodes_from(list(nx.isolates(g)))
    return g


def _serialise(g: nx.Graph) -> dict:
    return nx.node_link_data(g, edges="links")


def _deserialise(data: dict) -> nx.Graph:
    return nx.node_link_graph(data, edges="links")


def process_graph(storage: Storage, dump_id: str, tokens: list[str]) -> dict:
    """Build/store the per-dump graph and fold tokens into the global graph."""
    per_dump = build_pmi_graph(tokens)
    storage.write_json(dump_graph_path(dump_id), _serialise(per_dump),
                       message=f"Graph {dump_id}")

    # Rebuild the global graph from the union of all dump tokens would be
    # ideal, but we incrementally merge to keep cost bounded: accumulate raw
    # co-occurrence by re-deriving from this dump and unioning edges, taking
    # the max PMI seen. For correctness across the corpus we instead store the
    # global graph as the union of per-dump graphs' strongest edges.
    global_data = storage.read_json(GLOBAL_GRAPH_PATH)
    g_global = _deserialise(global_data) if global_data else nx.Graph()
    for node, attrs in per_dump.nodes(data=True):
        if g_global.has_node(node):
            g_global.nodes[node]["count"] = (
                g_global.nodes[node].get("count", 0) + attrs.get("count", 0)
            )
        else:
            g_global.add_node(node, count=attrs.get("count", 0))
    for a, b, attrs in per_dump.edges(data=True):
        w = attrs.get("weight", 0.0)
        if g_global.has_edge(a, b):
            e = g_global[a][b]
            e["weight"] = round(max(e.get("weight", 0.0), w), 6)
            e["cooccur"] = e.get("cooccur", 0) + attrs.get("cooccur", 0)
        else:
            g_global.add_edge(a, b, weight=w, cooccur=attrs.get("cooccur", 0))
    storage.write_json(GLOBAL_GRAPH_PATH, _serialise(g_global),
                       message="Global graph update")

    return {
        "nodes": per_dump.number_of_nodes(),
        "edges": per_dump.number_of_edges(),
        "top_edges": top_edges(per_dump),
    }


def top_edges(g: nx.Graph, k: int = 20) -> list[dict]:
    """Return the strongest PMI edges as a ranked list."""
    edges = sorted(
        (
            {"source": a, "target": b, "weight": d["weight"]}
            for a, b, d in g.edges(data=True)
        ),
        key=lambda e: e["weight"],
        reverse=True,
    )
    return edges[:k]


def load_graph(storage: Storage, dump_id: str) -> nx.Graph | None:
    data = storage.read_json(dump_graph_path(dump_id))
    return _deserialise(data) if data else None


def load_global_graph(storage: Storage) -> nx.Graph | None:
    data = storage.read_json(GLOBAL_GRAPH_PATH)
    return _deserialise(data) if data else None


def neighbors(g: nx.Graph, terms: list[str], top_n: int = 15) -> list[dict]:
    """Return the strongest PMI neighbours of any of ``terms`` in ``g``."""
    out = []
    seen = set()
    for term in terms:
        if term not in g:
            continue
        for nbr in g.neighbors(term):
            key = tuple(sorted((term, nbr)))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {"source": term, "target": nbr, "weight": g[term][nbr]["weight"]}
            )
    out.sort(key=lambda e: e["weight"], reverse=True)
    return out[:top_n]
