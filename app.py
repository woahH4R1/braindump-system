"""BrainDump Signal System — Streamlit front end.

Three views over the same data:

* Input   — capture a dump (text box or .txt upload) and run all layers.
* Explore — pick a past dump and inspect all four layer outputs side by side.
* Query   — type a concept and see related dumps (LSA cosine) plus the PMI
            connections around it.

Design: dark background, monospace data, minimal chrome. A tool, not a product.
"""

from __future__ import annotations

import streamlit as st

from core import graph as graph_layer
from core import lexical, raw, semantic
from core.ingest import tokenize
from core.pipeline import process_dump
from core.storage import Storage

st.set_page_config(page_title="BrainDump Signal System", page_icon="🧠", layout="wide")

# --- styling: dark, monospace data, minimal ---------------------------------
st.markdown(
    """
    <style>
      .stApp { background-color: #0d0d0d; color: #d7d7d7; }
      h1, h2, h3, h4 { color: #e8e8e8; font-family: ui-monospace, monospace; }
      .mono, .mono * { font-family: ui-monospace, "JetBrains Mono", monospace !important; }
      code, pre, .stCode { font-family: ui-monospace, monospace !important; }
      .stDataFrame, .stTable { font-family: ui-monospace, monospace; }
      div[data-testid="stMetricValue"] { font-family: ui-monospace, monospace; }
      .stApp a { color: #7fd1c1; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_storage() -> Storage:
    return Storage()


storage = get_storage()


def _index_dumps() -> list[dict]:
    return sorted(storage.load_index().get("dumps", []), key=lambda d: d["id"], reverse=True)


# --- sidebar ----------------------------------------------------------------
st.sidebar.title("🧠 BrainDump")
view = st.sidebar.radio("View", ["Input", "Explore", "Query"])
st.sidebar.caption(f"storage backend: `{storage.kind}`")
st.sidebar.caption(f"dumps stored: {len(_index_dumps())}")


# ============================================================================
# INPUT
# ============================================================================
if view == "Input":
    st.title("Input")
    st.caption("Capture a brain dump or paste a conversation. Stored verbatim, then decomposed into four layers.")

    dump_type = st.selectbox(
        "Dump type", ["brain dump", "gemini chat", "chatgpt chat", "structured notes"]
    )
    uploaded = st.file_uploader("…or upload a .txt file", type=["txt"])
    default_text = ""
    if uploaded is not None:
        default_text = uploaded.read().decode("utf-8", errors="replace")

    text = st.text_area("Text", value=default_text, height=300,
                        placeholder="Dump everything here…")

    if st.button("Submit", type="primary"):
        if not text.strip():
            st.warning("Nothing to ingest.")
        else:
            with st.spinner("Running layers 0-3…"):
                result = process_dump(storage, text, dump_type)
            st.success(f"**{result['title']}** — stored as `{result['id']}`"
                       + (" (conversation detected — only your turns were processed)"
                          if result["is_conversation"] else ""))
            c1, c2, c3 = st.columns(3)
            c1.metric("tokens", result["lexical"]["n_tokens"])
            c2.metric("graph nodes / edges",
                      f"{result['graph']['nodes']} / {result['graph']['edges']}")
            c3.metric("LSA dims (k)", result["semantic"].get("k", 0))
            st.subheader("Top keywords")
            st.table(result["lexical"]["top_keywords"])


# ============================================================================
# EXPLORE
# ============================================================================
elif view == "Explore":
    st.title("Explore")
    dumps = _index_dumps()
    if not dumps:
        st.info("No dumps yet. Add one in the Input view.")
    else:
        labels = {
            f"{d.get('title', d['id'])}  ·  {d.get('type','?')}  ·  {d['id'][:19]}": d["id"]
            for d in dumps
        }
        chosen = st.selectbox("Dump", list(labels.keys()))
        dump_id = labels[chosen]

        col0, col1 = st.columns(2)
        col2, col3 = st.columns(2)

        with col0:
            st.subheader("Layer 0 · Raw")
            text = raw.load_raw(storage, dump_id) or ""
            st.code(text[:5000] + ("…" if len(text) > 5000 else ""), language="text")

        with col1:
            st.subheader("Layer 1 · Lexical (TF-IDF)")
            lex = lexical.load_lexical(storage, dump_id)
            if lex:
                st.table(lex["top_keywords"])
            else:
                st.caption("no lexical output")

        with col2:
            st.subheader("Layer 2 · Semantic (LSA)")
            sims = semantic.similar_to_dump(storage, dump_id, top_n=10)
            if sims:
                st.caption("most similar dumps (cosine in latent space)")
                st.table(sims)
            else:
                st.caption("need ≥ 2 dumps for semantic similarity")

        with col3:
            st.subheader("Layer 3 · Graph (PMI)")
            g = graph_layer.load_graph(storage, dump_id)
            if g is not None and g.number_of_edges():
                st.caption(f"{g.number_of_nodes()} nodes / {g.number_of_edges()} edges — strongest PMI links")
                st.table(graph_layer.top_edges(g, k=20))
            else:
                st.caption("no graph edges (input too short)")


# ============================================================================
# QUERY
# ============================================================================
elif view == "Query":
    st.title("Query")
    st.caption("Find everything related to a concept. Ranking = LSA cosine similarity; connections = global PMI graph.")

    q = st.text_input("Concept", placeholder="e.g. attention, burnout, project ideas…")
    if q.strip():
        tokens = tokenize(q)
        st.markdown(f"<span class='mono'>query tokens: {tokens}</span>",
                    unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Related dumps (LSA)")
            results = semantic.search(storage, tokens, top_n=15)
            if results:
                st.table(results)
            else:
                st.caption("no semantic match yet (need ≥ 2 dumps and overlapping vocabulary)")

        with right:
            st.subheader("PMI connections")
            g = graph_layer.load_global_graph(storage)
            if g is not None:
                nbrs = graph_layer.neighbors(g, tokens, top_n=20)
                if nbrs:
                    st.table(nbrs)
                else:
                    st.caption("no PMI neighbours for these terms")
            else:
                st.caption("global graph not built yet")
