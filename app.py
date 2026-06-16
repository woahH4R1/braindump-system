"""BrainDump Signal System — Streamlit front end.

A small "text lab" for your own thoughts. Three views:

* Input   — paste raw text, pick preprocessing steps and watch them happen,
            run individual transformations (TF-IDF, LSA), then save to corpus.
* Explore — pick a past dump and inspect its layer outputs.
* Query   — type a concept and find related dumps (LSA cosine).

Design: dark, monospace data, minimal chrome. A tool, not a product.
"""

from __future__ import annotations

import streamlit as st

from core import lexical, raw, semantic
from core.ingest import PreprocessOptions, make_title, preprocess, tokenize
from core.pipeline import process_dump
from core.storage import Storage

st.set_page_config(page_title="BrainDump Signal System", page_icon="🧠", layout="wide")

# --- styling: dark, monospace data, minimal -------------------------------
st.markdown(
    """
    <style>
      .stApp { background-color: #0b0c0e; color: #cfd2d6; }
      section[data-testid="stSidebar"] { background-color: #101216; border-right: 1px solid #1d2127; }
      h1, h2, h3, h4 { color: #f0f2f4; font-family: ui-monospace, "JetBrains Mono", monospace; letter-spacing: -0.02em; }
      h1 { font-size: 1.7rem; }
      h2 { font-size: 1.25rem; margin-top: 0.4rem; }
      h3 { font-size: 1.05rem; }
      .mono, .mono * { font-family: ui-monospace, "JetBrains Mono", monospace !important; }
      code, pre, .stCode { font-family: ui-monospace, monospace !important; }
      .stApp a { color: #7fd1c1; }
      div[data-testid="stMetric"] { background: #13161b; border: 1px solid #1f242c; border-radius: 10px; padding: 8px 12px; }
      div[data-testid="stMetricValue"] { font-family: ui-monospace, monospace; color: #e8eaed; }
      .stButton > button { border-radius: 8px; border: 1px solid #2a2f37; background: #161a20; color: #dfe2e6; font-family: ui-monospace, monospace; }
      .stButton > button:hover { border-color: #7fd1c1; color: #ffffff; }
      .stTextArea textarea, .stTextInput input { background: #0e1014; color: #dfe2e6; font-family: ui-monospace, monospace; border: 1px solid #242a32; }
      hr { border-color: #1d2127; }
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


# --- sidebar ---------------------------------------------------------------
st.sidebar.title("🧠 BrainDump")
view = st.sidebar.radio("View", ["Input", "Explore", "Query"])
st.sidebar.caption(f"storage backend: `{storage.kind}`")
st.sidebar.caption(f"dumps stored: {len(_index_dumps())}")


# ============================================================================
# INPUT  —  raw text  ->  preprocess  ->  transform  ->  save
# ============================================================================
if view == "Input":
    st.title("Input & Transform")
    st.caption(
        "Paste raw text, choose how to clean it, watch each transformation "
        "happen, then save it to your corpus."
    )

    col_text, col_opts = st.columns([3, 2])

    with col_text:
        dump_type = st.selectbox(
            "Dump type", ["brain dump", "gemini chat", "chatgpt chat", "structured notes"]
        )
        uploaded = st.file_uploader("…or upload a .txt file", type=["txt"])
        if uploaded is not None and not st.session_state.get("raw_text"):
            st.session_state["raw_text"] = uploaded.read().decode("utf-8", errors="replace")
        text = st.text_area(
            "Raw text", height=280, key="raw_text", placeholder="Dump everything here…"
        )

    with col_opts:
        st.markdown("**Preprocessing steps**")
        st.caption("Pick what to strip. Applied when you hit *Preprocess*.")
        drop_stop = st.checkbox("Remove stopwords (the, and, is…)", value=True)
        drop_fill = st.checkbox(
            "Remove filler & contractions (i'm, yeah, basically…)", value=True
        )
        min_len = st.slider("Drop words shorter than (chars)", 1, 6, 3)
        conv_only = st.checkbox("Conversations: keep only my turns", value=True)

    opts = PreprocessOptions(
        drop_stopwords=drop_stop,
        drop_filler=drop_fill,
        min_len=min_len,
        conversation_user_only=conv_only,
    )

    if st.button("⚙️  Preprocess", type="primary"):
        if not text.strip():
            st.warning("Nothing to preprocess.")
            st.session_state.pop("pre", None)
        else:
            st.session_state["pre"] = preprocess(text, opts)
            st.session_state["pre_type"] = dump_type
            # transformations are recomputed against the new tokens
            st.session_state.pop("tfidf", None)
            st.session_state.pop("lsa", None)

    pre = st.session_state.get("pre")
    if pre:
        st.divider()

        # -- ① cleaned text -------------------------------------------------
        st.subheader("① Cleaned text")
        if pre["is_conversation"] and conv_only:
            st.caption("Conversation detected — only your turns were kept for analysis.")
        m = st.columns(4)
        m[0].metric("raw words", pre["n_raw"])
        m[1].metric("kept", pre["n_kept"])
        m[2].metric("removed", pre["n_removed"])
        m[3].metric("unique terms", pre["n_unique"])
        with st.expander("what got removed"):
            st.write(f"- stopwords / contractions: **{pre['removed_stop']}**")
            st.write(f"- filler words: **{pre['removed_filler']}**")
            st.write(f"- too short (< {min_len} chars): **{pre['removed_short']}**")
        st.caption("cleaned token stream (first 120)")
        preview = " ".join(pre["tokens"][:120]) + (" …" if pre["n_kept"] > 120 else "")
        st.code(preview or "(nothing left after cleaning)", language="text")
        if pre["top_freq"]:
            st.caption("most frequent terms")
            st.table([{"term": t, "count": c} for t, c in pre["top_freq"][:15]])

        # -- ② transformations ---------------------------------------------
        st.subheader("② Transformations")
        st.caption("Apply a layer to the cleaned tokens. Each is explicit math — no model.")
        t_left, t_right = st.columns(2)

        with t_left:
            st.markdown("**TF-IDF · Layer 1** — what is this dump mostly about?")
            st.caption(
                "Scores a word high when it's frequent here but rare across your "
                "corpus. Smoothed, so it gives real numbers even on your first dump."
            )
            if st.button("Run TF-IDF", use_container_width=True):
                table = lexical.load_global_idf(storage)
                scores = lexical.compute_tfidf(pre["tokens"], table)
                st.session_state["tfidf"] = lexical.top_keywords(scores, k=20)
            if st.session_state.get("tfidf"):
                st.table(st.session_state["tfidf"])

        with t_right:
            st.markdown("**LSA / SVD · Layer 2** — what else is related, by meaning?")
            st.caption(
                "Factorises the term–document matrix into a concept space and "
                "ranks dumps by cosine similarity. Needs ≥ 2 dumps in the corpus."
            )
            n_corpus = len(_index_dumps())
            run_lsa = st.button("Run LSA", use_container_width=True, disabled=n_corpus < 2)
            if run_lsa:
                st.session_state["lsa"] = semantic.search(storage, pre["tokens"], top_n=10)
            if n_corpus < 2:
                st.info(f"You have {n_corpus} dump(s). Save another to switch LSA on.")
            elif st.session_state.get("lsa") is not None:
                if st.session_state["lsa"]:
                    st.table(st.session_state["lsa"])
                else:
                    st.caption("No semantic matches (no overlapping vocabulary yet).")

        # -- ③ save ---------------------------------------------------------
        st.subheader("③ Save to corpus")
        st.caption(
            "Stores the raw text verbatim and runs the full pipeline so this dump "
            "joins your corpus, Explore and Query."
        )
        preview_title = make_title(pre["tokens"])
        st.caption(f"proposed title: **{preview_title}**")
        if st.button("💾  Save to corpus", type="primary"):
            with st.spinner("Running layers 0–3…"):
                result = process_dump(
                    storage, text, st.session_state.get("pre_type", dump_type), opts=opts
                )
            st.success(f"**{result['title']}** — saved as `{result['id']}`")
            c1, c2 = st.columns(2)
            c1.metric("tokens", result["lexical"]["n_tokens"])
            c2.metric("LSA dims (k)", result["semantic"].get("k", 0))
            st.subheader("Top keywords")
            st.table(result["lexical"]["top_keywords"])
            for k in ("tfidf", "lsa"):
                st.session_state.pop(k, None)


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
            f"{d.get('title', d['id'])}  ·  {d.get('type', '?')}  ·  {d['id'][:19]}": d["id"]
            for d in dumps
        }
        chosen = st.selectbox("Dump", list(labels.keys()))
        dump_id = labels[chosen]

        left, right = st.columns(2)

        with left:
            st.subheader("Layer 0 · Raw")
            text = raw.load_raw(storage, dump_id) or ""
            st.code(text[:5000] + ("…" if len(text) > 5000 else ""), language="text")

        with right:
            st.subheader("Layer 1 · Keywords (TF-IDF)")
            lex = lexical.load_lexical(storage, dump_id)
            if lex and lex.get("top_keywords"):
                st.table(lex["top_keywords"])
            else:
                st.caption("no lexical output")

            st.subheader("Layer 2 · Related dumps (LSA)")
            sims = semantic.similar_to_dump(storage, dump_id, top_n=10)
            if sims:
                st.caption("most similar dumps (cosine in latent space)")
                st.table(sims)
            else:
                st.caption("need ≥ 2 dumps for semantic similarity")


# ============================================================================
# QUERY
# ============================================================================
elif view == "Query":
    st.title("Query")
    st.caption("Find everything related to a concept. Ranking = LSA cosine similarity.")

    q = st.text_input("Concept", placeholder="e.g. attention, burnout, project ideas…")
    if q.strip():
        tokens = tokenize(q)
        st.markdown(f"<span class='mono'>query tokens: {tokens}</span>", unsafe_allow_html=True)

        st.subheader("Related dumps (LSA)")
        results = semantic.search(storage, tokens, top_n=15)
        if results:
            st.table(results)
        else:
            st.caption("no semantic match yet (need ≥ 2 dumps and overlapping vocabulary)")
