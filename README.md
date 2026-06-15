---
title: BrainDump Signal System
emoji: 🧠
colorFrom: gray
colorTo: gray
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# BrainDump Signal System

A personal **signal-processing pipeline for your own thoughts**. You dump raw
text — stream-of-consciousness notes, pasted AI conversations, structured
notes — and the system decomposes each dump into four progressively richer
*views* of the same data. Nothing is ever lost: the raw text is preserved
verbatim, and every other layer is a reconstructible transformation on top of
it.

This is a **tool, not a product**. No trained model, no embeddings API — just
explicit, inspectable mathematics (TF-IDF, SVD, PMI) you can read in the
source and verify by hand.

---

## What this system is

Most note tools store text and let you search it as strings. This one treats
your dumps as a small corpus and extracts *structure*:

- which words actually carry signal in each dump (vs. common filler),
- which dumps are *about the same thing* even when they share no exact words,
- which concepts *travel together* in your thinking.

Code lives in the public `braindump-system` repo (deployed to Hugging Face
Spaces). Data lives in the private `braindump-data` repo, written through the
GitHub API. The two are deliberately separated so your raw thoughts stay
private while the machinery stays open.

---

## The four layers (and why each exists)

| Layer | Name | Math | Question it answers |
|------:|------|------|---------------------|
| 0 | Raw | verbatim store | "What did I actually write?" |
| 1 | Lexical | TF-IDF | "What is *this dump* mostly about?" |
| 2 | Semantic | LSA via SVD | "What else is *related* to this, by meaning?" |
| 3 | Graph | positive PMI | "Which *concepts* co-occur / connect?" |

Each layer is strictly more abstract than the one below it, and each answers a
question the layer below cannot. Raw can't tell you what a dump is *about*;
lexical can't tell you two dumps mean the same thing in different words;
semantic gives you dump-level similarity but not concept-level structure —
that's what the PMI graph adds.

---

## Layer 0 — Raw

Store the input verbatim as `dumps/<id>/raw.txt`, where the *dump id* is the
timestamp plus a short concept slug derived from the content
(`2026-06-15_19-07-25__attention-transformers-tokens`). **Never modified,
always preserved.** The id is stable and reused by every other layer; a
human-readable `title` (e.g. "Attention Transformers Tokens") is also recorded
in `meta.json` and `index.json`. Raw files are write-once: a colliding id is nudged forward rather than
overwritten, so ground truth is never lost.

If the input is a pasted conversation, the **full transcript** is stored here —
only the *downstream* layers narrow to your own turns (see Input parsing).

---

## Layer 1 — Lexical (TF-IDF)

TF-IDF scores a term highly when it is frequent *in this dump* but rare *across
the corpus*, which is exactly the profile of a word that characterises the
dump.

For a term `t` in dump `d`, with a corpus of `N` dumps:

```
TF(t, d)    = count(t in d) / total_terms(d)
IDF(t)      = log( N / df(t) )           df(t) = number of dumps containing t
TFIDF(t, d) = TF(t, d) × IDF(t)
```

**Derivation / intuition.**
- **TF** normalises raw counts by document length so a long dump doesn't
  dominate a short one — it's the *proportion* of the dump spent on `t`.
- **IDF** is the surprise of seeing `t` at all. `df(t)/N` is the probability a
  random dump contains `t`; IDF is `−log(df(t)/N) = log(N/df(t))`, the
  information content (in nats) of that event. A term in every dump
  (`df = N`) has `IDF = log(1) = 0` — zero discriminating power. A term in one
  dump out of many has large IDF.
- Multiplying them rewards terms that are both *prominent here* and *globally
  rare*.

**Global IDF table.** `df(t)` and `N` are persisted in
`_system/global_idf.json` and updated on every new dump. This means each new
dump is scored against your **entire corpus**, not just itself, and scores
stay comparable over time. Per-dump output is the **top-20 keywords** by
TF-IDF, written to `dumps/<id>/keywords.json`.

---

## Layer 2 — Semantic (LSA via SVD)

Lexical matching fails when two dumps mean the same thing with different words
("burnout" vs. "exhausted, no energy"). **Latent Semantic Analysis** fixes
this by factoring the term-document matrix and working in a low-dimensional
*concept space*.

Build a term-document matrix `M` where `M[i, j] = TFIDF(term_i, dump_j)`, then
take its **truncated singular value decomposition**:

```
M  =  U  Σ  Vᵀ          (keep the top k = 50 singular values)

U : terms × k     term vectors
Σ : k × k         singular values  (diagonal)
V : dumps × k     dump vectors
```

**What the singular values represent.** `Σ` is diagonal with entries
`σ₁ ≥ σ₂ ≥ … ≥ σ_k ≥ 0`. Each `σ_r` is the *strength* of the r-th latent
dimension — an emergent "topic" that is a weighted blend of terms (its
direction is column `r` of `U`) and of dumps (column `r` of `V`). Large `σ`
means that latent topic explains a lot of the variance in your corpus; small
`σ` is fine detail/noise. Truncating to the top `k` keeps the dominant topics
and discards noise, which is precisely what lets *synonyms collapse together*:
words that co-occur with the same other words end up pointing in the same
latent direction.

**Dump coordinates.** Each dump is the corresponding row of `V` — a point in
`k`-dimensional semantic space.

**Similarity** between two dumps is the **cosine similarity** of their vectors:

```
cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

**Querying** ("find everything related to X") folds the query into the same
space. The query becomes a TF term vector `q` over the vocabulary, then:

```
v_query = Σ⁻¹ Uᵀ q
```

This is the same transform that produces a dump's coordinates (since
`v_j = Σ⁻¹ Uᵀ M[:, j]`), so the query lands in the dumps' coordinate system and
can be ranked against every dump by cosine similarity.

Implemented with `numpy` and `scipy.sparse.linalg.svds` — pure linear algebra,
no model. Outputs (`U`, `Σ`, `V`, vocabulary, dump order) are cached under
`_system/semantic/` as `.npy` / `.json`. `k` is automatically reduced for tiny corpora
(`k = min(50, min(M.shape) − 1)`); with fewer than two dumps the layer is a
no-op and similarity falls back to lexical overlap.

---

## Layer 3 — Graph (PMI)

LSA gives dump-level similarity. The PMI graph zooms in to **concept-level**
structure: which words genuinely *attract* each other.

```
PMI(x, y) = log( P(x, y) / ( P(x) · P(y) ) )
```

estimated over a sliding **window of 5 words**:

```
P(x)     = (# windows containing x)        / (# windows)
P(x, y)  = (# windows containing both x,y)  / (# windows)
```

**Co-occurrence window.** We slide a length-5 window across the token stream.
Within each window we record the *presence* of each term (once per window, not
per repeat) and of each unordered pair. `PMI` then compares how often `x` and
`y` actually share a window against what you'd expect if they were
independent. `PMI > 0` ⇒ they co-occur *more* than chance (a real
association); `PMI < 0` ⇒ less than chance.

**Positive PMI only.** Negative values are clipped to 0 — we keep the edges
that represent attraction and drop the rest. **Nodes** are concepts/keywords;
**edge weight** is the PMI score.

Two graphs are built with `networkx` and stored as node-link JSON:
- **per-dump** (`dumps/<id>/graph.json`) — the structure of a single dump,
- **global** (`_system/global_graph.json`) — accumulated across the whole corpus
  (union of edges, keeping the strongest PMI and summing co-occurrence).

---

## How the layers combine

The power is in using them *together* — four lenses on one dump:

- **Raw** anchors everything to what you literally wrote.
- **Lexical** gives a fast "what's this about" fingerprint per dump.
- **Semantic** connects dumps by *meaning*, surfacing related thinking you'd
  never find by keyword search.
- **Graph** reveals the *conceptual wiring* — the recurring associations in
  how you think, both within a dump and across all of them.

A single query flows through them: tokenise → fold into LSA space → rank
dumps by cosine (Layer 2) → and simultaneously light up the PMI neighbourhood
of the query terms in the global graph (Layer 3). The **Explore** view shows
all four outputs for one dump side by side; the **Query** view combines
Layers 2 and 3 for concept search.

---

## Input parsing

Input can be a **raw dump** or a **pasted conversation**. The system detects a
conversation by scanning for speaker prefixes at the start of a line:

```
You:  Me:  Human:  User:        → your turns
Gemini:  ChatGPT:  Assistant:  Claude:  Bard:  Copilot:  AI:  → assistant turns
```

When a conversation is detected the system:
1. stores the **full transcript verbatim** in Layer 0, and
2. tags every turn, then feeds **only your (user) turns** into Layers 1–3 —
   the assistant's words are not your signal.

Accepted input: the text box, or a `.txt` file upload. A dropdown labels the
dump (`brain dump`, `gemini chat`, `chatgpt chat`, `structured notes`).

---

## How to add a new dump

1. Open the **Input** view.
2. Choose a dump type, paste text (or upload a `.txt`).
3. Click **Submit**. The pipeline runs Layer 0 → 1 → 3 → 2 and updates
   `index.json`. (SVD is rebuilt last because it is global over the corpus.)

Programmatically:

```python
from core.storage import Storage
from core.pipeline import process_dump

process_dump(Storage(), "my raw thoughts about ...", dump_type="brain dump")
```

## How to query

1. Open the **Query** view and type a concept.
2. **Related dumps (LSA)** ranks every dump by cosine similarity to your query
   in semantic space.
3. **PMI connections** shows the strongest concept links around your query
   terms from the global graph.

---

## Data layout (`braindump-data`)

Organised so it's **browsable by a human**: one self-contained folder per
dump, with all machine-derived corpus state cordoned off under `_system/`.

```
braindump-data/
├── index.json                      # readable overview of every dump
├── dumps/
│   └── YYYY-MM-DD_HH-MM-SS__concept-slug/   # date + auto-title from content
│       ├── raw.txt                 # Layer 0: exactly what you typed (verbatim)
│       ├── meta.json               # title, type, timestamp, token count, conversation?
│       ├── keywords.json           # Layer 1: TF-IDF scores + top-20 keywords
│       └── graph.json              # Layer 3: this dump's PMI concept graph
└── _system/                        # machine-derived state (safe to ignore)
    ├── global_idf.json             # corpus-wide df / N for IDF
    ├── global_graph.json           # Layer 3: PMI graph across all dumps
    └── semantic/                   # Layer 2: LSA matrices (U, Σ, V) + vocab
```

These folders are created automatically on the first write — no manual
bootstrapping required.

---

## Running it

### On Hugging Face Spaces (production)

The Space is connected to `braindump-system`. Add these **secrets**:

| Secret | Value |
|--------|-------|
| `GITHUB_TOKEN` | a personal access token with **repo** read/write scope |
| `GITHUB_DATA_REPO` | `woahH4R1/braindump-data` |
| `GITHUB_DATA_BRANCH` | *(optional)* defaults to `main` |

The token is read from the environment and **never hardcoded**. With it set,
all data is read/written to `braindump-data` via the GitHub API (`PyGithub`).

### Locally (development / testing)

With **no** `GITHUB_TOKEN`, storage transparently falls back to a local mirror
under `_local_data/` (git-ignored) so you can run the whole app with zero
secrets:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project structure (`braindump-system`)

```
braindump-system/
├── app.py                  # Streamlit app, entry point (3 views)
├── requirements.txt
├── README.md               # this file
├── core/
│   ├── ingest.py           # input parsing, tokenisation, dump vs conversation
│   ├── raw.py              # Layer 0: verbatim storage
│   ├── lexical.py          # Layer 1: TF-IDF (+ global IDF table)
│   ├── semantic.py         # Layer 2: LSA via truncated SVD
│   ├── graph.py            # Layer 3: positive-PMI concept graph
│   ├── pipeline.py         # orchestrates all layers for one dump
│   └── storage.py          # GitHub API + local-fallback read/write
└── utils/
    └── parser.py           # conversation detection, turn tagging
```
