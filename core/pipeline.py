"""Orchestration: run one input through all four layers and update the index.

This ties Layers 0-3 together so the UI (and tests) have a single entry
point. The order matters:

1. Layer 0 stores the verbatim text and mints the dump id.
2. Layer 1 updates the global IDF table and writes per-dump TF-IDF.
3. Layer 3 builds the per-dump + global PMI graphs.
4. Layer 2 rebuilds the LSA factorisation across the whole (now larger)
   corpus, since SVD is global by nature.
5. The master ``index.json`` is updated last.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core import graph, lexical, raw, semantic
from core.ingest import PreprocessOptions, ingest, slugify
from core.storage import Storage, meta_path


def process_dump(
    storage: Storage,
    text: str,
    dump_type: str = "brain dump",
    opts: PreprocessOptions | None = None,
) -> dict:
    """Ingest one input and run every layer. Returns a summary dict.

    ``opts`` carries the user's chosen preprocessing (stop-word / filler
    removal, min length, conversation handling) so the corpus is built from
    the same cleaned tokens the user previewed in the UI.
    """
    parsed = ingest(text, dump_type, opts)

    # Folder id = timestamp + concept slug, e.g.
    # 2026-06-15_19-07-25__attention-transformers-tokens — sortable by time,
    # readable at a glance. store_raw guarantees uniqueness.
    proposed_id = f"{raw.make_dump_id()}__{slugify(parsed.title)}"
    dump_id = raw.store_raw(storage, parsed.full_text, dump_id=proposed_id)
    lex = lexical.process_lexical(storage, dump_id, parsed.tokens)
    grph = graph.process_graph(storage, dump_id, parsed.tokens)

    index = storage.load_index()
    dump_ids = sorted({d["id"] for d in index["dumps"]} | {dump_id})
    sem = semantic.rebuild(storage, dump_ids)

    entry = {
        "id": dump_id,
        "title": parsed.title,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": dump_type,
        "raw_path": raw.raw_path(dump_id),
        "is_conversation": parsed.is_conversation,
        "n_tokens": len(parsed.tokens),
        "top_keywords": [k["term"] for k in lex["top_keywords"][:5]],
    }
    # A human-readable summary lives alongside the dump's own files so each
    # dumps/<id>/ folder is self-describing when browsed directly.
    storage.write_json(meta_path(dump_id), entry, message=f"Meta {dump_id}")
    storage.add_to_index(entry)

    return {
        "id": dump_id,
        "title": parsed.title,
        "is_conversation": parsed.is_conversation,
        "lexical": lex,
        "graph": grph,
        "semantic": sem,
    }
