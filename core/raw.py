"""Layer 0 — Raw.

The simplest and most important layer: store the input verbatim and never
touch it again. Everything else in the system is a lossy *view*; this is
the ground truth you can always return to.

Each dump gets its own folder ``dumps/YYYY-MM-DD_HH-MM-SS/`` and the verbatim
text is written to ``raw.txt`` inside it. The timestamp doubles as the stable
dump id used everywhere downstream.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.storage import Storage, raw_path


def make_dump_id(when: datetime | None = None) -> str:
    """Return a sortable ``YYYY-MM-DD_HH-MM-SS`` id (UTC)."""
    when = when or datetime.now(timezone.utc)
    return when.strftime("%Y-%m-%d_%H-%M-%S")


def store_raw(storage: Storage, text: str, dump_id: str | None = None) -> str:
    """Store ``text`` verbatim and return its dump id.

    Raw files are write-once: if the id already exists a new one is minted
    so an existing record can never be overwritten.
    """
    dump_id = dump_id or make_dump_id()
    while storage.exists(raw_path(dump_id)):
        # Extremely unlikely collision (same second) — nudge forward.
        dump_id += "_x"
    storage.write_text(raw_path(dump_id), text, message=f"Raw dump {dump_id}")
    return dump_id


def load_raw(storage: Storage, dump_id: str) -> str | None:
    """Return the verbatim text for a dump id, or None if missing."""
    return storage.read_text(raw_path(dump_id))
