"""Input parsing and tokenisation — the front door for every dump.

``ingest`` takes raw input text plus a user-chosen label and returns a
normalised record:

* the full verbatim text (for Layer 0),
* the *signal* text actually fed to Layers 1-3 (for a conversation this is
  only the human's turns),
* the token stream used by every quantitative layer.

Keeping tokenisation here guarantees Layers 1, 2 and 3 all see exactly the
same tokens, so their views of a dump stay consistent.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from utils.parser import detect_conversation, extract_user_text

# A compact English stop-word list. Kept inline so the system has no
# external corpus dependency.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "for", "from", "had", "has",
    "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "just", "me", "my", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "about", "again", "all", "am", "any", "because", "before", "below",
    "between", "both", "during", "each", "few", "further", "more", "most",
    "now", "ok", "okay", "really", "thing", "things", "get", "got", "like",
    "im", "ive", "dont", "youre", "thats", "gonna", "wanna",
}

_TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")


def tokenize(text: str, min_len: int = 3, drop_stopwords: bool = True) -> list[str]:
    """Lower-case, strip punctuation, drop short tokens and stop words."""
    tokens = _TOKEN_RE.findall(text.lower())
    out = []
    for tok in tokens:
        tok = tok.strip("'-")
        if len(tok) < min_len:
            continue
        if drop_stopwords and tok in STOPWORDS:
            continue
        out.append(tok)
    return out


@dataclass
class Ingested:
    """Result of ingesting a single piece of input."""

    full_text: str                 # verbatim, goes to Layer 0
    signal_text: str               # text used for signal layers
    tokens: list[str] = field(default_factory=list)
    is_conversation: bool = False
    dump_type: str = "brain dump"
    title: str = "untitled"        # short concept label derived from content


def make_title(tokens: list[str], n: int = 4) -> str:
    """A short, human-readable concept label from the most frequent terms.

    Cheap and model-free: the words you used most (after stop-word removal)
    are a decent proxy for what a dump is about, e.g. "Attention Transformers
    Tokens". Falls back to "untitled" for empty/tokenless input.
    """
    if not tokens:
        return "untitled"
    # most_common preserves first-seen order on ties, keeping titles stable.
    terms = [t for t, _ in Counter(tokens).most_common(n)]
    return " ".join(terms).title()


def slugify(text: str, max_len: int = 48) -> str:
    """Filesystem/URL-safe slug, e.g. "attention-transformers-tokens"."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "untitled"


def ingest(text: str, dump_type: str = "brain dump") -> Ingested:
    """Parse input, route conversations vs raw dumps, and tokenise."""
    text = text.replace("\r\n", "\n").strip()
    is_conv = detect_conversation(text)
    signal = extract_user_text(text) if is_conv else text
    tokens = tokenize(signal)
    return Ingested(
        full_text=text,
        signal_text=signal,
        tokens=tokens,
        is_conversation=is_conv,
        dump_type=dump_type,
        title=make_title(tokens),
    )
