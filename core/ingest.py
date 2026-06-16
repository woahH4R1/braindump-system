"""Input parsing, tokenisation and configurable preprocessing.

This is the front door for every dump. It turns raw text into a clean token
stream, and it does so *transparently*: the :class:`PreprocessOptions` flags
let the UI offer "click to remove filler", "drop short words", etc., and
:func:`preprocess` reports exactly what each step removed so the user can see
the transformation happen instead of trusting a black box.

Keeping tokenisation here guarantees every downstream layer (TF-IDF, LSA,
PMI) sees exactly the same tokens, so their views of a dump stay consistent.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from utils.parser import detect_conversation, extract_user_text

# Grammatical stop-words — almost never carry signal. Kept inline so the
# system has no external corpus dependency.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "for", "from", "had", "has",
    "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "me", "my", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "about", "again", "all", "am", "any", "because", "before", "below",
    "between", "both", "during", "each", "few", "further", "more", "most",
    "now", "been", "being",
}

# Contractions — the tokeniser keeps the apostrophe, so these arrive as whole
# tokens (e.g. "i'm", "that's"). Treated as stop-words: they never carry the
# signal of a dump. This is what was leaking "i'm / that's / don't" into the
# old keyword lists and auto-titles.
CONTRACTIONS = {
    "i'm", "i've", "i'd", "i'll", "you're", "you've", "you'd", "you'll",
    "we're", "we've", "we'd", "we'll", "they're", "they've", "they'd",
    "they'll", "he's", "she's", "it's", "that's", "there's", "here's",
    "what's", "who's", "where's", "let's", "how's", "isn't", "aren't",
    "wasn't", "weren't", "don't", "doesn't", "didn't", "haven't", "hasn't",
    "hadn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't",
    "mustn't", "ain't",
    # apostrophe-stripped forms, just in case
    "im", "ive", "id", "ill", "youre", "youve", "weve", "theyre", "thats",
    "dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "havent", "cant",
    "wont", "gonna", "wanna", "gotta",
}

# Discourse fillers — real words that mostly pad writing. Opt-in removal
# (the "remove filler" switch), so the user decides how aggressive to be.
FILLER = {
    "like", "just", "really", "actually", "basically", "literally",
    "essentially", "kinda", "sorta", "guess", "maybe", "stuff", "thing",
    "things", "lot", "lots", "bit", "kind", "sort", "etc", "anyway",
    "anyways", "well", "right", "mean", "know", "yeah", "yep", "yup", "nope",
    "ok", "okay", "yes", "uh", "um", "hmm", "blah", "get", "got", "gets",
    "getting", "thats", "stuffs",
}

_TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")


def _removal_set(drop_stopwords: bool, drop_filler: bool) -> set[str]:
    """The union of word-lists to strip, given the chosen flags."""
    remove: set[str] = set()
    if drop_stopwords:
        remove |= STOPWORDS | CONTRACTIONS
    if drop_filler:
        remove |= FILLER
    return remove


def tokenize(
    text: str,
    min_len: int = 3,
    drop_stopwords: bool = True,
    drop_filler: bool = False,
) -> list[str]:
    """Lower-case, strip punctuation, drop short tokens and chosen word-lists."""
    remove = _removal_set(drop_stopwords, drop_filler)
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tok = tok.strip("'-")
        if len(tok) < min_len:
            continue
        if tok in remove:
            continue
        out.append(tok)
    return out


@dataclass
class PreprocessOptions:
    """User-selectable preprocessing steps for one dump."""

    drop_stopwords: bool = True
    drop_filler: bool = True
    min_len: int = 3
    conversation_user_only: bool = True


@dataclass
class Ingested:
    """Result of ingesting a single piece of input."""

    full_text: str                 # verbatim, goes to Layer 0
    signal_text: str               # text used for signal layers
    tokens: list[str] = field(default_factory=list)
    is_conversation: bool = False
    dump_type: str = "brain dump"
    title: str = "untitled"        # short concept label derived from content


def preprocess(text: str, opts: PreprocessOptions | None = None) -> dict:
    """Run the chosen preprocessing steps and report what each one removed.

    Returns a dict the UI can render directly: the cleaned token stream, the
    signal text actually analysed, and a per-category breakdown of removals so
    the transformation is visible rather than implicit.
    """
    opts = opts or PreprocessOptions()
    text = text.replace("\r\n", "\n").strip()

    is_conv = detect_conversation(text)
    signal = (
        extract_user_text(text)
        if (is_conv and opts.conversation_user_only)
        else text
    )

    # Raw tokens before any filtering, for honest before/after stats.
    raw_tokens = [t.strip("'-") for t in _TOKEN_RE.findall(signal.lower())]
    raw_tokens = [t for t in raw_tokens if t]

    stop_set = STOPWORDS | CONTRACTIONS
    removed_short = removed_stop = removed_filler = 0
    for tok in raw_tokens:
        if len(tok) < opts.min_len:
            removed_short += 1
        elif opts.drop_stopwords and tok in stop_set:
            removed_stop += 1
        elif opts.drop_filler and tok in FILLER:
            removed_filler += 1

    tokens = tokenize(
        signal,
        min_len=opts.min_len,
        drop_stopwords=opts.drop_stopwords,
        drop_filler=opts.drop_filler,
    )

    return {
        "is_conversation": is_conv,
        "signal_text": signal,
        "tokens": tokens,
        "n_raw": len(raw_tokens),
        "n_kept": len(tokens),
        "n_removed": len(raw_tokens) - len(tokens),
        "n_unique": len(set(tokens)),
        "removed_short": removed_short,
        "removed_stop": removed_stop,
        "removed_filler": removed_filler,
        "top_freq": Counter(tokens).most_common(25),
    }


def make_title(tokens: list[str], n: int = 4) -> str:
    """A short, human-readable concept label from the most frequent terms.

    Cheap and model-free: the words you used most (after cleaning) are a decent
    proxy for what a dump is about, e.g. "Attention Transformers Tokens".
    Falls back to "untitled" for empty/tokenless input.
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


def ingest(
    text: str,
    dump_type: str = "brain dump",
    opts: PreprocessOptions | None = None,
) -> Ingested:
    """Parse input, route conversations vs raw dumps, and tokenise."""
    opts = opts or PreprocessOptions()
    pre = preprocess(text, opts)
    return Ingested(
        full_text=text.replace("\r\n", "\n").strip(),
        signal_text=pre["signal_text"],
        tokens=pre["tokens"],
        is_conversation=pre["is_conversation"],
        dump_type=dump_type,
        title=make_title(pre["tokens"]),
    )
