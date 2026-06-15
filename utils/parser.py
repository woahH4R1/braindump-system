"""Conversation structure detection and turn tagging.

A "dump" can be either a raw brain dump (free text) or a copy-pasted
conversation with an assistant. We detect the latter by scanning for
speaker prefixes at the start of a line, e.g. ``You:``, ``Gemini:``.

When a conversation is detected we tag every turn with its speaker and
classify the speaker as the *user* (the human) or the *assistant*. Only
user turns carry the human's own signal, so downstream layers process
just those, while Layer 0 still preserves the full transcript verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Speaker labels that mark a HUMAN turn.
USER_SPEAKERS = {"you", "me", "human", "user", "i"}

# Speaker labels that mark an ASSISTANT turn.
ASSISTANT_SPEAKERS = {
    "gemini",
    "chatgpt",
    "gpt",
    "assistant",
    "claude",
    "bot",
    "ai",
    "bard",
    "copilot",
}

ALL_SPEAKERS = USER_SPEAKERS | ASSISTANT_SPEAKERS

# A turn header looks like ``Speaker:`` at the very start of a line.
# We allow optional surrounding markdown (**You:**) and trailing space.
_SPEAKER_RE = re.compile(
    r"^\s*\**\s*(" + "|".join(sorted(ALL_SPEAKERS, key=len, reverse=True)) + r")\s*\**\s*:",
    re.IGNORECASE,
)


@dataclass
class Turn:
    """A single tagged turn in a conversation."""

    speaker: str          # normalised lower-case label, e.g. "you"
    role: str             # "user" or "assistant"
    text: str             # the body of the turn (prefix stripped)


def _classify(speaker: str) -> str:
    return "user" if speaker.lower() in USER_SPEAKERS else "assistant"


def detect_conversation(text: str, min_turns: int = 2) -> bool:
    """Return True when ``text`` looks like a multi-turn conversation.

    We require at least ``min_turns`` lines that begin with a known
    speaker prefix. A single ``Note:`` style colon line therefore does
    not trip the detector.
    """
    matches = 0
    for line in text.splitlines():
        if _SPEAKER_RE.match(line):
            matches += 1
            if matches >= min_turns:
                return True
    return False


def parse_turns(text: str) -> list[Turn]:
    """Split a conversation into tagged :class:`Turn` objects.

    Lines that arrive before the first recognised speaker are attached to
    an implicit leading ``user`` turn so no text is lost.
    """
    turns: list[Turn] = []
    cur_speaker: str | None = None
    cur_lines: list[str] = []

    def flush() -> None:
        if cur_speaker is None and not any(s.strip() for s in cur_lines):
            return
        speaker = cur_speaker if cur_speaker is not None else "you"
        turns.append(
            Turn(
                speaker=speaker,
                role=_classify(speaker),
                text="\n".join(cur_lines).strip(),
            )
        )

    for line in text.splitlines():
        m = _SPEAKER_RE.match(line)
        if m:
            # Close the previous turn and start a new one.
            if cur_speaker is not None or any(s.strip() for s in cur_lines):
                flush()
            cur_speaker = m.group(1).lower()
            cur_lines = [line[m.end():].lstrip()]
        else:
            cur_lines.append(line)
    flush()
    return [t for t in turns if t.text]


def extract_user_text(text: str) -> str:
    """Return only the human turns of a conversation, joined by blank lines.

    If ``text`` is not a conversation it is returned unchanged.
    """
    if not detect_conversation(text):
        return text
    user_turns = [t.text for t in parse_turns(text) if t.role == "user"]
    return "\n\n".join(user_turns) if user_turns else text
