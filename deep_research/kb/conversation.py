"""Parses a pasted chat conversation into (speaker, turn_text) pairs so
claims extracted from it can be attributed to whoever said them -- a
conversation can be between two or more people (or a person and an AI), and
every claim gets fact-checked regardless of who made it.

Heuristic, not a real chat-export parser: a line starting with a short
name-like token followed by a colon (e.g. "Alice:", "User:", "ChatGPT:")
starts a new turn; everything until the next such line belongs to that
speaker. Text with no recognizable speaker lines at all is returned as one
untagged turn (speaker=None), so a plain-prose paste still works like any
other document instead of failing to parse.
"""

import re

_SPEAKER_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 ._'-]{0,40}):\s?(.*)$")


def parse_conversation_turns(text: str) -> list[tuple[str | None, str]]:
    turns: list[tuple[str | None, str]] = []
    current_speaker: str | None = None
    current_lines: list[str] = []

    def flush():
        content = "\n".join(current_lines).strip()
        if content:
            turns.append((current_speaker, content))

    for line in text.splitlines():
        match = _SPEAKER_LINE_RE.match(line)
        if match:
            flush()
            current_speaker = match.group(1).strip()
            current_lines = [match.group(2)] if match.group(2) else []
        else:
            current_lines.append(line)
    flush()

    if not turns:
        return [(None, text.strip())] if text.strip() else []

    # If nothing ever looked like a real speaker line (every "turn" is
    # actually just the whole text under one label from line 1), it's
    # probably not a conversation -- but this is rare enough in practice
    # (a real transcript has multiple turns) not to special-case further.
    return turns
