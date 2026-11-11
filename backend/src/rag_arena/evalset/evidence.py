"""Locate a model-quoted evidence passage in the document text to get character offsets."""

from __future__ import annotations

_DROP = frozenset("`*")


def _normalize(text: str, *, drop_marks: bool) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space (optionally drop ` and *), keeping an index map."""
    chars: list[str] = []
    index: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        if drop_marks and ch in _DROP:
            continue
        if ch.isspace():
            if prev_space:
                continue
            chars.append(" ")
            index.append(i)
            prev_space = True
        else:
            chars.append(ch)
            index.append(i)
            prev_space = False
    return "".join(chars), index


def locate(
    text: str, quote: str, *, hint_start: int = 0, hint_end: int | None = None
) -> tuple[int, int] | None:
    """Return (start, end) of `quote` in `text`, preferring a match inside the hint range.

    Tries an exact match first, then a whitespace-insensitive match, then one that also
    ignores backticks and asterisks (models sometimes drop markdown emphasis when quoting).
    """
    quote = quote.strip()
    if not quote:
        return None
    hint_end = len(text) if hint_end is None else hint_end
    i = text.find(quote, hint_start, hint_end)
    if i == -1:
        i = text.find(quote)
    if i != -1:
        return i, i + len(quote)
    for drop in (False, True):
        norm_text, index = _normalize(text, drop_marks=drop)
        norm_quote, _ = _normalize(quote, drop_marks=drop)
        norm_quote = norm_quote.strip()
        if not norm_quote:
            continue
        j = norm_text.find(norm_quote)
        if j != -1:
            return index[j], index[j + len(norm_quote) - 1] + 1
    return None
