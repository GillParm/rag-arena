"""Fixed-size windows with overlap, snapped to whitespace so words are not cut."""

from __future__ import annotations

from typing import Any

from rag_arena.chunking.base import register
from rag_arena.corpus.loader import Document


def _snap_back(text: str, pos: int, floor: int) -> int:
    """Move `pos` back to just after the nearest whitespace, but not before `floor`."""
    i = pos
    while i > floor and not text[i - 1].isspace():
        i -= 1
    return i if i > floor else pos


def _snap_forward(text: str, pos: int, ceiling: int) -> int:
    """Move `pos` forward to the start of the next word, but not past `ceiling`."""
    i = pos
    while i < ceiling and not text[i].isspace() and i > 0 and not text[i - 1].isspace():
        i += 1
    while i < ceiling and text[i].isspace():
        i += 1
    return i if i < ceiling else pos


class FixedChunker:
    name = "fixed"

    def __init__(self, size: int = 1200, overlap: int = 200) -> None:
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError("need size > 0 and 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    def params(self) -> dict[str, Any]:
        return {"size": self.size, "overlap": self.overlap}

    def split(self, doc: Document) -> list[tuple[int, int]]:
        text = doc.text
        n = len(text)
        spans: list[tuple[int, int]] = []
        start = 0
        while start < n:
            end = min(start + self.size, n)
            if end < n:
                # Snap to whitespace within the last 20% of the window.
                end = _snap_back(text, end, start + int(self.size * 0.8))
            spans.append((start, end))
            if end >= n:
                break
            nxt = _snap_forward(text, max(end - self.overlap, start + 1), end)
            start = nxt if nxt > start else end
        return spans

    def index_text(self, doc: Document, start: int, end: int) -> str:
        return doc.text[start:end]


@register("fixed")
def _make(**params: int) -> FixedChunker:
    return FixedChunker(**params)
