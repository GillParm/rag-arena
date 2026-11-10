"""Recursive splitting on a separator hierarchy, then greedy merging up to a target size."""

from __future__ import annotations

import itertools
from typing import Any

from rag_arena.chunking.base import register
from rag_arena.corpus.loader import Document

# Coarsest first. Heading separators cut before the heading so it starts the next piece;
# the others cut after the separator so it stays with the text it ends.
SEPARATORS: tuple[str, ...] = ("\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ")


def _cut_points(text: str, start: int, end: int, sep: str) -> list[int]:
    cuts: list[int] = []
    offset = 1 if sep.startswith("\n#") else len(sep)
    i = text.find(sep, start, end)
    while i != -1:
        cut = i + offset
        if start < cut < end:
            cuts.append(cut)
        i = text.find(sep, i + 1, end)
    return cuts


def _merge(pieces: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    """Merge neighbouring contiguous pieces while the result stays within `size`."""
    merged: list[tuple[int, int]] = []
    for a, b in pieces:
        if merged and b - merged[-1][0] <= size:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def split_range(
    text: str, start: int, end: int, size: int, separators: tuple[str, ...] = SEPARATORS
) -> list[tuple[int, int]]:
    """Split text[start:end] into contiguous spans of at most `size` characters."""
    if end - start <= size:
        return [(start, end)]
    for i, sep in enumerate(separators):
        cuts = _cut_points(text, start, end, sep)
        if not cuts:
            continue
        bounds = [start, *cuts, end]
        pieces: list[tuple[int, int]] = []
        for a, b in itertools.pairwise(bounds):
            if b - a > size:
                pieces.extend(split_range(text, a, b, size, separators[i + 1 :]))
            else:
                pieces.append((a, b))
        return _merge(pieces, size)
    return [(s, min(s + size, end)) for s in range(start, end, size)]


class RecursiveChunker:
    name = "recursive"

    def __init__(self, size: int = 1200) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size

    def params(self) -> dict[str, Any]:
        return {"size": self.size, "separators": list(SEPARATORS)}

    def split(self, doc: Document) -> list[tuple[int, int]]:
        return split_range(doc.text, 0, len(doc.text), self.size)

    def index_text(self, doc: Document, start: int, end: int) -> str:
        return doc.text[start:end]


@register("recursive")
def _make(**params: int) -> RecursiveChunker:
    return RecursiveChunker(**params)
