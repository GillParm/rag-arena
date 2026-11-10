"""One chunk per markdown section, keeping the heading path as context.

Sections shorter than `min_size` merge forward into the next section when that section is
nested under them (an intro paragraph joins its first subsection). Sections longer than
`max_size` are split with the recursive splitter. The heading path is prepended to the text
that BM25 and the embedder see.
"""

from __future__ import annotations

from typing import Any

from rag_arena.chunking.base import register, trim_span
from rag_arena.chunking.recursive import split_range
from rag_arena.corpus.loader import Document


class HeadingChunker:
    name = "heading"

    def __init__(self, max_size: int = 1600, min_size: int = 200, split_size: int = 1200) -> None:
        if not 0 < min_size < max_size or split_size > max_size:
            raise ValueError("need 0 < min_size < max_size and split_size <= max_size")
        self.max_size = max_size
        self.min_size = min_size
        self.split_size = split_size

    def params(self) -> dict[str, Any]:
        return {"max_size": self.max_size, "min_size": self.min_size, "split_size": self.split_size}

    def _sections(self, doc: Document) -> list[tuple[int, int, int]]:
        """(start, end, level) per section, plus any preamble before the first heading."""
        n = len(doc.text)
        starts = [s.start for s in doc.sections]
        out: list[tuple[int, int, int]] = []
        if not starts or starts[0] > 0:
            out.append((0, starts[0] if starts else n, 0))
        for i, sec in enumerate(doc.sections):
            end = starts[i + 1] if i + 1 < len(starts) else n
            out.append((sec.start, end, sec.level))
        return out

    def split(self, doc: Document) -> list[tuple[int, int]]:
        sections = self._sections(doc)
        merged: list[tuple[int, int]] = []
        carry: tuple[int, int] | None = None  # a short parent waiting to join its child
        for i, (start, end, level) in enumerate(sections):
            if carry is not None:
                start = carry[0]
                carry = None
            a, b = trim_span(doc.text, start, end)
            nxt = sections[i + 1] if i + 1 < len(sections) else None
            if (
                b - a < self.min_size
                and nxt is not None
                and nxt[2] > level
                and nxt[1] - start <= self.max_size
            ):
                carry = (start, end)
                continue
            merged.append((start, end))
        spans: list[tuple[int, int]] = []
        for start, end in merged:
            if end - start > self.max_size:
                spans.extend(split_range(doc.text, start, end, self.split_size))
            else:
                spans.append((start, end))
        return spans

    def index_text(self, doc: Document, start: int, end: int) -> str:
        section = doc.section_at(max(start, end - 1))
        path = " > ".join(section.path) if section else doc.title
        return f"{path}\n\n{doc.text[start:end]}"


@register("heading")
def _make(**params: int) -> HeadingChunker:
    return HeadingChunker(**params)
