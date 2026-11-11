"""Pick doc sections to write questions from: deterministic, spread across pages."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from rag_arena.corpus.loader import Document, Section


@dataclass(frozen=True)
class SectionSample:
    doc: Document
    section: Section
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.doc.text[self.start : self.end]

    @property
    def key(self) -> str:
        return f"{self.doc.id}#{self.section.anchor}"


def section_spans(doc: Document) -> list[tuple[Section, int, int]]:
    """Each section from its heading to the next heading of the same or a higher level."""
    out: list[tuple[Section, int, int]] = []
    for i, sec in enumerate(doc.sections):
        end = len(doc.text)
        for later in doc.sections[i + 1 :]:
            if later.level <= sec.level:
                end = later.start
                break
        # Stop at the first subsection too, so a sample is one focused section.
        if i + 1 < len(doc.sections) and doc.sections[i + 1].start < end:
            end = doc.sections[i + 1].start
        out.append((sec, sec.start, end))
    return out


def sample_sections(
    docs: Sequence[Document],
    n: int,
    *,
    seed: int = 42,
    min_chars: int = 400,
    max_chars: int = 6000,
    per_doc: int = 2,
) -> list[SectionSample]:
    """Round-robin over pages (shuffled), taking up to `per_doc` random eligible sections each."""
    rng = random.Random(seed)
    pools: list[list[SectionSample]] = []
    for doc in sorted(docs, key=lambda d: d.id):
        eligible = [
            SectionSample(doc, sec, a, b)
            for sec, a, b in section_spans(doc)
            if min_chars <= len(doc.text[a:b].strip()) <= max_chars
        ]
        rng.shuffle(eligible)
        if eligible:
            pools.append(eligible[:per_doc])
    rng.shuffle(pools)
    picked: list[SectionSample] = []
    for round_ in range(per_doc):
        for pool in pools:
            if round_ < len(pool) and len(picked) < n:
                picked.append(pool[round_])
    return picked
