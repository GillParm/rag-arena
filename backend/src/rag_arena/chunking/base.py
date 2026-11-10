"""Shared chunk types and the chunker registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_arena.corpus.loader import Document


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    ordinal: int
    start: int
    end: int
    text: str  # always document.text[start:end]
    heading_path: tuple[str, ...]
    url: str
    index_text: str  # what BM25 and the embedder see

    def chunk_id(self, chunk_set: str) -> str:
        return f"{chunk_set}:{self.doc_id}:{self.ordinal}"


class Chunker(Protocol):
    name: str

    def params(self) -> dict[str, Any]: ...

    def split(self, doc: Document) -> list[tuple[int, int]]:
        """Return (start, end) spans into doc.text, in order."""
        ...

    def index_text(self, doc: Document, start: int, end: int) -> str: ...


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a span so it does not start or end with whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def build_chunks(chunker: Chunker, doc: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    for start, end in chunker.split(doc):
        start, end = trim_span(doc.text, start, end)
        if end <= start:
            continue
        section = doc.section_at(start)
        chunks.append(
            Chunk(
                doc_id=doc.id,
                ordinal=len(chunks),
                start=start,
                end=end,
                text=doc.text[start:end],
                heading_path=section.path if section else (doc.title,),
                url=doc.url_for(section),
                index_text=chunker.index_text(doc, start, end),
            )
        )
    return chunks


@dataclass(frozen=True)
class ChunkerSpec:
    """A chunker name plus parameter overrides; `set_id` names the index it builds."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def set_id(self) -> str:
        if not self.params:
            return self.name
        suffix = "-".join(f"{k}{v}" for k, v in sorted(self.params.items()))
        return f"{self.name}-{suffix}"

    @classmethod
    def parse(cls, value: str) -> ChunkerSpec:
        """'fixed' or 'fixed:size=800,overlap=100'."""
        name, _, rest = value.partition(":")
        params: dict[str, Any] = {}
        for item in filter(None, rest.split(",")):
            key, _, raw = item.partition("=")
            params[key.strip()] = int(raw)
        return cls(name.strip(), params)


_REGISTRY: dict[str, Callable[..., Chunker]] = {}


def register(name: str) -> Callable[[Callable[..., Chunker]], Callable[..., Chunker]]:
    def deco(factory: Callable[..., Chunker]) -> Callable[..., Chunker]:
        _REGISTRY[name] = factory
        return factory

    return deco


def make_chunker(spec: ChunkerSpec | str) -> Chunker:
    # Import for the side effect of registering the built-in chunkers.
    from rag_arena.chunking import fixed, heading, recursive  # noqa: F401

    if isinstance(spec, str):
        spec = ChunkerSpec.parse(spec)
    try:
        factory = _REGISTRY[spec.name]
    except KeyError:
        known = sorted(_REGISTRY)
        raise ValueError(f"unknown chunker {spec.name!r}; choose from {known}") from None
    return factory(**spec.params)


def chunker_names() -> list[str]:
    from rag_arena.chunking import fixed, heading, recursive  # noqa: F401

    return sorted(_REGISTRY)
