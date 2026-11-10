from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from rag_arena.chunking.base import ChunkerSpec, build_chunks, chunker_names, make_chunker
from rag_arena.chunking.fixed import FixedChunker
from rag_arena.chunking.heading import HeadingChunker
from rag_arena.chunking.recursive import RecursiveChunker, split_range
from rag_arena.corpus.loader import Document, process_markdown

LOREM = (
    "FastAPI validates request data with Pydantic models and returns clear errors. "
    "Path operations declare parameters with type hints, and the docs are generated. "
)


def make_doc(raw: str, tmp_path: Path) -> Document:
    text, sections = process_markdown(raw, tmp_path)
    return Document("t/doc", "t/doc.md", "Doc", "https://x/t/doc/", text, tuple(sections))


@pytest.fixture
def long_doc(tmp_path: Path) -> Document:
    raw = (
        "# Guide { #guide }\n\nShort intro.\n\n"
        "## Install { #install }\n\n" + LOREM * 6 + "\n\n"
        "## Use it\n\n" + LOREM * 30 + "\n\n"
        "```python\n# this comment is not a heading\nprint('hi')\n```\n\n"
        "### Details\n\n" + LOREM * 3 + "\n"
    )
    return make_doc(raw, tmp_path)


@pytest.mark.parametrize("name", ["fixed", "recursive", "heading"])
def test_chunks_are_exact_slices_that_cover_the_text(name: str, long_doc: Document) -> None:
    chunks = build_chunks(make_chunker(name), long_doc)
    assert chunks
    covered = [False] * len(long_doc.text)
    for c in chunks:
        assert c.text == long_doc.text[c.start : c.end]
        assert c.text == c.text.strip()
        for i in range(c.start, c.end):
            covered[i] = True
    missed = [ch for ch, ok in zip(long_doc.text, covered, strict=True) if not ok]
    assert all(ch.isspace() for ch in missed)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_fixed_respects_size_and_overlaps(long_doc: Document) -> None:
    chunker = FixedChunker(size=400, overlap=80)
    chunks = build_chunks(chunker, long_doc)
    assert all(len(c.text) <= 400 for c in chunks)
    overlaps = [a.end - b.start for a, b in itertools.pairwise(chunks)]
    assert all(o > 0 for o in overlaps)
    assert max(overlaps) <= 80


def test_fixed_does_not_cut_words(long_doc: Document) -> None:
    for c in build_chunks(FixedChunker(size=300, overlap=50), long_doc):
        if c.end < len(long_doc.text):
            assert long_doc.text[c.end].isspace()
        if c.start > 0:
            assert long_doc.text[c.start - 1].isspace()


def test_fixed_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="overlap"):
        FixedChunker(size=100, overlap=100)


def test_recursive_prefers_heading_boundaries() -> None:
    text = "## A\n" + "x " * 100 + "\n## B\n" + "y " * 100
    spans = split_range(text, 0, len(text), 250)
    assert [text[a:b].lstrip()[:4] for a, b in spans] == ["## A", "## B"]


def test_recursive_never_exceeds_size(long_doc: Document) -> None:
    chunks = build_chunks(RecursiveChunker(size=300), long_doc)
    assert all(len(c.text) <= 300 for c in chunks)


def test_recursive_hard_splits_text_without_separators() -> None:
    text = "a" * 25
    assert split_range(text, 0, 25, 10) == [(0, 10), (10, 20), (20, 25)]


def test_heading_chunker_follows_sections(long_doc: Document) -> None:
    chunks = build_chunks(HeadingChunker(max_size=1600, min_size=200), long_doc)
    starts = [c.text.splitlines()[0] for c in chunks]
    # The short intro under "# Guide" merges into its first subsection.
    assert starts[0] == "# Guide"
    assert "## Install" in chunks[0].text
    assert all(len(c.text) <= 1600 for c in chunks)
    assert not any(c.text.startswith("# this comment") for c in chunks)
    assert chunks[-1].text.startswith("### Details")


def test_heading_chunker_prepends_heading_path(long_doc: Document) -> None:
    chunks = build_chunks(HeadingChunker(), long_doc)
    details = chunks[-1]
    assert details.index_text.startswith("Guide > Use it > Details\n\n### Details")
    assert details.heading_path == ("Guide", "Use it", "Details")
    assert details.url == "https://x/t/doc/#details"


def test_heading_chunker_splits_long_sections(long_doc: Document) -> None:
    use_it = [c for c in build_chunks(HeadingChunker(), long_doc) if c.heading_path[-1] == "Use it"]
    assert len(use_it) > 1
    assert all(c.index_text.startswith("Guide > Use it\n\n") for c in use_it)


def test_spec_parsing_and_set_ids() -> None:
    assert ChunkerSpec.parse("heading").set_id == "heading"
    spec = ChunkerSpec.parse("fixed:size=800,overlap=100")
    assert spec.params == {"size": 800, "overlap": 100}
    assert spec.set_id == "fixed-overlap100-size800"
    assert isinstance(make_chunker(spec), FixedChunker)
    assert chunker_names() == ["fixed", "heading", "recursive"]
    with pytest.raises(ValueError, match="unknown chunker"):
        make_chunker("sentences")


def test_chunk_ids_include_the_chunk_set(long_doc: Document) -> None:
    chunk = build_chunks(make_chunker("fixed"), long_doc)[0]
    assert chunk.chunk_id("fixed") == "fixed:t/doc:0"
