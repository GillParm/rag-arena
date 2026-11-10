from __future__ import annotations

from pathlib import Path

import pytest

from rag_arena.corpus.loader import Document, load_corpus
from rag_arena.embedding import HashEmbedder
from rag_arena.retrieval.backend import SearchBackend, numpy_dense
from rag_arena.retrieval.bm25 import BM25Index

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_root() -> Path:
    return FIXTURES / "fastapi_mini"


@pytest.fixture
def mini_docs(mini_root: Path) -> list[Document]:
    return load_corpus(mini_root)


def make_backend(texts: dict[str, str], set_id: str = "test") -> SearchBackend:
    """An offline SearchBackend: hashing embedder, in-memory dense search, real BM25."""
    from rag_arena.db.repo import ChunkRow

    embedder = HashEmbedder(64)
    ids = list(texts)
    rows = {
        cid: ChunkRow(
            id=cid,
            chunk_set=set_id,
            document_id=cid.split(":")[1] if ":" in cid else "doc",
            ordinal=n,
            start=0,
            end=len(text),
            text=text,
            index_text=text,
            heading_path=["Doc"],
            url="https://example.test/",
            doc_title="Doc",
        )
        for n, (cid, text) in enumerate(texts.items())
    }
    return SearchBackend(
        set_id=set_id,
        bm25=BM25Index.build(ids, list(texts.values())),
        chunks=rows,
        embedder=embedder,
        dense=numpy_dense(ids, embedder.embed_passages(list(texts.values()))),
    )


CHUNKS = {
    "t:deps:0": "Dependencies are declared with Depends in the path operation parameters.",
    "t:deps:1": "Sub-dependencies let a dependency depend on another dependency.",
    "t:resp:0": "Use response_model to declare the shape of the returned data.",
    "t:bg:0": "Background tasks run after returning a response to the client.",
    "t:cors:0": "Configure CORS with CORSMiddleware and a list of allowed origins.",
}


@pytest.fixture
def backend() -> SearchBackend:
    return make_backend(CHUNKS)
