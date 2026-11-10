"""Wires settings, indexes and models together for the CLI, the benchmark and the API."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import cached_property
from pathlib import Path

from rag_arena.config import Settings
from rag_arena.db import repo
from rag_arena.embedding import Embedder, FastEmbedder
from rag_arena.indexing import load_bm25
from rag_arena.llm.base import TextLLM
from rag_arena.retrieval.backend import SearchBackend, pg_dense
from rag_arena.retrieval.rerank import FastReranker, Reranker
from rag_arena.retrieval.rewrite import LLMRewriter, Rewriter

ConnFactory = Callable[[], AbstractContextManager[repo.Conn]]


class MissingKeyError(RuntimeError):
    pass


class Runtime:
    def __init__(
        self,
        settings: Settings,
        conn_factory: ConnFactory,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        llm: TextLLM | None = None,
    ) -> None:
        self.settings = settings
        self.conn_factory = conn_factory
        self._embedder = embedder
        self._reranker = reranker
        self._llm = llm
        self._backends: dict[str, SearchBackend] = {}

    @property
    def model_dir(self) -> Path:
        return self.settings.model_cache_dir or self.settings.cache_dir / "models"

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = FastEmbedder(self.settings.embedding_model, cache_dir=self.model_dir)
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = FastReranker(self.settings.reranker_model, cache_dir=self.model_dir)
        return self._reranker

    @property
    def llm(self) -> TextLLM:
        if self._llm is None:
            if not self.settings.has_gemini():
                raise MissingKeyError("GEMINI_API_KEY is not set")
            from rag_arena.llm.gemini import GeminiClient

            key = self.settings.gemini_api_key
            assert key is not None
            self._llm = GeminiClient(
                key.get_secret_value(),
                self.settings.generator_model,
                rpm=self.settings.gemini_rpm,
                concurrency=self.settings.gemini_concurrency,
            )
        return self._llm

    @cached_property
    def rewriter(self) -> Rewriter:
        return LLMRewriter(self.llm)

    def chunk_sets(self) -> list[repo.ChunkSetInfo]:
        with self.conn_factory() as conn:
            return repo.list_chunk_sets(conn)

    def backend(self, set_id: str) -> SearchBackend:
        if set_id not in self._backends:
            with self.conn_factory() as conn:
                info = repo.get_chunk_set(conn, set_id)
                if info is None:
                    raise LookupError(f"chunk set {set_id!r} is not indexed")
                if info.embedding_model != self.embedder.model_name:
                    raise RuntimeError(
                        f"{set_id} was embedded with {info.embedding_model}, but the runtime"
                        f" embedder is {self.embedder.model_name}; rebuild the index"
                    )
                bm25 = load_bm25(conn, set_id, self.settings.cache_dir)
                chunks = {row.id: row for row in repo.fetch_chunks(conn, set_id)}
            self._backends[set_id] = SearchBackend(
                set_id=set_id,
                bm25=bm25,
                chunks=chunks,
                embedder=self.embedder,
                dense=pg_dense(self.conn_factory, set_id, info.embedding_dim),
            )
        return self._backends[set_id]
