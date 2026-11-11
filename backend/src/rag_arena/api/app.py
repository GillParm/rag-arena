"""FastAPI application: `uvicorn --factory rag_arena.api.app:create_app --port 8300`."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from rag_arena import __version__
from rag_arena.api.schemas import AskRequest, AskResponse, ChunkOut, ConfigResponse
from rag_arena.api.service import ArenaService
from rag_arena.config import Settings, get_settings

log = logging.getLogger("rag_arena.api")


class SlidingWindowLimiter:
    """Per-client request cap over a rolling minute. In memory, so per instance."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        hits = self._hits[client]
        while hits and now - hits[0] >= 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            return False
        hits.append(now)
        return True


def _client_id(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def build_service(settings: Settings) -> tuple[ArenaService, Any]:
    """The production service: a connection pool and the real models."""
    from rag_arena.db.connection import make_pool
    from rag_arena.runtime import Runtime

    pool = make_pool(settings.database_url)
    runtime = Runtime(settings, pool.connection)
    service = ArenaService(
        runtime,
        generation_enabled=settings.has_gemini(),
        generator_model=settings.generator_model,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
        corpus=f"FastAPI docs {settings.corpus_ref}",
        leaderboard_path=settings.results_dir / "runs" / "main" / "summary.json",
    )
    return service, pool


def _warm(service: ArenaService) -> None:
    """Load indexes and models before the first request (Cloud Run start-up probe time)."""
    try:
        sets = [s.id for s in service.runtime.chunk_sets()]
        default = service.default_chunk_set(sets)
        if default:
            service.runtime.backend(default)
        _ = service.runtime.reranker
    except Exception:  # the API still serves /health and the leaderboard without an index
        log.exception("warm-up failed")


def create_app(settings: Settings | None = None, service: ArenaService | None = None) -> FastAPI:
    settings = settings or get_settings()
    limiter = SlidingWindowLimiter(settings.ask_rate_limit_per_minute)
    state: dict[str, Any] = {"service": service}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        pool = None
        if state["service"] is None:
            built, pool = build_service(settings)
            state["service"] = built
            await asyncio.to_thread(_warm, built)
        try:
            yield
        finally:
            if pool is not None:
                pool.close()

    app = FastAPI(
        title="rag-arena",
        version=__version__,
        summary="Compare RAG retrieval pipelines on the FastAPI docs.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    def svc() -> ArenaService:
        s = state["service"]
        if s is None:
            raise HTTPException(503, "service is starting")
        assert isinstance(s, ArenaService)
        return s

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/config", response_model=ConfigResponse)
    def config() -> ConfigResponse:
        try:
            return svc().config()
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("config failed")
            raise HTTPException(503, "index unavailable; has `rag-arena index build` run?") from exc

    @app.post("/api/ask", response_model=AskResponse)
    async def ask(req: AskRequest, request: Request) -> AskResponse:
        if not limiter.allow(_client_id(request)):
            raise HTTPException(429, "too many questions; try again in a minute")
        try:
            return await svc().ask(req)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/leaderboard")
    def leaderboard() -> dict[str, Any]:
        board = svc().leaderboard()
        if board is None:
            raise HTTPException(404, "no benchmark results yet; run `rag-arena bench run`")
        return board

    @app.get("/api/chunks/{chunk_id:path}", response_model=ChunkOut)
    async def chunk(chunk_id: str) -> ChunkOut:
        set_id = chunk_id.split(":", 1)[0]
        try:
            backend = await asyncio.to_thread(svc().runtime.backend, set_id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        row = backend.chunks.get(chunk_id)
        if row is None:
            raise HTTPException(404, f"no chunk {chunk_id!r}")
        return ChunkOut(
            id=row.id,
            rank=0,
            score=0.0,
            doc_id=row.document_id,
            doc_title=row.doc_title,
            heading_path=row.heading_path,
            url=row.url,
            text=row.text,
        )

    return app
