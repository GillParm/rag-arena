from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_arena.api.app import SlidingWindowLimiter, create_app
from rag_arena.api.service import ArenaService
from rag_arena.config import Settings
from rag_arena.db.repo import ChunkSetInfo
from rag_arena.llm.base import TextLLM, Usage
from rag_arena.llm.fake import FakeLLM
from rag_arena.retrieval.backend import SearchBackend
from rag_arena.retrieval.rerank import OverlapReranker, Reranker
from rag_arena.retrieval.rewrite import Rewriter, RewriteResult
from tests.conftest import CHUNKS, make_backend


class StaticRewriter:
    async def rewrite(self, question: str) -> RewriteResult:
        return RewriteResult(["dependency injection"], Usage(30, 10), 5.0, "gemini-3.5-flash-lite")


class FakeRuntime:
    def __init__(self, llm: TextLLM | None) -> None:
        texts = {"heading:" + k.split(":", 1)[1]: v for k, v in CHUNKS.items()}
        self._backends = {"heading": make_backend(texts, set_id="heading")}
        self._llm = llm

    def backend(self, set_id: str) -> SearchBackend:
        if set_id not in self._backends:
            raise LookupError(f"chunk set {set_id!r} is not indexed")
        return self._backends[set_id]

    def chunk_sets(self) -> list[ChunkSetInfo]:
        return [ChunkSetInfo("heading", "heading", {"max_size": 1600}, "hash-64", 64, len(CHUNKS))]

    @property
    def reranker(self) -> Reranker:
        return OverlapReranker()

    @property
    def rewriter(self) -> Rewriter:
        return StaticRewriter()

    @property
    def llm(self) -> TextLLM:
        assert self._llm is not None
        return self._llm


def _client(tmp_path: Path, *, llm: TextLLM | None = None, per_minute: int = 20) -> TestClient:
    board = tmp_path / "summary.json"
    board.write_text(json.dumps({"headline_chunk_set": "heading", "headline": []}))
    service = ArenaService(
        FakeRuntime(llm),
        generation_enabled=llm is not None,
        generator_model="gemini-3.5-flash-lite",
        embedding_model="hash-64",
        reranker_model="overlap",
        corpus="test",
        leaderboard_path=board,
    )
    settings = Settings(_env_file=None, ask_rate_limit_per_minute=per_minute)
    return TestClient(create_app(settings, service))


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    llm = FakeLLM(lambda _s, _p: "Use Depends [1].", model="gemini-3.5-flash-lite")
    return _client(tmp_path, llm=llm)


def test_health(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_config_lists_pipelines_and_chunk_sets(client: TestClient) -> None:
    body = client.get("/api/config").json()
    assert [p["id"] for p in body["pipelines"]] == [
        "bm25",
        "dense",
        "hybrid",
        "hybrid_rerank",
        "hybrid_rewrite",
    ]
    assert body["default_chunk_set"] == "heading"
    assert body["generation_enabled"] is True


def test_ask_returns_every_pipeline_side_by_side(client: TestClient) -> None:
    resp = client.post("/api/ask", json={"question": "  how do I declare dependencies? "})
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"] == "how do I declare dependencies?"
    assert body["chunk_set"] == "heading"
    results = {r["pipeline"]: r for r in body["results"]}
    assert set(results) == {"bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rewrite"}
    hybrid = results["hybrid"]
    assert hybrid["answer"]["text"] == "Use Depends [1]."
    assert hybrid["answer"]["citations"] == [hybrid["chunks"][0]["id"]]
    assert hybrid["chunks"][0]["cited"] is True
    assert hybrid["chunks"][0]["in_context"] is True
    assert hybrid["cost_usd"] > 0
    assert hybrid["latency"]["total_ms"] >= hybrid["latency"]["retrieval_ms"]
    assert results["hybrid_rewrite"]["rewrites"] == ["dependency injection"]
    assert results["hybrid_rewrite"]["input_tokens"] > hybrid["input_tokens"]


def test_ask_can_select_pipelines_and_skip_generation(client: TestClient) -> None:
    body = client.post(
        "/api/ask", json={"question": "cors origins", "pipelines": ["bm25"], "generate": False}
    ).json()
    assert [r["pipeline"] for r in body["results"]] == ["bm25"]
    assert body["results"][0]["answer"] is None
    assert body["results"][0]["cost_usd"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "x"},
        {"question": "a" * 501},
        {"question": "fine question", "pipelines": ["colbert"]},
        {"question": "fine question", "pipelines": []},
    ],
)
def test_ask_validates_input(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/api/ask", json=payload).status_code == 422


def test_unknown_chunk_set_is_a_404(client: TestClient) -> None:
    resp = client.post("/api/ask", json={"question": "cors", "chunk_set": "nope"})
    assert resp.status_code == 404


def test_without_a_generator_key_the_api_still_retrieves(tmp_path: Path) -> None:
    client = _client(tmp_path, llm=None)
    body = client.post("/api/ask", json={"question": "background tasks"}).json()
    assert body["generation_enabled"] is False
    pipelines = [r["pipeline"] for r in body["results"]]
    assert "hybrid_rewrite" not in pipelines
    assert all(r["answer"] is None and r["chunks"] for r in body["results"])


def test_generation_errors_are_reported_per_pipeline(tmp_path: Path) -> None:
    def boom(_s: str, _p: str) -> str:
        raise RuntimeError("quota")

    client = _client(tmp_path, llm=FakeLLM(boom))
    body = client.post("/api/ask", json={"question": "cors", "pipelines": ["bm25"]}).json()
    result = body["results"][0]
    assert result["answer"] is None
    assert "generation failed: quota" in result["error"]
    assert result["chunks"]


def test_rate_limit(tmp_path: Path) -> None:
    client = _client(tmp_path, llm=None, per_minute=2)
    payload = {"question": "cors", "pipelines": ["bm25"]}
    assert client.post("/api/ask", json=payload).status_code == 200
    assert client.post("/api/ask", json=payload).status_code == 200
    assert client.post("/api/ask", json=payload).status_code == 429


def test_sliding_window_limiter_frees_slots_after_a_minute() -> None:
    limiter = SlidingWindowLimiter(1)
    assert limiter.allow("a", now=0)
    assert not limiter.allow("a", now=30)
    assert limiter.allow("b", now=30)
    assert limiter.allow("a", now=61)


def test_leaderboard_and_chunk_lookup(client: TestClient) -> None:
    assert client.get("/api/leaderboard").json()["headline_chunk_set"] == "heading"
    chunk = client.get("/api/chunks/heading:deps:0").json()
    assert chunk["text"].startswith("Dependencies are declared")
    assert client.get("/api/chunks/heading:missing:0").status_code == 404
    assert client.get("/api/chunks/fixed:deps:0").status_code == 404


def test_missing_leaderboard_is_a_404(tmp_path: Path) -> None:
    client = _client(tmp_path, llm=None)
    (tmp_path / "summary.json").unlink()
    assert client.get("/api/leaderboard").status_code == 404
