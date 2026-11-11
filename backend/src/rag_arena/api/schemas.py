"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from rag_arena.retrieval.pipelines import PIPELINES


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    chunk_set: str | None = Field(default=None, max_length=64)
    pipelines: list[str] | None = None
    generate: bool = True

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 3:
            raise ValueError("question is too short")
        return v

    @field_validator("pipelines")
    @classmethod
    def _known(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        unknown = [p for p in v if p not in PIPELINES]
        if unknown:
            raise ValueError(f"unknown pipelines {unknown}; choose from {list(PIPELINES)}")
        if not v:
            raise ValueError("choose at least one pipeline")
        return list(dict.fromkeys(v))


class ChunkOut(BaseModel):
    id: str
    rank: int
    score: float
    doc_id: str
    doc_title: str
    heading_path: list[str]
    url: str
    text: str
    cited: bool = False
    in_context: bool = False


class AnswerOut(BaseModel):
    text: str
    citations: list[str]
    abstained: bool
    model: str | None


class LatencyOut(BaseModel):
    retrieval_ms: float
    generation_ms: float | None
    total_ms: float
    stages: dict[str, float]


class PipelineResult(BaseModel):
    pipeline: str
    label: str
    answer: AnswerOut | None
    error: str | None = None
    chunks: list[ChunkOut]
    rewrites: list[str]
    latency: LatencyOut
    input_tokens: int
    output_tokens: int
    cost_usd: float


class AskResponse(BaseModel):
    question: str
    chunk_set: str
    generation_enabled: bool
    results: list[PipelineResult]


class PipelineInfo(BaseModel):
    id: str
    label: str
    description: str


class ChunkSetInfoOut(BaseModel):
    id: str
    chunker: str
    chunk_count: int
    params: dict[str, Any]


class ConfigResponse(BaseModel):
    pipelines: list[PipelineInfo]
    chunk_sets: list[ChunkSetInfoOut]
    default_chunk_set: str | None
    generation_enabled: bool
    generator_model: str
    embedding_model: str
    reranker_model: str
    corpus: str
