"""LLM query rewriting for the hybrid_rewrite pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_arena.llm.base import TextLLM, Usage

REWRITE_SYSTEM = """\
You rewrite questions into search queries for a search engine over the FastAPI documentation.
The engine combines keyword search and semantic search, so varied wording helps it find the
right passage."""

REWRITE_PROMPT = """\
Question: {question}

Write exactly {n} alternative search queries that would find the documentation passage that
answers this question:
1. The question rephrased with the FastAPI terms the docs would likely use (class, function,
   parameter or concept names).
2. A short keyword query of 3 to 8 words.
3. One sentence written the way the documentation itself might state the answer.

Return JSON with a "rewrites" array of {n} strings."""

REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"rewrites": {"type": "array", "items": {"type": "string"}}},
    "required": ["rewrites"],
}


@dataclass(frozen=True)
class RewriteResult:
    rewrites: list[str]
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str | None = None


class Rewriter(Protocol):
    async def rewrite(self, question: str) -> RewriteResult: ...


def parse_rewrites(text: str, question: str, n: int) -> list[str]:
    """Pull rewrites from the model's JSON; drop blanks and copies of the question."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    raw = data.get("rewrites", []) if isinstance(data, dict) else []
    seen = {question.strip().lower()}
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        q = " ".join(item.split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:n]


class LLMRewriter:
    def __init__(self, llm: TextLLM, n: int = 3) -> None:
        self.llm = llm
        self.n = n

    async def rewrite(self, question: str) -> RewriteResult:
        resp = await self.llm.generate(
            system=REWRITE_SYSTEM,
            prompt=REWRITE_PROMPT.format(question=question, n=self.n),
            json_schema=REWRITE_SCHEMA,
            max_output_tokens=400,
        )
        return RewriteResult(
            rewrites=parse_rewrites(resp.text, question, self.n),
            usage=resp.usage,
            latency_ms=resp.latency_ms,
            model=resp.model,
        )
