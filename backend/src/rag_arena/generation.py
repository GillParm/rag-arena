"""Answer generation: the same prompt and model for every pipeline."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_arena.db.repo import ChunkRow
from rag_arena.llm.base import TextLLM, Usage

CONTEXT_CHUNKS = 5
ABSTAIN_TEXT = "I couldn't find this in the documentation provided."

GENERATE_SYSTEM = f"""\
You answer questions about FastAPI using only the numbered documentation passages provided.

- Base every statement on the passages and cite the passages that support it in square
  brackets, like [2] or [1][3].
- If the passages do not contain the answer, reply with exactly: "{ABSTAIN_TEXT}"
  Do not answer from general knowledge.
- Be concise: at most five sentences. Include a short code snippet only when the question asks
  how to write something."""

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Answer:
    text: str
    context_ids: list[str]  # chunk ids shown to the model, in passage order
    citations: list[str]  # chunk ids the answer cites, in first-mention order
    abstained: bool
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "context_ids": self.context_ids,
            "citations": self.citations,
            "abstained": self.abstained,
            "usage": self.usage.to_dict(),
            "latency_ms": round(self.latency_ms, 3),
            "model": self.model,
            "finish_reason": self.finish_reason,
        }


def format_passages(chunks: Sequence[ChunkRow]) -> str:
    blocks = []
    for n, c in enumerate(chunks, start=1):
        source = " > ".join([c.doc_title, *c.heading_path[1:]]) if c.heading_path else c.doc_title
        blocks.append(f"[{n}] {source}\n{c.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: Sequence[ChunkRow]) -> str:
    return f"Passages:\n\n{format_passages(chunks)}\n\nQuestion: {question}"


def parse_citations(text: str, n_passages: int) -> list[int]:
    """Passage numbers cited as [2], [1][3] or [1, 3], in first-mention order, in range."""
    seen: list[int] = []
    for m in _CITATION.finditer(text):
        for part in m.group(1).split(","):
            num = int(part)
            if 1 <= num <= n_passages and num not in seen:
                seen.append(num)
    return seen


def is_abstention(text: str) -> bool:
    return ABSTAIN_TEXT.lower().rstrip(".") in text.lower()


async def generate_answer(llm: TextLLM, question: str, chunks: Sequence[ChunkRow]) -> Answer:
    resp = await llm.generate(
        system=GENERATE_SYSTEM, prompt=build_prompt(question, chunks), max_output_tokens=1024
    )
    text = resp.text.strip()
    cited = parse_citations(text, len(chunks))
    return Answer(
        text=text,
        context_ids=[c.id for c in chunks],
        citations=[chunks[n - 1].id for n in cited],
        abstained=is_abstention(text),
        usage=resp.usage,
        latency_ms=resp.latency_ms,
        model=resp.model,
        finish_reason=resp.finish_reason,
    )
