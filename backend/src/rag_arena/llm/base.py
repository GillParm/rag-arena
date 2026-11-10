"""Provider-neutral result types for text generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0  # billed output, including thinking tokens
    thinking_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.thinking_tokens + other.thinking_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
        }


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: Usage
    latency_ms: float  # the successful attempt only; rate-limit waits are excluded
    attempts: int = 1
    finish_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class TextLLM(Protocol):
    model: str

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 1024,
    ) -> LLMResponse: ...
