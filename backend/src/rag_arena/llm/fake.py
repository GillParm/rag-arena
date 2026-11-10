"""A scripted LLM for offline tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rag_arena.llm.base import LLMResponse, Usage


class FakeLLM:
    def __init__(self, reply: Callable[[str, str], str], model: str = "fake-llm") -> None:
        self.model = model
        self._reply = reply
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append({"system": system, "prompt": prompt, "json_schema": json_schema})
        text = self._reply(system, prompt)
        usage = Usage(input_tokens=len((system + prompt).split()), output_tokens=len(text.split()))
        return LLMResponse(text=text, model=self.model, usage=usage, latency_ms=1.0)
