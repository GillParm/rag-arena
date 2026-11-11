"""Gemini client with a client-side rate limit, backoff on 429/5xx and token accounting."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx2
from google import genai
from google.genai import errors, types

from rag_arena.llm.base import LLMResponse, Usage
from rag_arena.llm.ratelimit import RateLimiter, retry_async

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

log = logging.getLogger("rag_arena.llm.gemini")


def _log_retry(attempt: int, exc: Exception, delay: float) -> None:
    code = exc.code if isinstance(exc, errors.APIError) else type(exc).__name__
    log.info("gemini retry %d after %s, waiting %.1fs", attempt, code, delay)


class DailyQuotaExhaustedError(RuntimeError):
    """The free tier's per-day quota is used up; retrying today will not help."""


def _retry_delay_seconds(details: Any) -> float:
    for d in details if isinstance(details, list) else []:
        delay = d.get("retryDelay") if isinstance(d, dict) else None
        if isinstance(delay, str) and (m := re.fullmatch(r"([\d.]+)s", delay)):
            return float(m.group(1))
    return 0.0


def is_daily_quota(exc: errors.APIError) -> bool:
    text = str(exc.details) if exc.details else str(exc)
    return exc.code == 429 and "PerDay" in text


def gemini_retry_after(exc: Exception) -> float | None:
    if isinstance(exc, errors.APIError):
        if is_daily_quota(exc):
            return None
        if exc.code in RETRYABLE_STATUS:
            details = (exc.details or {}).get("error", {}).get("details") if exc.details else None
            return _retry_delay_seconds(details)
        return None
    # Dropped connections, read errors and timeouts on the way to or from the API.
    if isinstance(exc, httpx2.TransportError):
        return 0.0
    return None


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        rpm: int = 30,
        concurrency: int = 4,
        thinking_level: str = "minimal",
        timeout_s: float = 60.0,
        seed: int | None = 7,
    ) -> None:
        self.model = model
        self.thinking_level = thinking_level
        self.seed = seed
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),  # we retry ourselves
            ),
        )
        self.limiter = RateLimiter(rpm, concurrency)

    def _config(
        self, system: str, json_schema: dict[str, Any] | None, max_output_tokens: int
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            seed=self.seed,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(self.thinking_level.upper())
            ),
            response_mime_type="application/json" if json_schema else None,
            response_json_schema=json_schema,
        )

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 1024,
    ) -> LLMResponse:
        config = self._config(system, json_schema, max_output_tokens)
        attempts = 0
        latency_ms = 0.0

        async def once() -> types.GenerateContentResponse:
            nonlocal attempts, latency_ms
            attempts += 1
            async with self.limiter:
                t0 = time.perf_counter()
                resp = await self._client.aio.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                latency_ms = (time.perf_counter() - t0) * 1000
            return resp

        try:
            resp = await retry_async(once, retry_after=gemini_retry_after, on_retry=_log_retry)
        except errors.APIError as exc:
            if is_daily_quota(exc):
                raise DailyQuotaExhaustedError(str(exc)) from exc
            raise
        meta = resp.usage_metadata
        thinking = (meta.thoughts_token_count or 0) if meta else 0
        usage = Usage(
            input_tokens=(meta.prompt_token_count or 0) if meta else 0,
            output_tokens=((meta.candidates_token_count or 0) + thinking) if meta else 0,
            thinking_tokens=thinking,
        )
        finish = None
        if resp.candidates and resp.candidates[0].finish_reason is not None:
            finish = str(resp.candidates[0].finish_reason.value)
        return LLMResponse(
            text=resp.text or "",
            model=self.model,
            usage=usage,
            latency_ms=latency_ms,
            attempts=attempts,
            finish_reason=finish,
        )
