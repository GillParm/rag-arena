from __future__ import annotations

import asyncio

import pytest
from google.genai import errors

from rag_arena.llm.gemini import gemini_retry_after, is_daily_quota
from rag_arena.llm.pricing import cost_usd
from rag_arena.llm.ratelimit import RateLimiter, RetryGiveUpError, backoff_delay, retry_async


def test_cost_uses_list_prices_and_batch_discount() -> None:
    assert cost_usd("gemini-3.5-flash-lite", 1_000_000, 0) == pytest.approx(0.30)
    assert cost_usd("gemini-3.5-flash-lite", 0, 1_000_000) == pytest.approx(2.50)
    assert cost_usd("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)
    assert cost_usd("claude-opus-5-5", 1_000_000, 1_000_000, batch=True) == pytest.approx(12.0)
    with pytest.raises(KeyError, match="no list price"):
        cost_usd("mystery-model", 1, 1)


def test_backoff_is_bounded() -> None:
    for attempt in range(10):
        assert 0 <= backoff_delay(attempt, base=1, cap=5) <= 5


async def test_retry_async_retries_then_succeeds() -> None:
    calls = 0
    slept: list[float] = []

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("busy")
        return "ok"

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    out = await retry_async(flaky, retry_after=lambda e: 1.5, sleep=fake_sleep)
    assert out == "ok"
    assert slept == [1.5, 1.5]


async def test_retry_async_stops_on_fatal_errors_and_gives_up() -> None:
    async def bad() -> None:
        raise ValueError("bad request")

    async def no_sleep(_: float) -> None:
        return None

    with pytest.raises(ValueError, match="bad request"):
        await retry_async(bad, retry_after=lambda e: None, sleep=no_sleep)
    with pytest.raises(RetryGiveUpError):
        await retry_async(bad, retry_after=lambda e: 0.0, max_attempts=3, sleep=no_sleep)


def _api_error(code: int, details: list[dict[str, object]] | None = None) -> errors.APIError:
    body = {"error": {"code": code, "message": "x", "status": "S", "details": details or []}}
    return errors.APIError(code, body)


def test_gemini_retry_classification() -> None:
    assert gemini_retry_after(_api_error(503)) == 0.0
    retry_info: dict[str, object] = {
        "@type": "type.googleapis.com/google.rpc.RetryInfo",
        "retryDelay": "12s",
    }
    assert gemini_retry_after(_api_error(429, [retry_info])) == 12.0
    assert gemini_retry_after(_api_error(400)) is None
    daily: dict[str, object] = {
        "@type": "QuotaFailure",
        "violations": [{"quotaId": "GenerateRequestsPerDay"}],
    }
    assert is_daily_quota(_api_error(429, [daily]))
    assert gemini_retry_after(_api_error(429, [daily])) is None
    assert gemini_retry_after(ValueError("x")) is None


async def test_rate_limiter_caps_requests_per_minute() -> None:
    now = [0.0]
    limiter = RateLimiter(rpm=2, concurrency=5, clock=lambda: now[0])
    async with limiter:
        pass
    async with limiter:
        pass
    third = asyncio.ensure_future(limiter.__aenter__())
    await asyncio.sleep(0.02)
    assert not third.done()  # the window is full until 60 s pass
    now[0] = 61.0
    await asyncio.wait_for(third, timeout=2)
    await limiter.__aexit__(None, None, None)


def test_rate_limiter_rejects_bad_settings() -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(rpm=0, concurrency=1)
