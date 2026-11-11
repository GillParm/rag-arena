"""Client-side rate limiting and retry with exponential backoff."""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable


class RateLimiter:
    """At most `rpm` request starts in any rolling 60 seconds and `concurrency` in flight."""

    def __init__(self, rpm: int, concurrency: int, *, clock: Callable[[], float] = time.monotonic):
        if rpm <= 0 or concurrency <= 0:
            raise ValueError("rpm and concurrency must be positive")
        self.rpm = rpm
        self._sem = asyncio.Semaphore(concurrency)
        self._starts: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._clock = clock

    async def _wait_for_slot(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                while self._starts and now - self._starts[0] >= 60.0:
                    self._starts.popleft()
                if len(self._starts) < self.rpm:
                    self._starts.append(now)
                    return
                # Sleep in short slices and re-check, so the window math always uses the
                # current clock.
                await asyncio.sleep(min(60.0 - (now - self._starts[0]) + 0.01, 0.25))

    async def __aenter__(self) -> RateLimiter:
        await self._sem.acquire()
        try:
            await self._wait_for_slot()
        except BaseException:
            self._sem.release()
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._sem.release()


class RetryGiveUpError(RuntimeError):
    """Raised when a retryable error persists past the attempt limit."""


def backoff_delay(attempt: int, *, base: float = 2.0, cap: float = 60.0) -> float:
    """Full-jitter exponential backoff: uniform(0, min(cap, base * 2**attempt))."""
    return random.uniform(0, min(cap, base * (2**attempt)))


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    *,
    retry_after: Callable[[Exception], float | None],
    max_attempts: int = 8,
    base: float = 2.0,
    cap: float = 60.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """Call `fn` until it succeeds.

    `retry_after(exc)` returns None for errors that should not be retried, 0 to use
    exponential backoff, or a positive server-suggested delay in seconds.
    """
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:
            hint = retry_after(exc)
            if hint is None:
                raise
            if attempt == max_attempts - 1:
                raise RetryGiveUpError(f"gave up after {max_attempts} attempts: {exc}") from exc
            delay = hint if hint > 0 else backoff_delay(attempt, base=base, cap=cap)
            if on_retry is not None:
                on_retry(attempt + 1, exc, delay)
            await sleep(delay)
    raise AssertionError("unreachable")
