"""Claude calls that return JSON via structured outputs, in sync or Message Batches mode.

Every call's token usage and list-price cost is appended to a spend ledger so the project's
Anthropic budget can be audited (`rag-arena spend`).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from rag_arena.llm.base import Usage
from rag_arena.llm.pricing import cost_usd

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class ClaudeRequest:
    custom_id: str
    system: str
    user: str
    schema: dict[str, Any]
    max_tokens: int = 4000


@dataclass
class ClaudeResult:
    custom_id: str
    data: dict[str, Any] | None
    error: str | None
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    stop_reason: str | None = None
    mode: str = "sync"

    def to_dict(self) -> dict[str, Any]:
        return {
            "custom_id": self.custom_id,
            "data": self.data,
            "error": self.error,
            "usage": self.usage.to_dict(),
            "model": self.model,
            "stop_reason": self.stop_reason,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClaudeResult:
        return cls(
            custom_id=d["custom_id"],
            data=d["data"],
            error=d["error"],
            usage=Usage(**d["usage"]),
            model=d.get("model"),
            stop_reason=d.get("stop_reason"),
            mode=d.get("mode", "sync"),
        )


class SpendLedger:
    """Append-only JSONL of every Anthropic call's tokens and cost."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(
        self, *, purpose: str, model: str, usage: Usage, batch: bool, custom_id: str
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "purpose": purpose,
            "model": model,
            "mode": "batch" if batch else "sync",
            "custom_id": custom_id,
            **usage.to_dict(),
            "cost_usd": round(
                cost_usd(model, usage.input_tokens, usage.output_tokens, batch=batch), 6
            ),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def total_usd(self) -> float:
        return sum(float(r["cost_usd"]) for r in self.rows())


def _usage(message: Any) -> Usage:
    u = message.usage
    return Usage(input_tokens=int(u.input_tokens or 0), output_tokens=int(u.output_tokens or 0))


def parse_message(custom_id: str, message: Any, mode: str) -> ClaudeResult:
    """Turn a Messages API response into a ClaudeResult, checking stop_reason first."""
    usage = _usage(message)
    stop = message.stop_reason
    if stop == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        return ClaudeResult(
            custom_id, None, f"refusal ({category})", usage, message.model, stop, mode
        )
    if stop == "max_tokens":
        return ClaudeResult(custom_id, None, "hit max_tokens", usage, message.model, stop, mode)
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return ClaudeResult(
            custom_id, None, f"invalid JSON: {exc}", usage, message.model, stop, mode
        )
    if not isinstance(data, dict):
        return ClaudeResult(
            custom_id, None, "JSON is not an object", usage, message.model, stop, mode
        )
    return ClaudeResult(custom_id, data, None, usage, message.model, stop, mode)


class ResultCache:
    """JSONL cache of ClaudeResults keyed by custom_id; the last line for an id wins."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, ClaudeResult]:
        if not self.path.exists():
            return {}
        out: dict[str, ClaudeResult] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = ClaudeResult.from_dict(json.loads(line))
                out[r.custom_id] = r
        return out

    def append(self, results: Iterable[ClaudeResult]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


class ClaudeJSON:
    def __init__(
        self,
        api_key: str,
        model: str,
        ledger: SpendLedger,
        *,
        effort: str = "low",
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.ledger = ledger
        self.client = client or anthropic.Anthropic(api_key=api_key, max_retries=4)

    def params(self, req: ClaudeRequest) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": req.max_tokens,
            "system": req.system,
            "messages": [{"role": "user", "content": req.user}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": req.schema},
            },
        }

    def _call_sync(self, req: ClaudeRequest, purpose: str) -> ClaudeResult:
        try:
            # Server-side fallback re-runs a refused request on another model; the result
            # records which model actually answered.
            message = self.client.beta.messages.create(
                **self.params(req), betas=[FALLBACK_BETA], fallbacks="default"
            )
        except anthropic.BadRequestError as exc:
            return ClaudeResult(req.custom_id, None, f"bad request: {exc.message}", mode="sync")
        except anthropic.APIStatusError as exc:
            return ClaudeResult(req.custom_id, None, f"API error {exc.status_code}", mode="sync")
        except anthropic.APIConnectionError as exc:
            return ClaudeResult(req.custom_id, None, f"connection error: {exc}", mode="sync")
        result = parse_message(req.custom_id, message, "sync")
        self.ledger.record(
            purpose=purpose,
            model=self.model,
            usage=result.usage,
            batch=False,
            custom_id=req.custom_id,
        )
        return result

    def run_sync(
        self,
        reqs: list[ClaudeRequest],
        *,
        purpose: str,
        cache: ResultCache,
        concurrency: int = 4,
        on_result: Callable[[ClaudeResult], None] | None = None,
    ) -> dict[str, ClaudeResult]:
        done = cache.load()
        todo = [r for r in reqs if r.custom_id not in done or done[r.custom_id].error]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for result in pool.map(lambda r: self._call_sync(r, purpose), todo):
                cache.append([result])
                done[result.custom_id] = result
                if on_result:
                    on_result(result)
        return {r.custom_id: done[r.custom_id] for r in reqs if r.custom_id in done}

    def run_batch(
        self,
        reqs: list[ClaudeRequest],
        *,
        purpose: str,
        cache: ResultCache,
        poll_seconds: float = 30.0,
        log: Callable[[str], None] = print,
    ) -> dict[str, ClaudeResult]:
        """Submit uncached requests as one Message Batch (half price), wait, cache results.

        The batch id is saved next to the cache so an interrupted run resumes polling the
        same batch instead of paying for a second one.
        """
        done = cache.load()
        todo = [r for r in reqs if r.custom_id not in done or done[r.custom_id].error]
        state_path = cache.path.with_suffix(".batch.json")
        if todo:
            batch_id = None
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if set(state["custom_ids"]) >= {r.custom_id for r in todo}:
                    batch_id = state["batch_id"]
                    log(f"resuming batch {batch_id}")
            if batch_id is None:
                batch = self.client.messages.batches.create(
                    requests=[
                        Request(
                            custom_id=r.custom_id,
                            params=MessageCreateParamsNonStreaming(**self.params(r)),  # type: ignore[typeddict-item]
                        )
                        for r in todo
                    ]
                )
                batch_id = batch.id
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(
                    json.dumps({"batch_id": batch_id, "custom_ids": [r.custom_id for r in todo]}),
                    encoding="utf-8",
                )
                log(f"submitted batch {batch_id} with {len(todo)} requests")
            while True:
                status = self.client.messages.batches.retrieve(batch_id)
                if status.processing_status == "ended":
                    break
                counts = status.request_counts
                log(f"batch {batch_id}: {counts.processing} processing, {counts.succeeded} done")
                time.sleep(poll_seconds)
            wanted = {r.custom_id for r in todo}
            fresh: list[ClaudeResult] = []
            for item in self.client.messages.batches.results(batch_id):
                if item.custom_id not in wanted:
                    continue
                if item.result.type == "succeeded":
                    result = parse_message(item.custom_id, item.result.message, "batch")
                    self.ledger.record(
                        purpose=purpose,
                        model=self.model,
                        usage=result.usage,
                        batch=True,
                        custom_id=item.custom_id,
                    )
                else:
                    result = ClaudeResult(
                        item.custom_id, None, f"batch {item.result.type}", mode="batch"
                    )
                fresh.append(result)
            cache.append(fresh)
            state_path.unlink(missing_ok=True)
            for r in fresh:
                done[r.custom_id] = r
        return {r.custom_id: done[r.custom_id] for r in reqs if r.custom_id in done}
