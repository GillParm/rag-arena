from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag_arena.llm.claude import (
    ClaudeJSON,
    ClaudeRequest,
    ResultCache,
    SpendLedger,
    parse_message,
)
from tests.fakes import FakeAnthropic, fake_message

SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "required": []}


def _claude(tmp_path: Path, reply: Any) -> tuple[ClaudeJSON, FakeAnthropic, SpendLedger]:
    fake = FakeAnthropic(reply)
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    client = ClaudeJSON("unused", "claude-opus-5-5", ledger, client=fake)  # type: ignore[arg-type]
    return client, fake, ledger


def _reqs(n: int) -> list[ClaudeRequest]:
    return [ClaudeRequest(f"r{i}", "sys", f"user {i}", SCHEMA, 500) for i in range(n)]


def test_parse_message_checks_stop_reason_first() -> None:
    assert parse_message("a", fake_message({"x": 1}), "sync").data == {"x": 1}
    refused = parse_message("a", fake_message({"x": 1}, stop="refusal"), "sync")
    assert refused.data is None
    assert refused.error == "refusal (cyber)"
    assert (
        parse_message("a", fake_message("{", stop="max_tokens"), "sync").error == "hit max_tokens"
    )
    assert "invalid JSON" in (parse_message("a", fake_message("nope"), "sync").error or "")
    assert parse_message("a", fake_message("[1]"), "sync").error == "JSON is not an object"


def test_params_use_structured_outputs_and_effort(tmp_path: Path) -> None:
    client, _, _ = _claude(tmp_path, lambda p: {})
    params = client.params(_reqs(1)[0])
    assert params["output_config"]["effort"] == "low"
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in params


def test_sync_calls_are_cached_and_recorded_in_the_ledger(tmp_path: Path) -> None:
    client, fake, ledger = _claude(tmp_path, lambda p: {"echo": p["messages"][0]["content"]})
    cache = ResultCache(tmp_path / "cache.jsonl")
    out = client.run_sync(_reqs(3), purpose="test", cache=cache)
    assert out["r2"].data == {"echo": "user 2"}
    assert len(fake.sync_calls) == 3
    assert fake.sync_calls[0]["fallbacks"] == "default"
    # Re-running costs nothing: every result comes from the cache.
    client.run_sync(_reqs(3), purpose="test", cache=cache)
    assert len(fake.sync_calls) == 3
    rows = ledger.rows()
    assert len(rows) == 3
    assert rows[0]["mode"] == "sync"
    assert ledger.total_usd() == pytest.approx(3 * (100 * 4 + 50 * 20) / 1_000_000)


def test_batch_mode_submits_once_and_prices_at_half(tmp_path: Path) -> None:
    client, fake, ledger = _claude(tmp_path, lambda p: {"ok": True})
    cache = ResultCache(tmp_path / "cache.jsonl")
    out = client.run_batch(
        _reqs(4), purpose="test", cache=cache, poll_seconds=0, log=lambda _: None
    )
    assert all(r.data == {"ok": True} for r in out.values())
    assert len(fake.batches_created) == 1
    assert not (tmp_path / "cache.batch.json").exists()
    client.run_batch(_reqs(4), purpose="test", cache=cache, poll_seconds=0, log=lambda _: None)
    assert len(fake.batches_created) == 1
    assert ledger.rows()[0]["mode"] == "batch"
    assert ledger.total_usd() == pytest.approx(4 * (100 * 4 + 50 * 20) / 2 / 1_000_000)


def test_failed_results_are_retried_on_the_next_run(tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky(_: dict[str, Any]) -> Any:
        calls["n"] += 1
        return "not json" if calls["n"] == 1 else {"ok": 1}

    client, _, _ = _claude(tmp_path, flaky)
    cache = ResultCache(tmp_path / "c.jsonl")
    first = client.run_sync(_reqs(1), purpose="t", cache=cache)
    assert first["r0"].error is not None
    second = client.run_sync(_reqs(1), purpose="t", cache=cache)
    assert second["r0"].data == {"ok": 1}
