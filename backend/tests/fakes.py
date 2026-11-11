"""Stand-ins for the Anthropic client used by offline tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


def fake_message(data: Any, *, stop: str = "end_turn", tokens: tuple[int, int] = (100, 50)) -> Any:
    text = data if isinstance(data, str) else json.dumps(data)
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop,
        stop_details=SimpleNamespace(category="cyber") if stop == "refusal" else None,
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]),
        model="claude-opus-5-5",
    )


class FakeAnthropic:
    """Answers every request with `reply(params)`; batches finish after one poll."""

    def __init__(self, reply: Callable[[dict[str, Any]], Any]) -> None:
        self.reply = reply
        self.sync_calls: list[dict[str, Any]] = []
        self.batches_created: list[list[Any]] = []
        self._results: dict[str, list[Any]] = {}
        self._polls = 0
        outer = self

        class _BetaMessages:
            def create(self, **params: Any) -> Any:
                outer.sync_calls.append(params)
                return fake_message(outer.reply(params))

        class _Batches:
            def create(self, requests: list[Any]) -> Any:
                batch_id = f"batch_{len(outer.batches_created)}"
                outer.batches_created.append(requests)
                outer._results[batch_id] = [
                    SimpleNamespace(
                        custom_id=r["custom_id"],
                        result=SimpleNamespace(
                            type="succeeded", message=fake_message(outer.reply(dict(r["params"])))
                        ),
                    )
                    for r in requests
                ]
                return SimpleNamespace(id=batch_id)

            def retrieve(self, batch_id: str) -> Any:
                outer._polls += 1
                status = "ended" if outer._polls > 1 else "in_progress"
                counts = SimpleNamespace(processing=0, succeeded=len(outer._results[batch_id]))
                return SimpleNamespace(processing_status=status, request_counts=counts)

            def results(self, batch_id: str) -> list[Any]:
                return outer._results[batch_id]

        self.beta = SimpleNamespace(messages=_BetaMessages())
        self.messages = SimpleNamespace(batches=_Batches())
