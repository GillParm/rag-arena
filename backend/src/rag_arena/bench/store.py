"""Append-only JSONL record stores keyed by a tuple of fields, for resumable stages."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

Key = tuple[str, ...]


class RecordStore:
    def __init__(self, path: Path, key_fields: Sequence[str]) -> None:
        self.path = path
        self.key_fields = tuple(key_fields)
        self._records: dict[Key, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._records[self.key(rec)] = rec  # a later line replaces an earlier one

    def key(self, record: dict[str, Any]) -> Key:
        return tuple(str(record[f]) for f in self.key_fields)

    def has(self, key: Key) -> bool:
        return key in self._records

    def get(self, key: Key) -> dict[str, Any] | None:
        return self._records.get(key)

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._records[self.key(record)] = record

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._records.values())

    def __len__(self) -> int:
        return len(self._records)
