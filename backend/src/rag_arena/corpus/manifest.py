"""A committed record of the processed corpus, so a fresh download can be verified."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_arena.corpus.loader import Document


@dataclass(frozen=True)
class ManifestDiff:
    added: list[str]
    removed: list[str]
    changed: list[str]

    @property
    def ok(self) -> bool:
        return not (self.added or self.removed or self.changed)


def build_manifest(
    docs: list[Document], *, ref: str, commit: str | None, licence: str
) -> dict[str, Any]:
    return {
        "source": "https://github.com/fastapi/fastapi",
        "ref": ref,
        "commit": commit,
        "licence": "MIT",
        "licence_holder": licence,
        "documents": len(docs),
        "total_chars": sum(len(d.text) for d in docs),
        "files": [
            {"id": d.id, "path": d.path, "chars": len(d.text), "sha256": d.sha256}
            for d in sorted(docs, key=lambda d: d.id)
        ],
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def diff_manifest(expected: dict[str, Any], docs: list[Document]) -> ManifestDiff:
    want = {f["id"]: f["sha256"] for f in expected["files"]}
    have = {d.id: d.sha256 for d in docs}
    return ManifestDiff(
        added=sorted(have.keys() - want.keys()),
        removed=sorted(want.keys() - have.keys()),
        changed=sorted(k for k in want.keys() & have.keys() if want[k] != have[k]),
    )
