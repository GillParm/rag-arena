from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from rag_arena.corpus.download import LicenceError, check_mit_licence, extract_corpus
from rag_arena.corpus.loader import (
    Document,
    doc_id,
    doc_url,
    is_excluded,
    list_doc_paths,
    load_document,
    select_lines,
)
from rag_arena.corpus.manifest import build_manifest, diff_manifest


def test_excluded_pages_are_skipped(mini_root: Path) -> None:
    assert list_doc_paths(mini_root) == ["index.md", "tutorial/first-steps.md"]
    assert is_excluded("release-notes.md")
    assert is_excluded("reference/apirouter.md")
    assert not is_excluded("tutorial/body.md")


def test_doc_urls_and_ids() -> None:
    assert doc_url("index.md") == "https://fastapi.tiangolo.com/"
    assert doc_url("tutorial/index.md") == "https://fastapi.tiangolo.com/tutorial/"
    assert (
        doc_url("tutorial/first-steps.md") == "https://fastapi.tiangolo.com/tutorial/first-steps/"
    )
    assert doc_id("tutorial/first-steps.md") == "tutorial/first-steps"


def test_includes_are_inlined_with_a_fence_only_outside_fences(mini_root: Path) -> None:
    doc = load_document(mini_root, "tutorial/first-steps.md")
    assert "{*" not in doc.text
    assert "{!" not in doc.text
    # Outside a fence: wrapped in a python fence.
    assert "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()" in doc.text
    # Inside an existing fence: inserted raw, so there is no nested fence.
    assert "```Python\nfrom fastapi import FastAPI" in doc.text


def test_line_selection_option() -> None:
    code = "a\nb\nc\nd\ne\nf"
    assert select_lines(code, "ln[2:3] hl[2]") == "b\nc"
    assert select_lines(code, "ln[1,4:5]") == "a\n...\nd\ne"
    assert select_lines(code, "hl[2]") == code


def test_heading_anchors_are_stripped_and_kept_as_metadata(mini_root: Path) -> None:
    doc = load_document(mini_root, "tutorial/first-steps.md")
    assert "{ #" not in doc.text
    assert doc.title == "First Steps"
    titles = [(s.level, s.title, s.anchor) for s in doc.sections]
    assert titles == [
        (1, "First Steps", "first-steps"),
        (2, "Run it", "run-it"),
        (3, "Check it", "check-it"),
        (2, "Recap", "recap"),
    ]
    check = doc.sections[2]
    assert check.path == ("First Steps", "Run it", "Check it")
    assert doc.text[check.start :].startswith("### Check it\n")


def test_comment_lines_in_code_are_not_headings(mini_root: Path) -> None:
    doc = load_document(mini_root, "tutorial/first-steps.md")
    assert "    # Return a greeting" in doc.text
    assert all("Return a greeting" not in s.title for s in doc.sections)


def test_section_lookup_by_offset(mini_root: Path) -> None:
    doc = load_document(mini_root, "tutorial/first-steps.md")
    offset = doc.text.index("Open your browser")
    section = doc.section_at(offset)
    assert section is not None
    assert section.title == "Check it"
    assert doc.url_for(section).endswith("/tutorial/first-steps/#check-it")
    assert doc.url_for(doc.section_at(0)) == doc.url


def test_mit_licence_check(mini_root: Path) -> None:
    holder = check_mit_licence((mini_root / "LICENSE").read_text())
    assert holder == "Copyright (c) 2018 Example Author"
    with pytest.raises(LicenceError):
        check_mit_licence("Apache License, Version 2.0")


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_extract_keeps_only_corpus_paths(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz", pax_headers={"comment": "abc123"}) as tar:
        _add(tar, "fastapi-1.0/LICENSE", b"MIT")
        _add(tar, "fastapi-1.0/docs/en/docs/index.md", b"# Hi")
        _add(tar, "fastapi-1.0/docs/es/docs/index.md", b"# Hola")
        _add(tar, "fastapi-1.0/docs_src/a.py", b"x = 1")
        _add(tar, "fastapi-1.0/tests/test_a.py", b"")
    dest = tmp_path / "out"
    count, commit = extract_corpus(archive, dest)
    assert count == 3
    assert commit == "abc123"
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == [
        "LICENSE",
        "docs/en/docs/index.md",
        "docs_src/a.py",
    ]


def test_manifest_detects_changes(mini_docs: list[Document]) -> None:
    manifest = build_manifest(mini_docs, ref="1.0", commit=None, licence="x")
    assert manifest["documents"] == 2
    assert diff_manifest(manifest, mini_docs).ok
    changed = [
        Document(d.id, d.path, d.title, d.url, d.text + "!", d.sections) for d in mini_docs[:1]
    ]
    diff = diff_manifest(manifest, changed + mini_docs[1:2][:0])
    assert diff.changed == [mini_docs[0].id]
    assert diff.removed == [mini_docs[1].id]
