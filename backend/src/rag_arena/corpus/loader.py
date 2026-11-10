"""Turn the raw FastAPI docs into processed documents.

Processing does three things and nothing else:

1. Inlines code samples referenced by include directives (`{* ../../docs_src/a.py hl[3] *}`,
   `{!> ../../docs_src/a.py!}`), because much of the docs' content lives in those files.
2. Strips explicit heading anchors (`## Title { #title }`) from the text and keeps them as
   section metadata, so chunks can link to the live docs.
3. Records every heading's offset, level, title, anchor and path.

The processed text is the canonical document: chunk offsets and eval evidence offsets point
into it.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from rag_arena.corpus.markdown import clean_title, iter_lines, slugify

DOCS_SUBDIR = "docs/en/docs"
INCLUDE_BASE = "docs/en"  # include paths like ../../docs_src/x.py are relative to this
SITE_URL = "https://fastapi.tiangolo.com"

# Generated, empty or non-documentation pages. See docs/plan.md for the reasons.
EXCLUDED_FILES = frozenset(
    {
        "release-notes.md",
        "fastapi-people.md",
        "external-links.md",
        "management.md",
        "newsletter.md",
        "_llm-test.md",
        "agent-banner.md",
        "translation-banner.md",
    }
)
EXCLUDED_DIRS = ("reference/",)

_INCLUDE_STAR = re.compile(r"\{\*\s+(\.\./\S+)((?:\s+[^*]*?)?)\s*\*\}")
_LINE_SPEC = re.compile(r"\bln\[([\d,:\s]+)\]")
_INCLUDE_BANG = re.compile(r"\{!>?\s*(\.\./\S+?)\s*!\}")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

_LANG_BY_SUFFIX = {
    ".py": "python",
    ".html": "html",
    ".js": "javascript",
    ".ts": "typescript",
    ".json": "json",
    ".toml": "toml",
    ".txt": "text",
    ".sh": "bash",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".css": "css",
}


@dataclass(frozen=True)
class Section:
    start: int
    level: int
    title: str
    anchor: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class Document:
    id: str  # e.g. "tutorial/path-params"
    path: str  # source path relative to docs/en/docs, e.g. "tutorial/path-params.md"
    title: str
    url: str
    text: str
    sections: tuple[Section, ...] = field(default_factory=tuple)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def section_at(self, offset: int) -> Section | None:
        """The section that contains `offset` (the last heading at or before it)."""
        starts = [s.start for s in self.sections]
        i = bisect.bisect_right(starts, offset) - 1
        return self.sections[i] if i >= 0 else None

    def url_for(self, section: Section | None) -> str:
        if section is None or section.level == 1:
            return self.url
        return f"{self.url}#{section.anchor}"


def is_excluded(rel_path: str) -> bool:
    return rel_path in EXCLUDED_FILES or rel_path.startswith(EXCLUDED_DIRS)


def doc_url(rel_path: str) -> str:
    p = PurePosixPath(rel_path)
    if p.name == "index.md":
        parent = str(p.parent)
        return f"{SITE_URL}/" if parent == "." else f"{SITE_URL}/{parent}/"
    return f"{SITE_URL}/{p.with_suffix('')}/"


def doc_id(rel_path: str) -> str:
    return str(PurePosixPath(rel_path).with_suffix(""))


class IncludeError(RuntimeError):
    pass


def _resolve_include(root: Path, include_path: str) -> Path:
    target = (root / INCLUDE_BASE / include_path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise IncludeError(f"include escapes the corpus root: {include_path}")
    if not target.is_file():
        raise IncludeError(f"include not found: {include_path}")
    return target


def _include_target(line: str) -> tuple[str, str] | None:
    """(path, options) for an include directive line, else None."""
    stripped = line.strip()
    m = _INCLUDE_STAR.fullmatch(stripped)
    if m:
        return m.group(1), m.group(2).strip()
    m = _INCLUDE_BANG.fullmatch(stripped)
    return (m.group(1), "") if m else None


def select_lines(code: str, options: str) -> str:
    """Apply an `ln[6,19:20]` option (1-based, inclusive ranges); other options are ignored.

    Gaps between selected ranges are marked with a `...` line, as the docs site renders them.
    """
    m = _LINE_SPEC.search(options)
    if not m:
        return code
    lines = code.split("\n")
    out: list[str] = []
    last = 0
    for part in m.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        lo_s, _, hi_s = part.partition(":")
        lo = int(lo_s)
        hi = int(hi_s) if hi_s else lo
        if out and lo > last + 1:
            out.append("...")
        out.extend(lines[lo - 1 : hi])
        last = hi
    return "\n".join(out)


def process_markdown(raw: str, root: Path) -> tuple[str, list[Section]]:
    """Inline includes and strip heading anchors. Returns (text, sections)."""
    out: list[str] = []
    pos = 0
    sections: list[Section] = []
    stack: list[Section] = []

    def emit(s: str) -> None:
        nonlocal pos
        out.append(s)
        pos += len(s)

    for line in iter_lines(raw):
        include = _include_target(line.text)
        if include is not None:
            include_path, options = include
            target = _resolve_include(root, include_path)
            code = select_lines(target.read_text(encoding="utf-8").rstrip("\n"), options)
            if line.in_fence:
                emit(code + "\n")
            else:
                lang = _LANG_BY_SUFFIX.get(target.suffix, "")
                emit(f"```{lang}\n{code}\n```\n")
            continue
        m = None if line.in_fence else _HEADING.match(line.text)
        if m:
            title, anchor = clean_title(m.group(2))
            level = len(m.group(1))
            # Drop the "{ #anchor }" suffix but keep the heading text as written.
            visible = re.sub(r"\s*\{\s*#[\w-]+\s*\}\s*$", "", m.group(2))
            while stack and stack[-1].level >= level:
                stack.pop()
            path = (*(s.title for s in stack), title)
            section = Section(pos, level, title, anchor or slugify(title), path)
            stack.append(section)
            sections.append(section)
            emit(f"{m.group(1)} {visible}\n")
            continue
        emit(line.text + "\n")
    return "".join(out), sections


def load_document(root: Path, rel_path: str) -> Document:
    raw = (root / DOCS_SUBDIR / rel_path).read_text(encoding="utf-8")
    text, sections = process_markdown(raw, root)
    h1 = next((s for s in sections if s.level == 1), None)
    title = h1.title if h1 else PurePosixPath(rel_path).stem.replace("-", " ").capitalize()
    return Document(
        id=doc_id(rel_path),
        path=rel_path,
        title=title,
        url=doc_url(rel_path),
        text=text,
        sections=tuple(sections),
    )


def list_doc_paths(root: Path) -> list[str]:
    base = root / DOCS_SUBDIR
    paths = sorted(p.relative_to(base).as_posix() for p in base.rglob("*.md"))
    return [p for p in paths if not is_excluded(p)]


def load_corpus(root: Path, paths: Iterable[str] | None = None) -> list[Document]:
    docs = [load_document(root, p) for p in (paths or list_doc_paths(root))]
    return [d for d in docs if d.text.strip()]
