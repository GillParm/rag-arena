"""Small markdown helpers that understand code fences, which is all the chunkers need."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_ANCHOR = re.compile(r"\s*\{\s*#([\w-]+)\s*\}\s*$")
_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Line:
    start: int  # offset of the first character of the line
    text: str  # without the trailing newline
    in_fence: bool  # inside a fenced code block (fence lines themselves count as inside)


@dataclass(frozen=True)
class Heading:
    start: int  # offset of the '#' that opens the heading line
    level: int
    title: str
    anchor: str


def iter_lines(text: str) -> Iterator[Line]:
    """Yield each line with its offset and whether it sits inside a code fence."""
    in_fence = False
    fence_marker = ""
    pos = 0
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\n").rstrip("\r")
        m = _FENCE.match(line)
        if m:
            marker = m.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
                yield Line(pos, line, True)
            elif marker == fence_marker:
                yield Line(pos, line, True)
                in_fence = False
            else:
                yield Line(pos, line, True)
        else:
            yield Line(pos, line, in_fence)
        pos += len(raw)


def slugify(title: str) -> str:
    s = _TAG.sub("", title).lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"[\s_]+", "-", s).strip("-")


def clean_title(raw_title: str) -> tuple[str, str | None]:
    """Split '# Title { #anchor }' into ('Title', 'anchor'); strip inline HTML and code ticks."""
    anchor = None
    m = _ANCHOR.search(raw_title)
    if m:
        anchor = m.group(1)
        raw_title = raw_title[: m.start()]
    title = _TAG.sub("", raw_title).replace("`", "").strip()
    title = re.sub(r"\s+", " ", title)
    return title, anchor


def find_headings(text: str) -> list[Heading]:
    """ATX headings outside code fences, in document order."""
    out: list[Heading] = []
    for line in iter_lines(text):
        if line.in_fence:
            continue
        m = _HEADING.match(line.text)
        if not m:
            continue
        title, anchor = clean_title(m.group(2))
        if not title:
            continue
        out.append(Heading(line.start, len(m.group(1)), title, anchor or slugify(title)))
    return out


def heading_paths(headings: list[Heading]) -> list[list[Heading]]:
    """For each heading, the chain of headings from the top level down to it."""
    stack: list[Heading] = []
    paths: list[list[Heading]] = []
    for h in headings:
        while stack and stack[-1].level >= h.level:
            stack.pop()
        stack.append(h)
        paths.append(list(stack))
    return paths
