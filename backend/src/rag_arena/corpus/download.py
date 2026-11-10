"""Download the pinned FastAPI release archive and extract only what the corpus needs."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ARCHIVE_URL = "https://codeload.github.com/fastapi/fastapi/tar.gz/refs/tags/{ref}"

# Members we keep from the archive, relative to its top-level directory. One page includes
# lines from the library source (fastapi/openapi/docs.py), so the package is kept too.
KEEP_PREFIXES = ("docs/en/docs/", "docs_src/", "fastapi/")
KEEP_FILES = ("LICENSE",)

MIT_MARKERS = (
    "MIT License",
    "Permission is hereby granted, free of charge, to any person obtaining a copy",
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND',
)


class LicenceError(RuntimeError):
    """The archive's LICENSE is missing or is not the MIT licence."""


@dataclass(frozen=True)
class DownloadResult:
    ref: str
    root: Path
    archive_sha256: str
    files_extracted: int
    licence_holder: str
    commit: str | None


def corpus_root(raw_dir: Path, ref: str) -> Path:
    return raw_dir / f"fastapi-{ref}"


def check_mit_licence(licence_text: str) -> str:
    """Return the copyright line if the text is the MIT licence, else raise LicenceError."""
    missing = [m for m in MIT_MARKERS if m not in licence_text]
    if missing:
        raise LicenceError(f"LICENSE does not look like MIT; missing: {missing}")
    for line in licence_text.splitlines():
        if line.strip().lower().startswith("copyright"):
            return line.strip()
    raise LicenceError("LICENSE has no copyright line")


def _wanted(member_path: str) -> str | None:
    """Map an archive member to its path inside the corpus root, or None to skip it."""
    parts = PurePosixPath(member_path).parts
    if len(parts) < 2:
        return None
    inner = "/".join(parts[1:])  # drop the "fastapi-<ref>/" prefix
    if inner in KEEP_FILES or inner.startswith(KEEP_PREFIXES):
        return inner
    return None


def extract_corpus(archive: Path, dest: Path) -> tuple[int, str | None]:
    """Extract the wanted members of `archive` into `dest`.

    Returns the number of files and the commit SHA that GitHub stores in the archive's pax
    header (None if absent).
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        members: list[tarfile.TarInfo] = []
        for member in tar.getmembers():
            if not member.isfile():
                continue
            inner = _wanted(member.name)
            if inner is None:
                continue
            member.name = inner
            members.append(member)
        # The "data" filter rejects absolute paths, links outside dest and device files.
        tar.extractall(dest, members=members, filter="data")
        count = len(members)
        commit = tar.pax_headers.get("comment")
    return count, commit


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_archive(ref: str, target: Path, url_template: str = ARCHIVE_URL) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    request = urllib.request.Request(
        url_template.format(ref=ref), headers={"User-Agent": "rag-arena-corpus-download"}
    )
    with urllib.request.urlopen(request, timeout=120) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(target)


def fetch_corpus(raw_dir: Path, ref: str, *, force: bool = False) -> DownloadResult:
    """Download (if needed), extract and licence-check the corpus for `ref`."""
    archive = raw_dir / f"fastapi-{ref}.tar.gz"
    root = corpus_root(raw_dir, ref)
    if force or not archive.exists():
        download_archive(ref, archive)
    files, commit = extract_corpus(archive, root)
    licence = root / "LICENSE"
    if not licence.exists():
        raise LicenceError("the archive has no LICENSE file at its root")
    holder = check_mit_licence(licence.read_text(encoding="utf-8"))
    return DownloadResult(
        ref=ref,
        root=root,
        archive_sha256=sha256_file(archive),
        files_extracted=files,
        licence_holder=holder,
        commit=commit,
    )
