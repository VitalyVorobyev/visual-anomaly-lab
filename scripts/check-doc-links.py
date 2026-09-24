#!/usr/bin/env python3
"""Reject broken repository-local links, and citations of decision records that do not exist."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
# The book links out to docs/ by URL, since mdBook cannot render a file outside its source.
REPO_BLOB = "https://github.com/VitalyVorobyev/visual-anomaly-lab/blob/main/"
SKIP_FILES = {ROOT / "docs" / "userfeedback.md", ROOT / "docs" / "initial-prompt.md"}


def _markdown_files() -> list[Path]:
    files = [
        ROOT / "README.md",
        *sorted((ROOT / "docs").rglob("*.md")),
        *sorted((ROOT / "book" / "src").rglob("*.md")),
    ]
    return [path for path in files if path not in SKIP_FILES]


def _local_target(source: Path, raw: str) -> Path | None:
    target = raw.strip().strip("<>")
    if target.startswith(REPO_BLOB):
        target = "/" + target.removeprefix(REPO_BLOB)
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    path_text = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not path_text:
        return None
    if path_text.startswith("/"):
        return ROOT / path_text.removeprefix("/")
    return source.parent / path_text


ADR_CITATION = re.compile(r"\bADR-(\d{4})\b")
CITING_ROOTS = (
    "backend/src",
    "backend/tests",
    "frontend/src",
    "docs",
    "book/src",
    "scripts",
    ".claude",
)
CITING_FILES = ("CLAUDE.md", "AGENTS.md", "README.md")


def _stale_adr_citations() -> list[str]:
    """A removed record's citations are repointed in the same change (ADR-0030).

    A record may still name the record it superseded, so docs/adr/ itself is exempt.
    """
    live = {path.name[:4] for path in (ROOT / "docs" / "adr").glob("[0-9][0-9][0-9][0-9]-*.md")}
    tracked = subprocess.run(
        ["git", "ls-files", "--", *CITING_ROOTS, *CITING_FILES],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    failures: list[str] = []
    for name in tracked:
        path = ROOT / name
        if not path.is_file() or path in SKIP_FILES or name.startswith("docs/adr/"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for number in ADR_CITATION.findall(line):
                if number not in live:
                    failures.append(f"{name}:{line_number}: cites removed record ADR-{number}")
    return failures


def main() -> int:
    failures: list[str] = []
    files = _markdown_files()
    for source in files:
        text = source.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for raw in LINK.findall(line):
                target = _local_target(source, raw)
                if target is not None and not target.resolve().exists():
                    failures.append(
                        f"{source.relative_to(ROOT)}:{line_number}: missing local target {raw!r}"
                    )
    failures.extend(_stale_adr_citations())
    if failures:
        print("\n".join(failures))
        return 1
    print(f"documentation links valid across {len(files)} maintained Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
