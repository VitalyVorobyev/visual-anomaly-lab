#!/usr/bin/env python3
"""Reject broken repository-local links in the user book and maintained docs."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP_FILES = {ROOT / "docs" / "userfeedback.md", ROOT / "docs" / "initial-prompt.md"}


def _markdown_files() -> list[Path]:
    files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    return [path for path in files if path not in SKIP_FILES]


def _local_target(source: Path, raw: str) -> Path | None:
    target = raw.strip().strip("<>")
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    path_text = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not path_text:
        return None
    if path_text.startswith("/"):
        return ROOT / path_text.removeprefix("/")
    return source.parent / path_text


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
    if failures:
        print("\n".join(failures))
        return 1
    print(f"documentation links valid across {len(files)} maintained Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
