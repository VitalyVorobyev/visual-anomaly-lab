#!/usr/bin/env -S uv run --project backend --extra dl python
"""Run the paired DINO patch-memory public-data quality gate."""

from m11_public_gate import main

if __name__ == "__main__":
    raise SystemExit(main(default_candidate="dino_memory"))
