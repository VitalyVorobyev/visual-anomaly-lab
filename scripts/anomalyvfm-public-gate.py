#!/usr/bin/env -S uv run --project backend --extra dl python
"""Run the paired AnomalyVFM zero-shot public-data quality gate."""

from m11_public_gate import main

if __name__ == "__main__":
    raise SystemExit(main(default_candidate="anomalyvfm_anomalib"))
