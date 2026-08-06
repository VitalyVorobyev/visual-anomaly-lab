#!/usr/bin/env bash
# Run the backend on its own, with reload, on the documented development port.
#
# This is the path the whole project leans on: every feature is exercisable from a
# plain browser, curl or pytest without the desktop shell in the way (ADR-0003).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

exec uv run --directory backend uvicorn anomaly_lab.api.app:create_app \
    --factory --reload --port "${ANOMALY_LAB_PORT:-8000}"
