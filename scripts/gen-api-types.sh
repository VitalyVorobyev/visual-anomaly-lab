#!/usr/bin/env bash
# Regenerate frontend/src/api/generated.ts from the backend's OpenAPI schema.
#
# ADR-0003 accepts "contract drift is a runtime failure" as a cost of the sidecar
# boundary. Generating the TypeScript types from the live schema removes it: the
# frontend cannot reference a route or field the backend does not serve.
#
# Starts its own throwaway backend on an ephemeral port against a temporary data
# directory, so it needs nothing running and never touches the real database.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

WORK_DIR="$(mktemp -d)"
UV_PID=""
SIDECAR_PID=""

cleanup() {
    [ -n "$SIDECAR_PID" ] && kill "$SIDECAR_PID" 2>/dev/null || true
    [ -n "$UV_PID" ] && kill "$UV_PID" 2>/dev/null || true
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

echo "Starting a temporary backend..."
ANOMALY_LAB_PORT=0 ANOMALY_LAB_DATA_DIR="$WORK_DIR/data" \
    uv run --directory backend python -m anomaly_lab.serve \
    >"$WORK_DIR/ready.jsonl" 2>"$WORK_DIR/backend.log" &
UV_PID=$!

# The sidecar announces its port on stdout as a single JSON line (ADR-0009 envelope).
for _ in $(seq 1 120); do
    [ -s "$WORK_DIR/ready.jsonl" ] && break
    sleep 0.25
done

if [ ! -s "$WORK_DIR/ready.jsonl" ]; then
    echo "ERROR: the backend never announced readiness." >&2
    cat "$WORK_DIR/backend.log" >&2
    exit 1
fi

PORT="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["port"])' <"$WORK_DIR/ready.jsonl")"
SIDECAR_PID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["pid"])' <"$WORK_DIR/ready.jsonl")"

echo "Fetching the OpenAPI schema from port $PORT..."
curl -sSf "http://127.0.0.1:$PORT/openapi.json" -o "$WORK_DIR/openapi.json"

echo "Generating frontend/src/api/generated.ts..."
(cd frontend && bunx openapi-typescript "$WORK_DIR/openapi.json" --output src/api/generated.ts)

echo "Done. Review the diff -- it is the API contract changing."
