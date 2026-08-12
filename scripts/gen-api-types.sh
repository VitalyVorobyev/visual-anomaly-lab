#!/usr/bin/env bash
# Regenerate frontend/src/api/generated.ts from the backend's OpenAPI schema.
#
# ADR-0003 accepts "contract drift is a runtime failure" as a cost of the sidecar
# boundary. Generating the TypeScript types from the live schema removes it: the
# frontend cannot reference a route or field the backend does not serve.
#
# Two things are deliberately isolated here:
#
#   * The backend. A throwaway sidecar is started on an ephemeral port against a
#     temporary data directory, so this script needs nothing running and never
#     touches the real database.
#
#   * The generator's TypeScript. openapi-typescript builds its output with the
#     TypeScript *JS compiler API* and still declares `peerDependencies:
#     {typescript: ^5.x}`. The 7.0 native compiler does not expose that API, so the
#     tool crashes on `ts.factory`. Rather than hold the whole frontend back a major
#     version for one build tool, the generator runs in its own throwaway project
#     with its own TypeScript 5. The emitted file is plain type declarations and is
#     consumed by TypeScript 7 without issue. Drop this isolation once
#     openapi-typescript supports TS 7.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

OUTPUT="$REPO_ROOT/frontend/src/api/generated.ts"
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
OPENAPI_READY=0
for _ in $(seq 1 120); do
    if curl -sSf "http://127.0.0.1:$PORT/openapi.json" -o "$WORK_DIR/openapi.json" 2>/dev/null; then
        OPENAPI_READY=1
        break
    fi
    if ! kill -0 "$UV_PID" 2>/dev/null; then
        break
    fi
    sleep 0.25
done

if [ "$OPENAPI_READY" -ne 1 ]; then
    echo "ERROR: the backend announced port $PORT but never accepted the OpenAPI request." >&2
    cat "$WORK_DIR/backend.log" >&2
    exit 1
fi

echo "Running the generator in an isolated project..."
GEN_DIR="$WORK_DIR/codegen"
mkdir -p "$GEN_DIR"
echo '{"name":"anomaly-lab-codegen","private":true,"version":"0.0.0"}' >"$GEN_DIR/package.json"
(
    cd "$GEN_DIR"
    bun add --silent openapi-typescript@^7 typescript@^5 >/dev/null 2>&1
    bunx openapi-typescript "$WORK_DIR/openapi.json" --output "$OUTPUT"
)

echo "Done. Review the diff -- it is the API contract changing."
