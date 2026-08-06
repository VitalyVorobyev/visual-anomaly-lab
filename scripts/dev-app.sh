#!/usr/bin/env bash
# Run the full desktop app: Tauri shell + Vite dev server + spawned sidecar.
#
# The shell starts its own backend on an ephemeral port, so this does not clash with a
# backend already running from scripts/dev-backend.sh.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)/frontend"

exec bun run tauri dev
