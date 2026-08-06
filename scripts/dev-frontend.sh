#!/usr/bin/env bash
# Run the React app in a browser against a separately started backend.
#
# Pair with scripts/dev-backend.sh in another terminal, then open
# http://localhost:5173. With no backend URL injected, the app falls back to
# http://127.0.0.1:8000 — the port dev-backend.sh uses.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)/frontend"

exec bun run dev
