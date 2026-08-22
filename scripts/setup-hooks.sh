#!/usr/bin/env bash
# Point git at the repo's versioned hooks.
#
# ADR-0022 notes that the private-data guard "is advisory unless invoked". This makes it
# run on every commit. Git does not track .git/hooks, so the hooks live in .githooks/ and
# core.hooksPath points there — one command, and it survives a fresh clone.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

git config core.hooksPath .githooks
chmod +x .githooks/* scripts/*.sh

echo "Hooks enabled: $(git config core.hooksPath)"
echo "Every commit now runs scripts/check-repo-safety.sh."
