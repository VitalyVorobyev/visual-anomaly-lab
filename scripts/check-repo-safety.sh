#!/usr/bin/env bash
# Guard against committing private data. Run before every commit/push.
# Fails if tracked/staged files include privatedata, BMP images, or files > 5 MB.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
fail=0

if git ls-files --cached | grep -iE 'privatedata|\.bmp$'; then
  echo "ERROR: private data or BMP images are tracked/staged (see above)." >&2
  fail=1
fi

while IFS= read -r f; do
  [ -f "$f" ] || continue
  size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
  if [ "$size" -gt 5242880 ]; then
    echo "ERROR: $f exceeds 5 MB ($size bytes)." >&2
    fail=1
  fi
done < <(git ls-files --cached)

if [ "$fail" -ne 0 ]; then
  echo "Repo safety check FAILED." >&2
  exit 1
fi
echo "Repo safety check passed."
