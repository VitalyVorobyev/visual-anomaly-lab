#!/usr/bin/env bash
# Guard against committing private data. Run before every commit/push (ADR-0001).
#
# Inspects the index -- exactly what is about to be committed. Install it as a hook
# with scripts/setup-hooks.sh so it runs whether or not anyone remembers to.
#
# Three rules:
#   1. Nothing under privatedata/, ever.
#   2. No photographic or scientific image formats. These only enter this repository
#      by accident, since source images are referenced in place and never copied.
#   3. PNG and SVG are permitted -- app icons and synthetic test fixtures need them --
#      but only while they stay small. A large PNG is a dataset file in disguise.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
fail=0

MAX_BYTES=5242880      # 5 MB: nothing in a source repository has any business being larger
MAX_IMAGE_BYTES=262144 # 256 KB: fixtures and icons are small by construction

# Raster and camera-raw formats. PNG and SVG are deliberately absent; they are
# size-capped below instead.
BANNED_EXTENSIONS='\.(bmp|jpe?g|tiff?|gif|webp|avif|heic|heif|raw|dng|nef|cr2|cr3|arw|orf|rw2|pgm|ppm|pnm|npy|npz)$'

if git ls-files --cached | grep -iE 'privatedata'; then
  echo "ERROR: paths under privatedata/ are tracked/staged (see above)." >&2
  fail=1
fi

if git ls-files --cached | grep -iE "$BANNED_EXTENSIONS"; then
  echo "ERROR: image or array data is tracked/staged (see above)." >&2
  echo "       Source images are referenced in place, never committed. Test fixtures" >&2
  echo "       are small synthetic PNGs." >&2
  fail=1
fi

while IFS= read -r f; do
  [ -f "$f" ] || continue
  size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")

  if [ "$size" -gt "$MAX_BYTES" ]; then
    echo "ERROR: $f exceeds 5 MB ($size bytes)." >&2
    fail=1
    continue
  fi

  # Lower-cased with tr rather than ${f,,}: macOS ships bash 3.2, which has no such
  # parameter expansion, and this script must run on the machine it protects.
  lower=$(printf '%s' "$f" | tr '[:upper:]' '[:lower:]')
  case "$lower" in
    *.png | *.svg)
      if [ "$size" -gt "$MAX_IMAGE_BYTES" ]; then
        echo "ERROR: $f is a $size byte image; fixtures and icons must stay under 256 KB." >&2
        fail=1
      fi
      ;;
  esac
done < <(git ls-files --cached)

if [ "$fail" -ne 0 ]; then
  echo "Repo safety check FAILED." >&2
  exit 1
fi
echo "Repo safety check passed."
