#!/usr/bin/env bash
# Wrapper to generate a minimal dataset under tests/data/minimal/20231109
set -euo pipefail

ROOT=${1:-tests/data/20231109}
OUT=${2:-tests/data/minimal/20231109}
WINDOW_SECONDS=${WINDOW_SECONDS:-300}
DELETE_ORIGINALS=${DELETE_ORIGINALS:-0}

mkdir -p "$OUT"

echo "Computing median time for $ROOT"
CENTER_ISO=$("$(pwd)"/tests/data/scripts/compute_median_time.py "$ROOT") || { echo "failed to compute median"; exit 1; }
echo "Center time: $CENTER_ISO"

CENTER_EPOCH=$(python3 - <<PY
from datetime import datetime, timezone
print(int(datetime.fromisoformat('$CENTER_ISO').replace(tzinfo=timezone.utc).timestamp()))
PY
)

echo "Using window +/- ${WINDOW_SECONDS}s"

for f in $(find "$ROOT" -type f); do
  rel=${f#${ROOT}/}
  outpath="$OUT/$rel"
  mkdir -p "$(dirname "$outpath")"
  case "$f" in
    *.h5)
      echo "Shrinking HDF5: $rel"
      python3 tests/data/scripts/shrink_h5.py "$f" "$outpath" --center $CENTER_EPOCH --window $WINDOW_SECONDS || cp "$f" "$outpath"
      ;;
    *.nc)
      echo "Shrinking netCDF: $rel"
      python3 tests/data/scripts/shrink_netcdf.py "$f" "$outpath" --center "$CENTER_ISO" --window $WINDOW_SECONDS || cp "$f" "$outpath"
      ;;
    *.23o|*.23p|*.23b|*.o|*.p|*.b)
      echo "Shrinking RINEX: $rel"
      python3 tests/data/scripts/shrink_rinex.py "$f" "$outpath" --center $CENTER_EPOCH --window $WINDOW_SECONDS || cp "$f" "$outpath"
      ;;
    *)
      # images, archives, other small files: copy
      cp "$f" "$outpath"
      ;;
  esac
  if [ "$DELETE_ORIGINALS" = "1" ]; then
    echo "(User set DELETE_ORIGINALS=1) Deleting original $f"
    rm -f "$f"
  fi
done

echo "Minimal dataset created under $OUT"
