#!/usr/bin/env bash
# Check star removers the agreed way: crux's own measurements, no other
# program. For each master: crux's classic starless (the reference), then
# each run's network starless; stars crux still detects afterwards, and
# starless_check.py (change away from stars, light removed on stars).
#
#   tools/ml/eval_starless.sh <crux.exe> <scratch dir> <run dir>... [-- <master>...]
# Default masters: M 17 H (trained on) and the held-out STL NGC 6188 L and
# QHY9 NGC 3324 L.
set -euo pipefail
CRUX=$1; OUT=$2; shift 2
RUNS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do RUNS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
MASTERS=("$@")
if [ ${#MASTERS[@]} -eq 0 ]; then
  MASTERS=(
    "C:/crux-work/2026-10-05_M 17/integrated/M 17/masterLight_M 17_H.xisf"
    "C:/crux-work/2012-04-19 NGC 6188b/integrated/NGC6188/masterLight_NGC6188_LUMINANCE.xisf"
    "C:/crux-work/2010-02-25 QHY9 NGC 3324/integrated/unknown/masterLight_unknown_LUMINANCE.xisf"
  )
fi
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/../../.local/ml/venv/Scripts/python"
mkdir -p "$OUT/fits"
stars() { "$CRUX" info "$1" 2>&1 | grep -o "stars [0-9]* clean" | grep -o "[0-9]*"; }
for m in "${MASTERS[@]}"; do
  k=$(basename "$m" .xisf | sed 's/masterLight_//; s/ /_/g')
  echo "== $k"
  [ -f "$OUT/fits/${k}_orig.fits" ] || "$CRUX" dev export-fits "$m" --out-dir "$OUT/fits" --prefix "${k}_orig_" > /dev/null
  orig="$OUT/fits/${k}_orig_$(basename "$m" .xisf).fits"
  [ -f "$OUT/fits/${k}_classic_$(basename "$m" .xisf).fits" ] || "$CRUX" dev starless "$m" --out-dir "$OUT/fits" --prefix "${k}_classic_" > /dev/null
  classic="$OUT/fits/${k}_classic_$(basename "$m" .xisf).fits"
  echo "  original $(stars "$m") stars, classic fill $(stars "$classic")"
  for r in "${RUNS[@]}"; do
    n=$(basename "$r")
    out="$OUT/${k}_$n.xisf"
    [ -f "$out" ] || "$CRUX" dev denoise "$m" --model "$r/starless.onnx" --out "$out" > /dev/null 2>&1
    "$CRUX" dev export-fits "$out" --out-dir "$OUT/fits" > /dev/null
    echo "  $n: $(stars "$out") stars left"
    (cd "$HERE" && PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore "$PY" starless_check.py "$orig" "$classic" "$OUT/fits/${k}_$n.fits" | tail -3 | sed 's/^/    /')
  done
done
