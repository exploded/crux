#!/usr/bin/env bash
# Check a star corrector the agreed way: crux's own measurements (FWHM and
# eccentricity in a 3 × 3 grid, flux kept) on M 17 H and the two held-out
# masters (STL-11000M NGC 6188 L, QHY9 NGC 3324 L), in both modes.
#
#   tools/ml/eval_sharpen.sh <run dir with sharpen.onnx> <crux.exe> <scratch dir>
set -euo pipefail
RUN=$1; CRUX=$2; OUT=$3
mkdir -p "$OUT/fits"
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/../../.local/ml/venv/Scripts/python"
declare -A R=( [m17]=16 [ngc6188]=6 [qhy9]=16 )
declare -A M=(
  [m17]="C:/crux-work/2026-10-05_M 17/integrated/M 17/masterLight_M 17_H.xisf"
  [ngc6188]="C:/crux-work/2012-04-19 NGC 6188b/integrated/NGC6188/masterLight_NGC6188_LUMINANCE.xisf"
  [qhy9]="C:/crux-work/2010-02-25 QHY9 NGC 3324/integrated/unknown/masterLight_unknown_LUMINANCE.xisf"
)
for k in m17 ngc6188 qhy9; do
  m=${M[$k]}
  echo "== $k"
  cp_list=("$m")
  for c in 0 1; do
    "$CRUX" dev denoise "$m" --model "$RUN/sharpen.onnx" --control $c --out "$OUT/${k}_c$c.xisf" > /dev/null 2>&1
    cp_list+=("$OUT/${k}_c$c.xisf")
  done
  for f in "${cp_list[@]}"; do
    printf '  %-14s ' "$(basename "$f" .xisf | cut -c1-14)"
    "$CRUX" info "$f" 2>&1 | grep -o "FWHM [0-9.]* HFR [0-9.]* ecc [0-9.]*" | tr '\n' ' '
    "$CRUX" info "$f" 2>&1 | grep -A3 "regions" | tail -3 | awk '{printf "| %s %s %s ", $1, $3, $7}'
    echo
  done
  "$CRUX" dev export-fits "${cp_list[@]}" --out-dir "$OUT/fits" --prefix "${k}_" > /dev/null
  names=()
  for f in "${cp_list[@]}"; do names+=("$OUT/fits/${k}_$(basename "$f" .xisf).fits"); done
  (cd "$HERE" && PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore "$PY" starcheck.py "${names[@]}" --r ${R[$k]} | tail -2)
done
