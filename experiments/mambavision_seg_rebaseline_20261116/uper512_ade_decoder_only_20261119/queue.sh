#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
TMP=/tmp/uper512_ade_decoder_20261119
PY=/home/waas/mambavision_stage22/env/bin/python
exec 9>"$ROOT/queue.lock"
flock -n 9
exec 8>"$ROOT/../ham10000_window16_20g_20261119/gpu.lock"
flock -n 8
export PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMP"
for ds in ISIC2018 PH2 Waterloo HAM10000; do
  if [[ -s "$ds/test_result.json" ]]; then continue; fi
  mkdir -p "$ds"
  echo "$(date -Is) start $ds"
  "$PY" -u train.py "$ds" >>"$TMP/$ds.log" 2>&1 || { echo "$(date -Is) ERROR $ds"; exit 1; }
  test -s "$ds/test_result.json"
  echo "$(date -Is) complete $ds"
done
echo "$(date -Is) COMPLETE all four"
