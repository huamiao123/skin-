#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>queue.lock
flock -n 9
exec 8>../ham10000_window16_20g_20261119/gpu.lock
flock -n 8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/waas/mambavision_stage22/env/bin/python
if [[ ! -s audit.json ]]; then "$PY" -u audit.py >audit.log 2>&1; fi
for ds in ISIC2018 PH2 Waterloo HAM10000; do
  mkdir -p "$ds"
  if [[ ! -s "$ds/final_test.json" ]]; then
    echo "$(date -Is) start $ds"
    "$PY" -u train_branch.py "$ds" >>"$ds/train.log" 2>&1
    test -s "$ds/final_test.json"
    echo "$(date -Is) completed $ds"
  fi
done
echo "$(date -Is) completed all four"

