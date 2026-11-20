#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>isic2017.lock
flock -n 9
PY=/home/waas/mv_branch_sm120_20261120/venv/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$PY" -u prepare_isic2017.py
"$PY" -u run_pair.py check
echo "$(date -Is) waiting for 5090 GPU lock, baseline then native_two_branch"
exec 8>../ham10000_window16_20g_20261119/gpu.lock
flock 8
echo "$(date -Is) GPU acquired"
for arm in baseline native_two_branch; do
  mkdir -p "$arm/ISIC2017"
  if [[ ! -s "$arm/ISIC2017/final_test.json" ]]; then
    echo "$(date -Is) start $arm"
    "$PY" -u run_pair.py "$arm" >>"$arm/ISIC2017/train.log" 2>&1
  fi
  test -s "$arm/ISIC2017/final_test.json"
done
"$PY" -u compare.py
echo "$(date -Is) both complete"

