#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
exec 9>queue.lock
flock -n 9 || exit 0
export OMP_NUM_THREADS=8 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/waas/mambavision_stage22/env/bin/python
"$PY" -u fit_aligned.py >>train.log 2>&1
"$PY" -u score_aligned.py >>score.log 2>&1
