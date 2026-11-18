#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec 8>verification.lock
flock -n 8 || exit 22
exec 9<queue.lock
flock -s 9
test -s comparison.json
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=/home/waas/mambavision_stage22/env/bin/python
"$py" -u verify_best.py
"$py" -u report.py
