#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec 9>queue.lock
flock -n 9 || exit 23
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
for arm in f3 uper64 f4
do
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$arm" > active.txt
  /home/waas/mambavision_stage22/env/bin/python -u train.py "$arm" >> "$arm.log" 2>&1
done
/home/waas/mambavision_stage22/env/bin/python -u compare.py > comparison.log 2>&1
printf '%s complete\n' "$(date -u +%FT%TZ)" > active.txt
