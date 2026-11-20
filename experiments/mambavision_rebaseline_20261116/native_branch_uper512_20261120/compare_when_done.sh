#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 7>comparison.lock
flock -n 7
PY=/home/waas/mambavision_stage22/env/bin/python
for ds in ISIC2018 PH2 Waterloo HAM10000; do
  until [[ -s "$ds/final_test.json" ]]; do
    if flock -n queue.lock -c true; then exit 1; fi
    sleep 30
  done
  "$PY" compare.py
done

