#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
R="$PWD"
W="$(dirname "$R")"
PY=/home/waas/mv_branch_sm120_20261120/venv/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while [[ ! -s "$R/scan.ok" && ! -e "$R/scan.ok" ]]; do sleep 5; done
exec 9>"$W/queue.lock"
flock -n 9
exec 8>"$W/../ham10000_window16_20g_20261119/gpu.lock"
flock -n 8
if [[ ! -e verified.ok ]]; then
  "$PY" -u verify5090.py >verify.log 2>&1
  touch verified.ok
fi
mode=$("$PY" - <<'PY'
import json
x=json.load(open('check5090.json'))['median_step_s']
print('resident' if x['resident']<=1.03*x['stream'] else 'stream')
PY
)
export MV_LOADER="$mode"
echo "$(date -Is) 5090 numerical checks complete, data mode=$mode"
for ds in ISIC2018 PH2 Waterloo HAM10000; do
  mkdir -p "$W/$ds"
  [[ -s "$W/$ds/final_test.json" ]] && continue
  echo "$(date -Is) start/resume $ds"
  if ! "$PY" -u run5090.py "$ds" >> "$W/$ds/train.log" 2>&1; then
    if [[ "$mode" == resident ]] && tail -25 "$W/$ds/train.log" | grep -qi 'out of memory'; then
      echo "$(date -Is) retry $ds with byte storage"
      MV_RESIDENT_BYTES=1 "$PY" -u run5090.py "$ds" >> "$W/$ds/train.log" 2>&1
    else
      echo "$(date -Is) stopped $ds, see train.log"
      exit 1
    fi
  fi
  test -s "$W/$ds/final_test.json"
  "$PY" "$W/compare.py"
done
echo "$(date -Is) completed all 4"

