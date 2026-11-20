#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>queue.lock
flock -n 9
exec 8>../ham10000_window16_20g_20261119/gpu.lock
flock -n 8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/waas/mambavision_stage22/env/bin/python
if [[ ! -s /tmp/mv_four_matched_20261119/ISIC2018/ready || ! -s /tmp/mv_four_matched_20261119/PH2/ready || ! -s /tmp/mv_four_matched_20261119/Waterloo/ready || ! -s /tmp/ham10000_window16_20261119/ready.json ]]; then
 "$PY" ../four_dataset_matched_20261119/prepare_matched.py ISIC2018 PH2 Waterloo
 "$PY" ../ham10000_window16_20261119/prepare.py
fi
"$PY" -u evaluate.py >>epoch0.log 2>&1
if [[ ! -s batch_probe.json ]]; then
 for m in 64 32 16 8; do
  if "$PY" -u fit.py probe --micro "$m" >>probe.log 2>&1; then break
  else rc=$?; if [[ $rc -ne 42 ]]; then echo "Probe failed ($rc)"; exit "$rc"; fi; echo "CUDA OOM at microbatch $m"; fi
 done
 test -s batch_probe.json
fi
for ds in ISIC2018 PH2 Waterloo HAM10000; do
 if [[ -s "$ds/final_test.json" ]]; then continue; fi
 echo "$(date -Is) start $ds"
 "$PY" -u fit.py "$ds" >>"$ds/train.log" 2>&1
 test -s "$ds/final_test.json"
 echo "$(date -Is) completed $ds"
done
echo "$(date -Is) all four complete"
