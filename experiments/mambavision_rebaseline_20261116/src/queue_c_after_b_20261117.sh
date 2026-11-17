#!/bin/bash
set -euo pipefail
cd /home/featurize/work/mambavision_seg_rebaseline_20261116
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
out=reports/C_queue_20261117.status
printf 'waiting_for_B pid=9201\n' > "$out"
while kill -0 9201 2>/dev/null; do sleep 15; done
/home/waas/mambavision_stage22/env/bin/python -c 'import json;from pathlib import Path; d=Path("results/formal_mb64/B-512-noaux"); h=json.loads((d/"history.json").read_text());s=json.loads((d/"status.json").read_text());assert len(h)==100 and h[-1]["epoch"]==100 and s["state"]=="complete" and (d/"best_dice.pth").is_file(), "B not successfully complete"' || { printf 'blocked: B incomplete\n' > "$out"; exit 1; }
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]; then printf 'blocked: GPU in use\n' > "$out"; exit 1; fi
if [ -e results/formal_mb64/C-512-aux ]; then printf 'blocked: C output already exists\n' > "$out"; exit 1; fi
printf 'running C\n' > "$out"
if /home/waas/mambavision_stage22/env/bin/python -u src/train_mb64.py --arm C-512-aux --micro-batch 64 > reports/formal_mb64_C_20261117.log 2>&1; then
 printf 'C complete; test evaluation not done\n' > "$out"
else
 rc=$?
 printf 'C failed exit=%s; inspect log\n' "$rc" > "$out"
 exit "$rc"
fi
