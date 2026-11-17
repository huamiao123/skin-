#!/bin/bash
set -euo pipefail
cd /home/featurize/work/mambavision_seg_rebaseline_20261116
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
py=/home/waas/mambavision_stage22/env/bin/python
"$py" -u src/smoke_train.py --arm A-64-noaux --micro-batch 2 --resume > reports/smoke_A_resume.log 2>&1
for arm in B-512-noaux C-512-aux
do
 "$py" -u src/smoke_train.py --arm "$arm" --micro-batch 2 --stop-after-epoch 1 > "reports/smoke_$arm.epoch1.log" 2>&1
 "$py" -u src/smoke_train.py --arm "$arm" --micro-batch 2 --resume > "reports/smoke_$arm.resume.log" 2>&1
done
