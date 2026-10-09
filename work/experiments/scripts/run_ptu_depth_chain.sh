#!/bin/bash
cd /root/pure_transformer_unet
export CUDA_VISIBLE_DEVICES=0
python3 train.py --depth3 2 --lr 1e-4 --tag d2 > /root/ptu_d2_run.log 2>&1
echo "=== PTU D2 COMPLETE ===" >> /root/ptu_d2_run.log
python3 train.py --depth3 6 --lr 1e-4 --tag d6 > /root/ptu_d6_run.log 2>&1
echo "=== PTU D6 COMPLETE ===" >> /root/ptu_d6_run.log
