#!/bin/bash
cd /root/pure_transformer_unet
export CUDA_VISIBLE_DEVICES=0
python3 train.py --dual --depth3 4 --t_lr 1e-4 --cnn_lr 1e-3 --tag dual_d4 > /root/ptu_dual_run.log 2>&1
echo "=== PTU-DUAL TRAINING COMPLETE ===" >> /root/ptu_dual_run.log
