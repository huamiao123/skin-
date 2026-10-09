#!/bin/bash
cd /root/Pytorch-UNet-master
export CUDA_VISIBLE_DEVICES=0
python3 train_unet_isic.py
echo "=== UNET TRAINING COMPLETE ==="
