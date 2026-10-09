#!/bin/bash
cd /root/EGE-UNet-main
export CUDA_VISIBLE_DEVICES=0
python3 -c "
import torch, os, sys, random, numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
sys.argv = ['train.py']
from train import main
from configs.config_ege_dual_difflr import ege_dual_difflr_config
config = ege_dual_difflr_config
main(config)
" 2>&1 | tee /tmp/ege_dual_difflr.log
echo "=== EGE-DUAL DIFFLR TRAINING COMPLETE ==="
