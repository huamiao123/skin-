#!/bin/bash
cd /root/EGE-UNet-main
export CUDA_VISIBLE_DEVICES=0
python3 << 'PYEOF' > /tmp/weak_baseline_final.log 2>&1
import torch, os, sys, random, numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
sys.argv = ['train_weak.py']
from train_weak import main
from configs.weak_config import WeakBaseConfig
config = WeakBaseConfig()
config.batch_size = 64
config.weak_mode = 'baseline'
main(config)
PYEOF
