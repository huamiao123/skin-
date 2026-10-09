#!/bin/bash
cd /root/EGE-UNet-main

echo "=== Starting Baseline Training ==="
python -c "
import torch, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import random, numpy as np
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
" 2>&1 | tee /tmp/weak_baseline_v3.log

echo "=== Baseline Done ==="

echo "=== Starting Full Method Training ==="
python -c "
import torch, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import random, numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
sys.argv = ['train_weak.py']
from train_weak import main
from configs.weak_config import WeakFullConfig
config = WeakFullConfig()
config.batch_size = 32
config.weak_mode = 'full'
main(config)
" 2>&1 | tee /tmp/weak_full_v2.log

echo "=== Full Done ==="
echo "=== BOTH TRAININGS COMPLETE ==="
