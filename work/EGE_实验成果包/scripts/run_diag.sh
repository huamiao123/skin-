#!/bin/bash
cd /root/EGE-UNet-main
export CUDA_VISIBLE_DEVICES=0

echo "========================================"
echo "B3: Baseline + Box-LF Projection (FIRST)"
echo "========================================"
python3 -c "
import torch, os, sys, random, numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
sys.argv = ['train_diag.py', '--exp', 'B3', '--batch_size', '64']
from train_diag import main
from configs.diag_config import DiagConfig
config = DiagConfig()
config.batch_size = 64
config.set_exp('B3', box_lf=True)
main(config)
" 2>&1 | tee /tmp/diag_B3.log

echo "========================================"
echo "B1: Baseline + Local Point Aux"
echo "========================================"
python3 -c "
import torch, os, sys, random, numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
sys.argv = ['train_diag.py', '--exp', 'B1', '--batch_size', '64']
from train_diag import main
from configs.diag_config import DiagConfig
config = DiagConfig()
config.batch_size = 64
config.set_exp('B1', local_point=True)
main(config)
" 2>&1 | tee /tmp/diag_B1.log

echo "========================================"
echo "B2: Baseline + Consistency"
echo "========================================"
python3 -c "
import torch, os, sys, random, numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
sys.argv = ['train_diag.py', '--exp', 'B2', '--batch_size', '32']
from train_diag import main
from configs.diag_config import DiagConfig
config = DiagConfig()
config.batch_size = 32
config.set_exp('B2', consistency=True)
main(config)
" 2>&1 | tee /tmp/diag_B2.log

echo "========================================"
echo "ALL DIAG EXPERIMENTS COMPLETE"
echo "========================================"
