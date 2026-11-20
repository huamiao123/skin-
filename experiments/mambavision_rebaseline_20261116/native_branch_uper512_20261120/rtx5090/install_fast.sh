#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python - <<'PY'
import psutil
for p in psutil.process_iter(['cmdline']):
 a=p.info.get('cmdline') or []
 if a[:4]==['/home/waas/mv_branch_sm120_20261120/venv/bin/python','-m','pip','install'] and 'https://mirrors.aliyun.com/pypi/simple' in a: p.terminate()
PY
python -u fast_wheels.py
P=/home/waas/mv_branch_sm120_20261120/venv/bin/python
"$P" -m pip install --no-cache-dir --no-index --no-deps /home/waas/mv_branch_sm120_20261120/wheels/*.whl
"$P" -m pip install --disable-pip-version-check --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'sympy>=1.13.3' mmcv-lite==2.1.0 mmengine==0.10.7 timm==1.0.22 ninja transformers==4.44.2
"$P" -c "import torch,torchvision;print(torch.__version__,torchvision.__version__,torch.cuda.get_arch_list());print(torch.randn(5,device='cuda').sum())"
touch env.ok

