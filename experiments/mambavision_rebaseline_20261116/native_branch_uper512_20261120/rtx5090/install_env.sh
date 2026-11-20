#!/usr/bin/env bash
set -euo pipefail
dir=/home/waas/mv_branch_sm120_20261120
mkdir -p "$dir"
python -m venv --system-site-packages "$dir/venv"
"$dir/venv/bin/python" -m pip install --disable-pip-version-check --no-cache-dir --retries 2 --timeout 60 --index-url https://mirrors.aliyun.com/pypi/simple torch==2.8.0 torchvision==0.23.0 mmcv-lite==2.1.0 mmengine==0.10.7 timm==1.0.22 ninja
"$dir/venv/bin/python" -c "import torch;print(torch.__version__,torch.cuda.get_arch_list());print(torch.randn(5,device='cuda').sum())"

