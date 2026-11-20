#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
here="$PWD"
while [[ ! -e env.ok ]]; do sleep 5; done
E=/home/waas/mv_branch_sm120_20261120
P="$E/venv/bin/python"
export PATH="$E/cuda/bin:$E/venv/bin:$PATH" CUDA_HOME="$E/cuda" TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4
SITE=$("$P" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
includes=$("$P" -c 'import sysconfig,pathlib;print(":".join(map(str,(pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia").glob("*/include"))))')
export CPATH="$includes:$E/cuda/include"
mkdir -p "$E/extension"
cd "$E/extension"
"$P" "$here/setup_scan.py" build_ext --inplace > "$here/compiler.log" 2>&1
for binary in selective_scan_cuda*.so; do ln -sf "$PWD/$binary" "$SITE/$binary"; done
ln -sfn "$E/mamba-v1.2.2/mamba_ssm" "$SITE/mamba_ssm"
"$P" -c 'import torch,selective_scan_cuda;from mamba_ssm.ops.selective_scan_interface import selective_scan_fn;print(torch.cuda.get_device_name(0),selective_scan_cuda.__file__)'
touch "$here/scan.ok"

