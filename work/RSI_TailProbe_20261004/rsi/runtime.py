from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def save_checkpoint(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.pth')
    torch.save(payload, temp); temp.replace(path)


def append_csv(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    if exists:
        with path.open(newline='') as f:
            fields = next(csv.reader(f))
        extra = set(row)-set(fields)
        if extra:
            raise ValueError(f'CSV schema changed for {path}: {extra}')
    else:
        fields = list(row)
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists: w.writeheader()
        w.writerow(row)


def load_config(path=None):
    path = Path(path) if path else PROJECT/'configs/p2_ima_m_v2.yaml'
    with path.open() as f:
        return yaml.safe_load(f)


def set_seed(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(bool(deterministic))


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])


def environment():
    import timm, torchvision, scipy
    return dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                timm=timm.__version__, scipy=scipy.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                cpu_threads=torch.get_num_threads())


def code_provenance():
    paths = sorted(PROJECT.glob('rsi/*.py')) + sorted(PROJECT.glob('configs/*.yaml'))
    hashes = {str(p.relative_to(PROJECT)): sha256_file(p) for p in paths}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return dict(commit=commit, source_sha256=hashes,
                source_bundle_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())
