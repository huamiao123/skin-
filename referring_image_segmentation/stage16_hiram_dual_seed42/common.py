"""Shared utilities for the HiRAM transfer experiment."""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'stage8_text_baselines' / 'runtime'))
sys.path.insert(0, str(PROJECT / 'stage14_mambavision_s_isic2018_seed42' / 'runtime'))


def setup():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])


def tree_hash(value):
    """Device-independent digest used for exact checkpoint restore audits."""
    h = hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            v = x.detach().cpu().contiguous()
            h.update(str((str(v.dtype), tuple(v.shape))).encode())
            h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((str(x.dtype), x.shape)).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x,key=repr):
                h.update(repr(k).encode())
                add(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for item in x:
                add(item)
        else:
            h.update(repr(x).encode())
    add(value)
    return h.hexdigest()
