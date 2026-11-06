"""Stage14 paths, fixed protocol and serialization; historical files are read-only."""
from pathlib import Path
import sys, json, hashlib, random, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
# Reuse the historical environment without installing into or modifying it.
for path in (PROJECT, PROJECT/'stage1_visual/runtime', PROJECT/'stage8_text_baselines/runtime', ROOT/'runtime'):
    sys.path.insert(0, str(path))
SEED = 42
CHANNELS = (96,192,384,768)
CRITERIA = ('best_dice','best_loss','last')
HF_REVISION = 'e4f9ea93877fba6f1e2d231fd0f0a391368ad895'
GITHUB_REVISION = '7860a506b2eb844eaaae676f08461ce8c3c26f43'
WEIGHT_SHA256 = '006af39903f2d88c5b455431d087d0fcbf5d5932114aef52542c0d2f0b1ec180'

def dump(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def save_checkpoint(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def setup():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

def tree_hash(value):
    """Deterministic digest of optimizer/scheduler/RNG state, independent of pickle."""
    h=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            x=x.detach().cpu().contiguous();h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((str(x.dtype),x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=repr):h.update(repr(k).encode());add(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for item in x:add(item)
        else:h.update(repr(x).encode())
    add(value);return h.hexdigest()

def verify_frozen_sources():
    p=ROOT/'launch_provenance.json'
    if p.exists():
        source=json.loads(p.read_text())
        for group in ('historical_sha256','new_source_sha256','official_sha256'):
            for path,digest in source[group].items():assert sha(path)==digest,('Changed frozen file',path)
