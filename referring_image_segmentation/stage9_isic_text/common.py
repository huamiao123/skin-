"""Independent ISIC text experiment; historical stages remain read-only."""
from pathlib import Path
import sys, json, hashlib
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'stage8_text_baselines/runtime'))
sys.path.insert(0, str(PROJECT / 'stage1_visual/runtime'))
sys.path.insert(0, str(PROJECT))
ASSETS = ROOT / 'assets'
SEED = 42
NAMES = ('dinov3_text', 'vmamba_text', 'fusion_text', 'dinov3_fixed_text')
SPLITS = ('train', 'val', 'test')
EPOCHS = 100
BATCH = 16
FIXED_TEXT = 'skin lesion'

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''): h.update(chunk)
    return h.hexdigest()
