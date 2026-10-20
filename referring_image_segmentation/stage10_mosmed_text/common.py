"""Independent MosMedData+ experiment; historical stages are read-only."""
from pathlib import Path
import sys, json, hashlib

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'stage8_text_baselines/runtime'))
sys.path.insert(0, str(PROJECT / 'stage1_visual/runtime'))
sys.path.insert(0, str(PROJECT))
ASSETS = ROOT / 'assets'
RAW = Path('/home/featurize/rsi_data/benchmarks/mosmed_plus')
NAMES = ('E0_visual', 'E1_fixed_text', 'E2_image_text')
SEEDS = (42, 43, 44)
SPLITS = ('train', 'val', 'test')
FIXED_TEXT = 'lung infection'
BATCH = 64
PILOT_EPOCHS = 50

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()
