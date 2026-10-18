"""Paths, atomic records and immutable-source helpers for the RIS baseline."""
from pathlib import Path
import sys,hashlib,json
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
sys.path.insert(0,str(ROOT/'runtime'))
sys.path.insert(0,str(PROJECT))
ASSETS=ROOT/'assets'
SEED=42
NAMES=('dinov3_text','vmamba_text','fusion_text','dinov3_fixed_text')

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
