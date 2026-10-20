"""ISIC2016 token-level text interaction before the decoder; seed42 only."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
PREVIOUS=PROJECT/'stage9_isic_text'
CONTROLS=PROJECT/'stage11_isic_text_controls'
sys.path.insert(0,str(PROJECT/'stage8_text_baselines/runtime'))
sys.path.insert(0,str(PROJECT/'stage1_visual/runtime'))
sys.path.insert(0,str(PROJECT))
# Historical image/sentence assets are reused read-only; word tokens live locally.
ASSETS=PREVIOUS/'assets'
TOKEN_ASSETS=ROOT/'assets'
SEED=42
NAMES=('predecoder_fixed_text','predecoder_image_text')
SPLITS=('train','val','test')
BATCH=16
EPOCHS=100
FIXED_TEXT='skin lesion'
RESIDUAL_SCALE=.1

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
