"""Complete the historical ISIC2016 seed42 E0/E1/E2 matrix without retraining E1/E2."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
PREVIOUS=PROJECT/'stage9_isic_text'
sys.path.insert(0,str(PROJECT/'stage8_text_baselines/runtime'))
sys.path.insert(0,str(PROJECT/'stage1_visual/runtime'))
sys.path.insert(0,str(PROJECT))
# Shared historical assets are opened read-only; this stage never overwrites them.
ASSETS=PREVIOUS/'assets'
SEED=42
NAMES=('E0_visual','E1_fixed_text','E2_image_text')
SPLITS=('train','val','test')
BATCH=16
EPOCHS=100
FIXED_TEXT='skin lesion'
REUSED={NAMES[1]:'dinov3_fixed_text',NAMES[2]:'dinov3_text'}

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
