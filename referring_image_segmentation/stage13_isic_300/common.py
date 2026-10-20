"""Five ISIC2016 seed42 runs with a common full 300-epoch schedule."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
PREVIOUS=PROJECT/'stage9_isic_text'
CONTROLS=PROJECT/'stage11_isic_text_controls'
EARLY=PROJECT/'stage12_predecoder_text'
sys.path.insert(0,str(PROJECT/'stage8_text_baselines/runtime'))
sys.path.insert(0,str(PROJECT/'stage1_visual/runtime'))
sys.path.insert(0,str(PROJECT))
ASSETS=PREVIOUS/'assets'
TOKEN_ASSETS=EARLY/'assets'
SEED=42
NAMES=('E0_visual','E1_fixed_text','E2_image_text','predecoder_fixed_text','predecoder_image_text')
PRE_NAMES=NAMES[3:]
FIXED_NAMES=(NAMES[1],NAMES[3])
SPLITS=('train','val','test')
BATCH=16
EPOCHS=300
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
