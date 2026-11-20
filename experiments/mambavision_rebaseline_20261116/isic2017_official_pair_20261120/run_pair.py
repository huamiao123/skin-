"""Use the established official full-finetuning trainer with the ISIC 2017 Part 1 split."""
import argparse,json,sys
from pathlib import Path
from types import FunctionType
import numpy as np

ROOT=Path(__file__).resolve().parent
BRANCH=ROOT.parent/'native_branch_uper512_20261120'
sys.path.insert(0,str(BRANCH/'rtx5090'))
from resident_images import ResidentImages,official
from branch_model import load_branch_model,method_config
import train_branch
from prepare_isic2017 import CACHE,COUNTS

DS='ISIC2017'
def report():
    rec=json.loads((ROOT/'dataset.json').read_text())
    initial=json.loads((official.ROOT/'initialization.json').read_text())
    assert rec['counts']==COUNTS and rec==json.loads((CACHE/'ready.json').read_text())
    assert rec['split_sha256']==official.sha(ROOT/'split.csv')
    assert np.load(CACHE/'images.npy',mmap_mode='r').shape==(2750,256,256,3)
    assert np.load(CACHE/'masks.npy',mmap_mode='r').shape==(2750,256,256)
    assert official.sha(official.ROOT/'common_initial.pth')==initial['initial_sha256']
    return rec,initial
def data(ds,parts):
    assert ds==DS and set(parts)<=set(COUNTS)
    official.recipe.ROOT=ROOT
    official.recipe.CACHE=CACHE
    out=ResidentImages(parts)
    assert all(len(out.ids[p])==COUNTS[p] for p in parts)
    return out
def bind(fn,**changes):
    return FunctionType(fn.__code__,dict(fn.__globals__,**changes),fn.__name__,fn.__defaults__,fn.__closure__)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('arm',choices=['baseline','native_two_branch','check'])
    arm=ap.parse_args().arm
    rec,initial=report()
    if arm=='check':
        print('VERIFIED split and immutable common initialization',rec['split_sha256'],initial['initial_sha256'],flush=True)
        return
    dest=ROOT/arm
    dest.mkdir(exist_ok=True)
    splits={DS:rec}
    result=bind(train_branch.result,ROOT=dest,SPLITS=splits,dataset=data)
    # Epoch-zero checks from the historical four-dataset study are replaced here by the verified common weight digest.
    # Test images are first evaluated only after validation checkpoint selection.
    model=official.load_model if arm=='baseline' else load_branch_model
    method=(lambda:dict(name='official_full_uper512_fcn')) if arm=='baseline' else method_config
    run=bind(train_branch.run,OUT=dest,SPLITS=splits,DATASETS=(),dataset=data,
             load_branch_model=model,method_config=method,result=result)
    run(DS,64)
if __name__=='__main__':main()

