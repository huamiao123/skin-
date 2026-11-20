"""Dataset-local training with the matched baseline recipe and an ADE20K-initialized UPer512 decoder."""
from pathlib import Path
import csv, fcntl, hashlib, json, os, shutil, sys, time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
HAM=PROJECT/'ham10000_window16_20g_20261119'
MATCHED=PROJECT/'four_dataset_matched_20261119'
FAST=Path('/tmp/uper512_ade_decoder_20261119')/'runs'
sys.path.insert(0,str(HAM))
import train_swin as recipe
from model import SegmentationSystem,load_pretrained_encoder
from fetch_decoder import sha, ID

ARM='uper512_ade'
DATASETS=('ISIC2018','PH2','Waterloo','HAM10000')
BASE_CONFIG=recipe.config
WRITE_JSON=recipe.write_json
def copy(src,dst):
    tmp=dst.with_suffix(dst.suffix+'.copying')
    shutil.copyfile(src,tmp)
    tmp.replace(dst)
class BoundedPrefix(recipe.PrefixCache):
    def __init__(self,n):
        super().__init__(512 if n>1024 else n)

class MappedImages(recipe.HamImages):
    def __init__(self,partitions):
        with (recipe.ROOT/'split.csv').open() as f:rows=list(csv.DictReader(f))
        self.seed=recipe.SEED
        self.images={};self.masks={};self.ids={};self.indices={}
        self.x=np.load(recipe.CACHE/'images.npy',mmap_mode='r')
        self.y=np.load(recipe.CACHE/'masks.npy',mmap_mode='r')
        for part in partitions:
            selected=[r for r in rows if r['split']==part]
            self.ids[part]=[r['image_id'] for r in selected]
            self.indices[part]=np.array([int(r['index']) for r in selected])
            self.images[part]=self.x;self.masks[part]=self.y
        self.mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    def batch(self,split,indices,matrices=None):
        ix=self.indices[split][indices.cpu().numpy()]
        x=torch.from_numpy(self.x[ix]).permute(0,3,1,2).contiguous().cuda().float().div_(255.)
        y=torch.from_numpy(self.y[ix]).unsqueeze(1).to(device='cuda',dtype=torch.float32).div_(255.)
        if matrices is not None:
            grid=F.affine_grid(matrices[indices],x.shape,align_corners=False)
            x=F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
            y=F.grid_sample(y,grid,mode='nearest',padding_mode='zeros',align_corners=False)
        return (x-self.mean)/self.std,y

def get_model(arm):
    assert arm==ARM
    recipe.setup(recipe.SEED)
    model=SegmentationSystem(load_pretrained_encoder(),'B-512-noaux',seed=recipe.SEED)
    for i,level in enumerate(model.encoder.model.levels):
        if i in (2,3):level.window_size=(16,8)[i-2]
        assert all(p.requires_grad==(i in (2,3)) for p in level.parameters())
    head=model.probe.decode_head
    initial=head.state_dict()
    output=('conv_seg.weight','conv_seg.bias')
    assert initial[output[0]].shape[:2]==(1,512)
    saved_bin={k:initial[k].clone() for k in output}
    srcfile=ROOT/'ade20k_decode_head.pth'
    record=json.loads((ROOT/'checkpoint_source.json').read_text())
    assert sha(srcfile)==record['decoder_sha256'] and record['full_sha256']==ID
    source=torch.load(srcfile,map_location='cpu',weights_only=True)
    assert source[output[0]].shape[:2]==(150,512)
    # Resolve MMCV's named norm layer if the local native adapter uses another attribute.
    loaded={}
    for name in initial:
        if name in output:continue
        candidates=(name,name.replace('.norm.','.bn.'),name.replace('.bn.','.norm.'))
        srcname=next((key for key in candidates if key in source),None)
        if srcname is None or initial[name].shape!=source[srcname].shape:raise ValueError((name,srcname,list(source)[:12]))
        loaded[name]=(srcname,source[srcname])
    assert {a for a,v in loaded.values()}==set(source)-set(output)
    head.load_state_dict({**saved_bin,**{k:v for k,(a,v) in loaded.items()}},strict=True)
    state=head.state_dict()
    assert all(torch.equal(state[k],v) for k,(a,v) in loaded.items())
    assert all(torch.equal(state[k],v) for k,v in saved_bin.items())
    assert model.probe.auxiliary_head is None and all(p.requires_grad for p in head.parameters())
    WRITE_JSON(recipe.ROOT/'weight_load.json',dict(verified_tensors=len(loaded),source_tensors=len(source),
               verified_numel=sum(v.numel() for a,v in loaded.values()),decoder_parameters=sum(p.numel() for p in head.parameters()),
               excluded_source=list(output),backbone_weights=recipe.WEIGHT_SHA256,decoder_sha256=record['decoder_sha256'],
               source_full_sha256=ID,trainable_stages=[3,4],windows=[16,8],auxiliary=False))
    print('WEIGHTS VERIFIED',len(loaded),'/',len(source),flush=True)
    return model.cuda()

def main(ds):
    assert ds in DATASETS
    source=HAM if ds=='HAM10000' else MATCHED/ds
    dst=ROOT/ds
    dst.mkdir(exist_ok=True)
    local=FAST/ds; job=local/ARM;job.mkdir(parents=True,exist_ok=True)
    split=source/'split.csv'
    if not (local/'split.csv').exists():(local/'split.csv').symlink_to(split)
    assert (local/'split.csv').resolve()==split.resolve()
    recipe.ROOT=local
    recipe.CACHE=Path('/tmp/ham10000_window16_20261119') if ds=='HAM10000' else Path('/tmp/mv_four_matched_20261119')/ds
    stamp=(json.loads((recipe.CACHE/'ready.json').read_text())['split_sha256'] if ds=='HAM10000' else (recipe.CACHE/'ready').read_text().strip())
    assert stamp==sha(split)
    baseline=json.loads((source/'uper64/config.json').read_text())
    recipe.MAX_EPOCHS=100
    def config(arm):
        c=BASE_CONFIG(arm)
        for k in ('split_sha256','seed','backbone_sha256','batch','windows','patience','lr_encoder','lr_decoder','weight_decay','threshold','encoder_eval','selection'):
            assert c[k]==baseline[k],(k,c[k],baseline[k])
        c.update(dataset=ds,scheduler='cosine to 1% over 100 epochs',prefix_cache='frozen BF16 prefix; training in aligned blocks when needed',
                 adapter_sha256=sha(Path(__file__)),full_ade_decoder_source_sha256=ID,
                 decoder_width=512,auxiliary=False,classifier='new one-logit output',init_scope='ADE20K decoder only',training_epochs=100)
        return c
    recipe.config=config;recipe.make_model=get_model;recipe.PrefixCache=BoundedPrefix
    recipe.HamImages=MappedImages
    last_sync=time.monotonic()
    def record(path,value):
        nonlocal last_sync
        WRITE_JSON(path,value)
        if path.name in ('config.json','first_update.json') or (path.name=='history.json' and (len(value)==1 or len(value)%10==0)):
            WRITE_JSON(dst/path.name,value)
        if path.name=='first_update.json':copy(local/'weight_load.json',dst/'weight_load.json')
        if path.name=='history.json' and len(value)%5==0 and time.monotonic()-last_sync>900:
            best=max(value,key=lambda r:r['val_macro_dice'])['epoch']
            recovery=dst/'recovery';recovery.mkdir(exist_ok=True)
            copy(job/f'best_epoch_{best}.pth',recovery/f'best_epoch_{best}.pth')
            copy(job/'resume.pth',recovery/'resume.pth')
            for f in recovery.glob('best_epoch_*.pth'):
                if f.name!=f'best_epoch_{best}.pth':f.unlink()
            last_sync=time.monotonic()
    recipe.write_json=record
    recovery=dst/'recovery'
    if (recovery/'resume.pth').is_file() and not (job/'resume.pth').is_file() and not (job/'fit_complete.json').is_file():
        for f in recovery.glob('*.pth'):copy(f,job/f.name)
    recipe.fit(ARM)
    result=json.loads((job/'test_result.json').read_text()); run=json.loads((job/'fit_complete.json').read_text())
    history=json.loads((job/'history.json').read_text()); best=run['best_checkpoint']
    with split.open() as f:ids=[r['image_id'] for r in csv.DictReader(f) if r['split']=='test']
    with (job/'test_per_image.csv').open() as f:cases=list(csv.DictReader(f))
    assert ids==[r['image_id'] for r in cases] and len(ids)==result['n']
    assert abs(np.mean([float(r['foreground_dice']) for r in cases])-result['test_macro_dice'])<1e-12
    assert max(history,key=lambda r:r['val_macro_dice'])['epoch']==result['best_epoch']
    assert run['split_sha256']==sha(split) and sha(job/best)==run['best_sha256']==result['best_sha256']
    for name in (best,'config.json','history.json','first_update.json','fit_complete.json','test_per_image.csv'):copy(job/name,dst/name)
    copy(local/'weight_load.json',dst/'weight_load.json')
    assert sha(dst/best)==run['best_sha256']
    copy(job/'test_result.json',dst/'test_result.json')
    print('VERIFIED TEST',ds,result['n'],result['test_macro_dice'],len(history),flush=True)
if __name__=='__main__':main(sys.argv[1])
