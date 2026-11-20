"""Prepare the complete ISIC 2017 Part 1 official split from the existing raw archive copy."""
import collections,csv,hashlib,io,json,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
CACHE=Path('/tmp/isic2017_official_pair_20261120')
MANIFEST=Path('/home/featurize/work/RSI_Implementation_20261003/data_manifests/external_isic2017_manifest.csv')
COUNTS={'train':2000,'val':150,'test':600}
NAMES={'train':'Training','val':'Validation','test':'Test_v2'}
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(path,content):
    tmp=path.with_suffix(path.suffix+'.new');tmp.write_text(content);tmp.replace(path)
def prepare():
    CACHE.mkdir(exist_ok=True)
    raw=list(csv.DictReader(MANIFEST.open()))
    ids={p:{r['image_id'] for r in raw if r['split']==p} for p in COUNTS}
    assert sum(COUNTS.values())==len(raw)==len({r['image_id'] for r in raw})
    assert {p:len(group) for p,group in ids.items()}==COUNTS
    for p in COUNTS:
        members=lambda kind:json.loads((MANIFEST.parent/f'external_isic2017_{p}_{kind}_members.json').read_text())
        photos={Path(f).stem for f in members('images') if f.lower().endswith('.jpg')}
        masks={f.removesuffix('_segmentation.png') for f in members('masks') if f.endswith('_segmentation.png')}
        assert ids[p]==photos==masks,(p,len(photos),len(masks))
    rows=sorted(raw,key=lambda r:(list(COUNTS).index(r['split']),r['image_id']))
    manifest=io.StringIO(newline='');writer=csv.writer(manifest);writer.writerow(['index','image_id','split'])
    for i,r in enumerate(rows):writer.writerow([i,r['image_id'],r['split']])
    data=manifest.getvalue().encode();fingerprint=hashlib.sha256(data).hexdigest()
    split=ROOT/'split.csv'
    if split.exists(): assert split.read_bytes()==data
    else:
        tmp=split.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(split)
    ready=CACHE/'ready.json'
    if ready.exists():
        rec=json.loads(ready.read_text())
        if rec['split_sha256']==fingerprint and (ROOT/'dataset.json').exists() and all((CACHE/f'{k}.npy').is_file() for k in ('images','masks')):return
    im=np.lib.format.open_memmap(CACHE/'images.build.npy',mode='w+',dtype='uint8',shape=(len(rows),256,256,3))
    gt=np.lib.format.open_memmap(CACHE/'masks.build.npy',mode='w+',dtype='uint8',shape=(len(rows),256,256))
    def convert(item):
        i,r=item
        encoded=[]
        for path,k in [(r['image_path'],'rgb_sha256'),(r['mask_path'],'mask_sha256')]:
            b=Path(path).read_bytes();assert hashlib.sha256(b).hexdigest()==r[k],path
            encoded.append(b)
        with Image.open(io.BytesIO(encoded[0])) as x, Image.open(io.BytesIO(encoded[1])) as y:
            assert x.size==y.size==(int(r['width']),int(r['height']))
            im[i]=np.asarray(x.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            a=np.asarray(y.convert('L').resize((256,256),Image.Resampling.NEAREST))
            assert np.all((a==0)|(a==255)) and a.max()>0,r['image_id']
            gt[i]=a
    with ThreadPoolExecutor(max_workers=5) as pool:
        for i,_ in enumerate(pool.map(convert,enumerate(rows)),1):
            if i%500==0:print('verified pairs',i,flush=True)
    im.flush();gt.flush();del im,gt
    for k in ('images','masks'):(CACHE/f'{k}.build.npy').replace(CACHE/f'{k}.npy')
    old=ROOT.parent/'four_dataset_matched_20261119/ISIC2018/split.csv'
    oldrows=list(csv.DictReader(old.open()))
    all18={r['image_id'] for r in oldrows}
    rec=dict(dataset='ISIC2017',official_task='Part 1 lesion segmentation',counts=COUNTS,paired_raw_files_verified=len(rows),
             split_sha256=fingerprint,source_manifest_sha256=sha(MANIFEST),size=256,images_resize='PIL bilinear',masks_resize='PIL nearest',
             image_id_overlap_with_existing_ISIC2018_by_2017_split={p:len(v&all18) for p,v in ids.items()},
             urls={p:[f'https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_{s}_{t}.zip' for t in ('Data','Part1_GroundTruth')] for p,s in NAMES.items()})
    atomic(ROOT/'dataset.json',json.dumps(rec,indent=2)+'\n');atomic(ready,json.dumps(rec,indent=2)+'\n')
    print('READY',rec,flush=True)
if __name__=='__main__':prepare()

