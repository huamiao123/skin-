"""Build disposable 256px arrays from the pre-existing, frozen target splits."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv, json, hashlib, io, os, zipfile, sys
import numpy as np
from PIL import Image
BASE=Path(__file__).resolve().parent
PROJECT=Path('/home/waas/mambavision_stage22/source/referring_image_segmentation')
ORIGINAL=BASE.parent/'uper64_feature_audit_20261119/transfer_20261119/target_training_20261119'
HIST=BASE.parent/'diagnostics/stageC_20261117/data_provenance'
SCRATCH=Path('/tmp/mv_four_matched_20261119')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def persist(p,s):
    if p.exists() and p.read_text()!=s:raise ValueError(f'frozen manifest changed {p}')
    tmp=p.parent/(p.name+'.tmp')
    tmp.write_text(s);tmp.replace(p)
def prepare(ds):
    out=BASE/ds;out.mkdir(exist_ok=True)
    fast=SCRATCH/ds;fast.mkdir(parents=True,exist_ok=True)
    files=[fast/'images.npy',fast/'masks.npy']
    if (fast/'ready').is_file() and (out/'split.csv').is_file() and (fast/'ready').read_text()==sha(out/'split.csv') and all(p.is_file() for p in files):return
    rows=[]
    if ds=='ISIC2018':
        identity=json.loads((PROJECT/'stage45_upernet_decoder/identity_partition.json').read_text())
        split=identity['splits']
        total=sum(map(len,split.values()))
        mapping={(r['split'],r['id']):r for r in json.loads((HIST/'BICUBIC_MASK_ID_MAPPING.json').read_text())}
        audit_file=PROJECT.parent.parent/'local_audit_records/test_assets/test_manifest.json'
        assert sha(audit_file)==identity['audit_sha256']['test_manifest.json']
        audit={r['id']:r for r in json.loads(audit_file.read_text())}
    else:
        source=ORIGINAL/ds
        old=list(csv.DictReader((source/'split.csv').open()))
        spec=json.loads((source/'split.json').read_text())
        assert sha(source/'split.csv')==spec['manifest_sha256']
        total=len(old)
    x=np.lib.format.open_memmap(files[0],mode='w+',dtype=np.uint8,shape=(total,256,256,3))
    y=np.lib.format.open_memmap(files[1],mode='w+',dtype=np.uint8,shape=(total,256,256))
    for part in ('train','val','test'):
        if ds!='ISIC2018':
            rr=[r for r in old if r['split']==part]
            with np.load(source/(part+'.npz'),allow_pickle=False) as z:
                assert [r['id'] for r in rr]==z['ids'].tolist()
                xx=z['images'];yy=z['masks']
                assert yy.dtype==np.bool_
                for r,im,mask in zip(rr,xx,yy):
                    assert hashlib.sha256(im.tobytes()).hexdigest()==r['sha256_rgb']
                    i=len(rows);x[i]=im;y[i]=mask.astype(np.uint8)*255
                    rows.append((i,r['id'],r['group'],part))
        else:
            root=PROJECT/'data/isic2018'/part if part!='test' else PROJECT.parent.parent/'heldout_data/test'
            with ExitStack() as stack:
                archives={k:stack.enter_context(zipfile.ZipFile(HIST/f'ISIC2018_Task1_{k}_GroundTruth.zip')) for k in ('Training','Validation')} if part!='test' else {}
                def example(sid):
                    ip=root/'images'/(sid+'.png');mp=root/'masks'/(sid+'.png')
                    if part=='test':assert sha(ip)==audit[sid]['image_256_sha256'] and sha(mp)==audit[sid]['mask_256_sha256']
                    with Image.open(ip) as im:img=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                    if part=='test':
                        with Image.open(mp) as im:mask=np.asarray(im.convert('L').resize((256,256),Image.Resampling.NEAREST))>=128
                        key=sid
                    else:
                        m=mapping[(part,sid)];assert len(m['official_matches'])==1
                        with Image.open(mp) as legacy:assert hashlib.sha256(legacy.convert('L').tobytes()).hexdigest()==m['raw_pixel_sha256']
                        match=m['official_matches'][0];key=match['official_id'];kind=match['official_split']
                        entry=f'ISIC2018_Task1_{kind}_GroundTruth/{key}_segmentation.png'
                        with Image.open(io.BytesIO(archives[kind].read(entry))) as im:
                            mask=np.asarray(im.convert('L').resize((256,256),Image.Resampling.NEAREST))>=128
                    return key,img,mask
                with ThreadPoolExecutor(6) as ex:
                    for key,img,mask in ex.map(example,split[part]):
                        i=len(rows);x[i]=img;y[i]=mask.astype(np.uint8)*255;rows.append((i,key,'',part))
    assert len(rows)==total and len({r[1] for r in rows})==total
    if ds!='ISIC2018':
        groups={s:{r[2] for r in rows if r[3]==s} for s in ('train','val','test')}
        assert all(not(groups[a]&groups[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    x.flush();y.flush()
    buf=io.StringIO();w=csv.writer(buf,lineterminator='\n');w.writerow(['index','image_id','known_group','split']);w.writerows(rows)
    persist(out/'split.csv',buf.getvalue())
    (fast/'ready').write_text(sha(out/'split.csv'))
    print(ds,{s:sum(r[3]==s for r in rows) for s in ('train','val','test')},sha(out/'split.csv'),flush=True)
if __name__=='__main__':
    for dataset in sys.argv[1:]:prepare(dataset)
