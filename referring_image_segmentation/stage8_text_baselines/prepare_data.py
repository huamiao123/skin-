"""Recover original RefCOCO fields from pinned metadata; fetch official COCO images."""
from common import *
import io,threading,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from PIL import Image
import requests
import pyarrow.parquet as pq
from pycocotools import mask as mask_api

EXPECTED={'train':(16994,42404,120624),'val':(1500,3811,10834),'testA':(750,1975,5657),'testB':(750,1810,5095)}
SOURCE='jxu124/refcoco';REV='9fba8200c5326e996f789191f095bd464ef1d09e'
FILES=['data/test-00000-of-00001-82af0c1b600890ac.parquet','data/testB-00000-of-00001-60990e4598892dc1.parquet','data/train-00000-of-00001-94431d5f4bd5b93f.parquet','data/validation-00000-of-00001-bfeafdc84ca37aa2.parquet']

def download(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size:return
    for attempt in range(4):
        try:
            with requests.get(url,stream=True,timeout=(15,90)) as r:
                r.raise_for_status()
                with path.with_suffix('.part').open('wb') as f:
                    for block in r.iter_content(1024*1024):f.write(block)
            path.with_suffix('.part').replace(path);return
        except Exception:
            if attempt==3:raise
            time.sleep(attempt+1)

def native_mask(ann,image):
    h,w=image['height'],image['width'];seg=ann['segmentation']
    if isinstance(seg,list):rle=mask_api.merge(mask_api.frPyObjects(seg,h,w))
    elif isinstance(seg['counts'],list):rle=mask_api.frPyObjects(seg,h,w)
    else:rle=seg
    m=mask_api.decode(rle)
    if m.ndim==3:m=np.any(m,axis=2)
    return m.astype(np.uint8)

def geometry(w,h):
    scale=min(256/w,256/h);nw=max(1,round(w*scale));nh=max(1,round(h*scale))
    return dict(width=w,height=h,new_width=nw,new_height=nh,left=(256-nw)//2,top=(256-nh)//2)

def main():
    ASSETS.mkdir(exist_ok=True)
    if (ROOT/'data_manifest.json').exists():
        assert json.loads((ROOT/'data_manifest.json').read_text())['passed'];print('DATA ALREADY READY',flush=True);return
    for f in FILES:download(f'https://huggingface.co/datasets/{SOURCE}/resolve/{REV}/{f}',ASSETS/Path(f).name)
    refs=[];images={};annotations={}
    for file in FILES:
        for row in pq.read_table(ASSETS/Path(file).name).to_pylist():
            ann=json.loads(row['raw_anns']);info=json.loads(row['raw_image_info'])
            if isinstance(ann,list):assert len(ann)==1;ann=ann[0]
            assert ann['id']==row['ann_id'] and ann['image_id']==row['image_id']==info['id']
            if info['id'] in images:assert images[info['id']]==info
            images[info['id']]=info;annotations[ann['id']]=ann
            sentences=row['sentences'];assert {s['sent_id'] for s in sentences}==set(row['sent_ids'])
            refs.append(dict(ref_id=row['ref_id'],ann_id=ann['id'],image_id=info['id'],split=row['split'],sentences=sentences))
    refs.sort(key=lambda x:x['ref_id']);image_list=sorted(images.values(),key=lambda x:x['id'])
    image_index={r['id']:i for i,r in enumerate(image_list)};ann_list=sorted(annotations.values(),key=lambda x:x['id']);ann_index={a['id']:i for i,a in enumerate(ann_list)}
    stats={};rows={};sets={}
    for split,want in EXPECTED.items():
        rr=[r for r in refs if r['split']==split];ids={r['image_id'] for r in rr};sets[split]=ids
        stats[split]=dict(images=len(ids),references=len(rr),expressions=sum(len(r['sentences']) for r in rr))
        assert tuple(stats[split].values())==want,(split,stats[split],want)
        rows[split]=[dict(sample_id=s['sent_id'],ref_id=r['ref_id'],ann_id=r['ann_id'],image_id=r['image_id'],
            image_index=image_index[r['image_id']],ann_index=ann_index[r['ann_id']],expression=s['raw']) for r in rr for s in r['sentences']]
        rows[split].sort(key=lambda x:x['sample_id'])
    assert len({r['ref_id'] for r in refs})==len(refs)
    assert len({r['sample_id'] for rs in rows.values() for r in rs})==sum(len(rs) for rs in rows.values())
    for a in sets:
        for b in sets:
            if a!=b:assert not sets[a]&sets[b]
    dump(ROOT/'data_status.json',dict(phase='downloading_images',total=len(image_list),statistics=stats))
    local=threading.local()
    def image_job(info):
        if not hasattr(local,'session'):local.session=requests.Session()
        name=f"COCO_train2014_{info['id']:012d}.jpg";path=ASSETS/'images'/name;path.parent.mkdir(exist_ok=True)
        if not path.exists():
            for attempt in range(5):
                try:
                    r=local.session.get('https://s3.amazonaws.com/images.cocodataset.org/train2014/'+name,timeout=(10,45));r.raise_for_status()
                    im=Image.open(io.BytesIO(r.content));im.load();assert im.size==(info['width'],info['height'])
                    tmp=path.with_suffix('.part');tmp.write_bytes(r.content);tmp.replace(path);break
                except Exception:
                    if attempt==4:raise
                    time.sleep(attempt+1)
        im=Image.open(path).convert('RGB');assert im.size==(info['width'],info['height'])
        g=geometry(*im.size);canvas=Image.new('RGB',(256,256),(124,116,104))
        canvas.paste(im.resize((g['new_width'],g['new_height']),Image.Resampling.BILINEAR),(g['left'],g['top']))
        return dict(id=info['id'],file_name=name,sha256=sha(path),**g),np.asarray(canvas,dtype=np.uint8)
    array=np.lib.format.open_memmap(ASSETS/'images256.npy',mode='w+',dtype='uint8',shape=(len(image_list),256,256,3));audits=[None]*len(image_list)
    with ThreadPoolExecutor(32) as pool:
        jobs={pool.submit(image_job,info):i for i,info in enumerate(image_list)}
        for count,future in enumerate(as_completed(jobs),1):
            i=jobs[future];audit,img=future.result();audits[i]=audit;array[i]=img
            if count%250==0:
                dump(ROOT/'data_status.json',dict(phase='downloading_images',completed=count,total=len(image_list),statistics=stats));print('IMAGES',count,len(image_list),flush=True)
    array.flush();del array
    # Exact native masks stay as lossless PNG; 256 labels are training-only.
    target=np.lib.format.open_memmap(ASSETS/'masks256.npy',mode='w+',dtype='uint8',shape=(len(ann_list),256,256));(ASSETS/'masks').mkdir(exist_ok=True)
    def mask_job(pair):
        i,ann=pair;info=images[ann['image_id']];m=native_mask(ann,info);assert m.any()
        path=ASSETS/'masks'/f"{ann['id']}.png";Image.fromarray(m*255).save(path)
        g=audits[image_index[ann['image_id']]];canvas=Image.new('L',(256,256),0)
        canvas.paste(Image.fromarray(m).resize((g['new_width'],g['new_height']),Image.Resampling.NEAREST),(g['left'],g['top']))
        return i,np.asarray(canvas),dict(id=ann['id'],image_id=ann['image_id'],sha256=sha(path),area=int(m.sum()))
    mask_audit=[None]*len(ann_list)
    with ThreadPoolExecutor(8) as pool:
        for i,m,audit in pool.map(mask_job,enumerate(ann_list)):target[i]=m;mask_audit[i]=audit
    target.flush();del target
    for split,r in rows.items():dump(ASSETS/f'{split}.json',r)
    dump(ASSETS/'images.json',audits);dump(ASSETS/'masks.json',mask_audit)
    dump(ROOT/'data_manifest.json',dict(passed=True,dataset='RefCOCO',splitBy='unc',statistics=stats,
        source=dict(repo=SOURCE,revision=REV,annotation_mirror=True,official_zip_byte_identity_unverified=True,
            official_definition='https://github.com/lichengunc/refer',images='https://s3.amazonaws.com/images.cocodataset.org/train2014/'),
        annotation_sha256={Path(f).name:sha(ASSETS/Path(f).name) for f in FILES},
        manifest_sha256={p.name:sha(p) for p in ASSETS.glob('*.json')},image_split_disjoint=True,
        image_sha256_in=str(ASSETS/'images.json'),mask_sha256_in=str(ASSETS/'masks.json'),native_masks='official COCO polygon/RLE rasterization via pycocotools',
        transforms='aspect-preserving bilinear resize; centered mean-color padding; nearest mask; padding loss excluded'))
    dump(ROOT/'data_status.json',dict(phase='complete',statistics=stats));print('DATA PREPARED',stats,flush=True)

if __name__=='__main__':main()
