"""Freeze one diagnosis-stratified, lesion-grouped HAM10000 split. Build a disposable uint8 input cache."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import collections
import csv
import hashlib
import json
import os
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = Path('/home/featurize/work/datasets/HAM10000')
CACHE = Path('/tmp/ham10000_window16_20261119')
SIZE = 256

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write_json(path, value):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def prepare():
    CACHE.mkdir(exist_ok=True)
    with (DATA/'HAM10000_metadata.tab').open() as f:
        rows = sorted(csv.DictReader(f, delimiter='\t'), key=lambda r:r['image_id'])
    assert len(rows) == len({r['image_id'] for r in rows}) == 10015
    verified = json.loads((DATA/'verification.json').read_text())
    assert verified['verified_pairs'] == len(rows)
    stamp = CACHE/'ready.json'
    if stamp.exists() and (ROOT/'split.csv').exists():
        old = json.loads(stamp.read_text())
        if old['split_sha256'] == digest(ROOT/'split.csv') and all((CACHE/f'{key}.npy').exists() for key in ('images','masks')):
            return
    inputs = np.lib.format.open_memmap(CACHE/'images.npy', mode='w+',dtype=np.uint8,shape=(len(rows),SIZE,SIZE,3))
    masks = np.lib.format.open_memmap(CACHE/'masks.npy', mode='w+',dtype=np.uint8,shape=(len(rows),SIZE,SIZE))
    def convert(ix):
        name=rows[ix]['image_id']
        with Image.open(DATA/'images'/(name+'.jpg')) as im, Image.open(DATA/'masks'/(name+'_segmentation.png')) as gt:
            x=np.asarray(im.convert('RGB').resize((SIZE,SIZE),Image.Resampling.BILINEAR))
            y=np.asarray(gt.convert('L').resize((SIZE,SIZE),Image.Resampling.NEAREST))
        inputs[ix]=x
        masks[ix]=y
        return hashlib.sha256(x.tobytes()).hexdigest()
    with ThreadPoolExecutor(10) as pool:
        pixel_hashes=list(pool.map(convert,range(len(rows))))
    inputs.flush()
    masks.flush()
    parents = {r['lesion_id']:r['lesion_id'] for r in rows}
    def find(k):
        while parents[k]!=k:
            parents[k]=parents[parents[k]]
            k=parents[k]
        return k
    seen={}
    for row,h in zip(rows,pixel_hashes):
        lid=row['lesion_id']
        parents[find(lid)]=find(seen.setdefault(h,lid))
    groups=[find(r['lesion_id']) for r in rows]
    labels=[r['dx'] for r in rows]
    rng=np.random.default_rng(42)
    members=collections.defaultdict(list)
    for ix,group in enumerate(groups):members[group].append(ix)
    codes={label:k for k,label in enumerate(sorted(set(labels)))}
    totals=np.zeros((10,len(codes)),dtype=int)
    split=np.full(len(rows),'train',dtype='U5')
    order=list(members)
    rng.shuffle(order)
    order.sort(key=lambda g:-len(members[g]))
    for group in order:
        ix=members[group]
        counts=np.bincount([codes[labels[i]] for i in ix],minlength=len(codes))
        # Place intact groups where their diagnoses currently have the fewest examples.
        fold=min(rng.permutation(10),key=lambda j:(int(totals[j] @ counts),int(totals[j].sum())))
        totals[fold]+=counts
        if fold<2:split[ix]=('test','val')[fold]
    report={}
    for s in ('train','val','test'):
        ix=np.flatnonzero(split==s)
        report[s]=dict(images=len(ix),lesions=len({rows[i]['lesion_id'] for i in ix}),diagnoses=dict(collections.Counter(labels[i] for i in ix)))
        assert len(report[s]['diagnoses'])==7
        for t in ('train','val','test'):
            if t==s:continue
            jx=np.flatnonzero(split==t)
            assert not ({groups[i] for i in ix}&{groups[i] for i in jx})
            assert not ({pixel_hashes[i] for i in ix}&{pixel_hashes[i] for i in jx})
    final=ROOT/'split.csv'
    tmp=ROOT/'split.csv.new'
    with tmp.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['index','image_id','lesion_id','diagnosis','split'])
        for i,r in enumerate(rows):w.writerow([i,r['image_id'],r['lesion_id'],r['dx'],split[i]])
    if final.exists() and digest(final)!=digest(tmp):raise RuntimeError('existing split changed')
    os.replace(tmp,final)
    result=dict(split_sha256=digest(final),counts=report,seed=42,equal_resized_image_groups=len(set(pixel_hashes)),lesions=len(parents),cross_split_lesions=0,cross_split_exact_inputs=0)
    write_json(ROOT/'split_summary.json',result)
    write_json(stamp,result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':prepare()
