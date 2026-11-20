from pathlib import Path
from PIL import Image
import numpy as np, hashlib,json,csv,collections,itertools,zipfile,io
b=Path('/home/featurize/work/mambavision_seg_rebaseline_20261116');run=Path('/home/featurize/work/mambavision_seg_rebaseline_20261116/official_full_uper512_fcn_20261120');ext=b/'uper64_feature_audit_20261119/transfer_20261119/external_test_20261119'
manifest={(r['dataset'],r['id']):r for r in csv.DictReader((ext/'manifest.csv').open())}
def digest(x): return hashlib.sha256(x).hexdigest()
def paired(file):
 with file.open() as f:return {r['image_id']:float(r['foreground_dice']) for r in csv.DictReader(f)}
out={}
for ds in ('PH2','Waterloo','ISIC2018'):
 d=run/ds;fd=b/'four_dataset_matched_20261119'/ds
 rows=list(csv.DictReader((fd/'split.csv').open()));cache=Path('/tmp/mv_four_matched_20261119')/ds
 x=np.load(cache/'images.npy',mmap_mode='r');y=np.load(cache/'masks.npy',mmap_mode='r')
 assert x.shape==(len(rows),256,256,3) and y.shape==(len(rows),256,256)
 assert len({r['image_id'] for r in rows})==len(rows) and {int(r['index']) for r in rows}==set(range(len(rows)))
 counts=collections.Counter(r['split'] for r in rows);h=collections.defaultdict(set);g=collections.defaultdict(set)
 source_match=0;empty=0;allfg=0
 z=zipfile.ZipFile(ext/'data/waterloo.zip') if ds=='Waterloo' else None
 for r in rows:
  i=int(r['index']); key=r['image_id'];s=r['split'];image=x[i];mask=y[i]
  h[s].add(digest(image)); 
  if r['known_group']:g[s].add(r['known_group'])
  assert set(np.unique(mask)).issubset({0,255});empty+=int(not mask.any());allfg+=int(mask.min()==255)
  if ds in ('PH2','Waterloo'):
   ref=manifest[ds,key]
   if z:
    assert ref['image_path'].lower().endswith('_orig.jpg') and ref['mask_path'].lower().endswith('_contour.png')
    ib=z.read(ref['image_path']);mb=z.read(ref['mask_path'])
   else:
    assert '_Dermoscopic_Image/' in ref['image_path'] and '_lesion/' in ref['mask_path']
    ib=Path(ref['image_path']).read_bytes();mb=Path(ref['mask_path']).read_bytes()
   assert digest(ib)==ref['image_sha256'] and digest(mb)==ref['mask_sha256']
   with Image.open(io.BytesIO(ib)) as im: assert np.array_equal(image,np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)))
   with Image.open(io.BytesIO(mb)) as gt: assert np.array_equal(mask>=128,np.asarray(gt.convert('L').resize((256,256),Image.Resampling.NEAREST))>=128)
   source_match+=1
 if z:z.close()
 splits=('train','val','test')
 intersections={'/'.join([i,j]):len(h[i]&h[j]) for i,j in itertools.combinations(splits,2)}
 groups={'/'.join([i,j]):len(g[i]&g[j]) for i,j in itertools.combinations(splits,2)}
 results=paired(d/'final_test_per_image.csv')
 test=[r for r in rows if r['split']=='test'];assert list(results)==[r['image_id'] for r in test]
 v=np.array(list(results.values()))
 rng=np.random.default_rng(12042)
 names=sorted({r['known_group'] or r['image_id'] for r in test});bucket={k:[] for k in names}
 for r in test:bucket[r['known_group'] or r['image_id']].append(results[r['image_id']])
 sizes=np.array([len(bucket[n]) for n in names]);sums=np.array([sum(bucket[n]) for n in names])
 sample=rng.integers(0,len(names),size=(5000,len(names)));ci=np.quantile(sums[sample].sum(1)/sizes[sample].sum(1),[.025,.975])
 tr=[int(r['index']) for r in rows if r['split']=='train'];shape=sum((y[i]>=128).astype(np.int32) for i in tr)>=len(tr)*.5
 template=[2*np.count_nonzero(shape&(y[int(r['index'])]>=128))/(shape.sum()+np.count_nonzero(y[int(r['index'])])) for r in test]
 hist=json.loads((d/'history.json').read_text());fit=json.loads((d/'fit_complete.json').read_text());assert max(hist,key=lambda z:z['validation_macro_dice'])['epoch']==fit['best_epoch']
 out[ds]=dict(counts=counts,exact_cross=intersections,known_groups={k:len(g[k]) for k in splits},group_intersection=groups,raw_matched=source_match,empty=empty,all_foreground=allfg,mean=float(v.mean()),median=float(np.median(v)),worst=float(v.min()),below_80=int((v<.8).sum()),at_least_90=int((v>=.9).sum()),bootstrap=[float(k) for k in ci],mean_shape=float(np.mean(template)),best_epoch=fit['best_epoch'])
 print(ds,json.dumps(out[ds]),flush=True)
(run/'audit_coverage/coverage.json').write_text(json.dumps(out,indent=2))

