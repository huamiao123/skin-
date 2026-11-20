from pathlib import Path
import json,csv,hashlib,io,zipfile,requests,collections,numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
b=Path('/home/featurize/work/mambavision_seg_rebaseline_20261116'); out=Path('/home/featurize/work/mambavision_seg_rebaseline_20261116/official_full_uper512_fcn_20261120/audit_coverage'); audit=Path('/home/waas/mambavision_stage22/local_audit_records/test_assets')
meta=json.loads((audit/'acquisition.json').read_text()); source=Path('/tmp/isic_gt_audit_20261120.zip')
if not source.exists():
 r=requests.get(meta['mask_url'],timeout=(12,45));r.raise_for_status()
 assert hashlib.sha256(r.content).hexdigest()==meta['mask_sha256'];source.write_bytes(r.content)
with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==meta['mask_sha256']
cache=np.load('/tmp/mv_four_matched_20261119/ISIC2018/masks.npy',mmap_mode='r')
rows=list(csv.DictReader((b/'four_dataset_matched_20261119/ISIC2018/split.csv').open()))
raw=b/'diagnostics/stageC_20261117/data_provenance'
arc=[zipfile.ZipFile(path) for path in (raw/'ISIC2018_Task1_Training_GroundTruth.zip',raw/'ISIC2018_Task1_Validation_GroundTruth.zip',source)]
options={}
for z in arc:
 for k in z.namelist():
  if k.endswith('_segmentation.png') and not k.startswith('__MACOSX'):options[Path(k).name]=(z,k)
def compare(row):
 q=row['image_id']; z,k=options[q+'_segmentation.png']
 with Image.open(io.BytesIO(z.read(k))) as im:
  im=im.convert('L'); old=np.asarray(im.resize((256,256),Image.Resampling.NEAREST))>=128
  full=float(np.asarray(im).mean()/255) if q=='ISIC_0023056' else None
 ours=cache[int(row['index'])]>=128
 return (row['split'],int(np.count_nonzero(old!=ours)),full)
scores={};note=None
with ThreadPoolExecutor(3) as pool:
 for split,delta,full in pool.map(compare,rows):
  d=scores.setdefault(split,dict(n=0,exact=0,total_different_pixels=0,max_pixels=0))
  d['n']+=1;d['exact']+=int(delta==0);d['total_different_pixels']+=delta;d['max_pixels']=max(d['max_pixels'],delta)
  if full is not None:note=full
outdata=dict(reference_sha256=meta['mask_sha256'],groups=scores,ISIC_0023056_original_area=note)
(out/'official_mask_audit.json').write_text(json.dumps(outdata,indent=2));print(outdata)

