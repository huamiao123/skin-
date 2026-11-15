from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent.parent
r=P/'stage48_cfam_refinement'
history={(x['split'],x['id']):x for x in json.loads((P/'stage1_visual/data_manifest.json').read_text())['members']}
ids=['511','536','53','673','126','136'];rows=[]
fig,axes=plt.subplots(len(ids),3,figsize=(9,16))
for sid,axs in zip(ids,axes):
 p=P/'data/isic2018/val/masks'/f'{sid}.png';im=Image.open(p);arr=np.asarray(im);a=np.asarray(im.convert('L'))
 vals,cnts=np.unique(a,return_counts=True);rank=np.argsort(cnts)[-10:][::-1]
 row=dict(id=sid,mode=im.mode,size=im.size,raw_shape=arr.shape,min=int(a.min()),max=int(a.max()),
 top_values=[(int(vals[i]),int(cnts[i])) for i in rank],
 greater0=int((a>0).sum()),greater127=int((a>=127.5).sum()),is255=int((a==255).sum()),
 legacy_hash_matches=hashlib.sha256(p.read_bytes()).hexdigest()==history['val',sid]['mask_sha256'])
 rows.append(row)
 axs[0].imshow(Image.open(P/'data/isic2018/val/images'/f'{sid}.png'));axs[0].set_title(sid+' image')
 axs[1].imshow(a,cmap='gray',vmin=0,vmax=255);axs[1].set_title('stored L mask')
 axs[2].imshow(a>=127.5,cmap='gray',vmin=0,vmax=1);axs[2].set_title('>=127.5')
 for ax in axs:ax.axis('off')
fig.tight_layout();fig.savefig(r/'diagnosis/mask_audit.png',dpi=110);fig.savefig(r/'diagnosis/mask_audit_preview.jpg',dpi=65,pil_kwargs={'quality':65})
allstats={}
for split in ['train','val']:
 hist=np.zeros(256,dtype=np.int64);nonbinary=0;bad_hash=[];modes={};severe_threshold=[]
 for p in (P/'data/isic2018'/split/'masks').glob('*.png'):
  im=Image.open(p);a=np.asarray(im.convert('L'));h=np.bincount(a.ravel(),minlength=256);hist+=h;modes[im.mode]=modes.get(im.mode,0)+1
  nonbinary+=int(np.any((a!=0)&(a!=255)))
  if hashlib.sha256(p.read_bytes()).hexdigest()!=history[split,p.stem]['mask_sha256']:bad_hash.append(p.stem)
  if (a>0).sum()>2*max((a>=127.5).sum(),1):severe_threshold.append(p.stem)
 allstats[split]=dict(histogram=hist.tolist(),nonbinary_files=nonbinary,hash_mismatches=bad_hash,modes=modes,positive_area_more_than_twice_threshold_area_ids=severe_threshold)
out=dict(examples=rows,all_legacy_masks=allstats,scope='Stored legacy train/val only; no test labels read; no masks or thresholds changed.')
(r/'diagnosis/MASK_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(examples=rows,summary={k:{z:v for z,v in a.items() if z!='histogram'} for k,a in allstats.items()}),indent=2))
