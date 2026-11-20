from pathlib import Path
import json,csv,hashlib,gc,sys
import numpy as np,torch
root=Path('/home/featurize/work/mambavision_seg_rebaseline_20261116/official_full_uper512_fcn_20261120');sys.path.insert(0,str(root))
import official
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
torch.cuda.set_per_process_memory_fraction(.095,0)
torch.set_num_threads(2)
out={}
for ds in ('PH2','Waterloo','ISIC2018'):
 folder=root/ds;fit=json.loads((folder/'fit_complete.json').read_text());ck=folder/fit['best_checkpoint']
 assert sha(ck)==fit['sha256']
 official.recipe.setup(42)
 net=official.Segmenter()
 net.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu'),strict=True)
 net.cuda().eval()
 cache=Path('/tmp/mv_four_matched_20261119')/ds
 raw=np.load(cache/'images.npy',mmap_mode='r');mask=np.load(cache/'masks.npy',mmap_mode='r')
 ps=Path('/home/featurize/work/mambavision_seg_rebaseline_20261116')/'four_dataset_matched_20261119'/ds/'split.csv'
 rows=[v for v in csv.DictReader(ps.open()) if v['split']=='test']
 stored={r['image_id']:float(r['foreground_dice']) for r in csv.DictReader((folder/'final_test_per_image.csv').open())}
 assert len(rows)==len(stored) and set(stored)=={r['image_id'] for r in rows}
 calc=[];errors=[];both_empty=0
 with torch.inference_mode():
  for i in range(0,len(rows),4):
   chunk=rows[i:i+4];indices=[int(v['index']) for v in chunk]
   x=torch.from_numpy(np.array(raw[indices])).permute(0,3,1,2).contiguous().cuda().float()/255.
   x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
   with torch.autocast('cuda',dtype=torch.bfloat16):o=net(x)
   assert set(o)=={'main'} and torch.isfinite(o['main']).all()
   pred=o['main'][:,0].float().cpu().numpy()>=0
   gt=mask[indices]>=128
   inter=(pred&gt).sum((1,2));den=pred.sum((1,2))+gt.sum((1,2))
   both_empty+=int((den==0).sum()); vals=np.divide(2*inter,den,out=np.ones(len(den)),where=den!=0)
   calc.extend(vals);errors.extend(v-stored[r['image_id']] for v,r in zip(vals,chunk))
 out[ds]=dict(n=len(rows),epoch=fit['best_epoch'],independent_mean=float(np.mean(calc)),saved_mean=float(np.mean(list(stored.values()))),max_abs_case_diff=float(np.max(np.abs(errors))),mean_abs_case_diff=float(np.mean(np.abs(errors))),both_empty=both_empty)
 print('REPLAY',ds,out[ds],flush=True)
 dest=root/'audit_coverage/replay.json';temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(out,indent=2));temp.replace(dest)
 del net;gc.collect();torch.cuda.empty_cache()

