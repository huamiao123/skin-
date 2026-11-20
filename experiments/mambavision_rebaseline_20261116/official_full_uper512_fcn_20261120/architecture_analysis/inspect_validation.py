import sys,json,csv,math,hashlib
from pathlib import Path
import numpy as np, torch, cv2
from torch.nn import functional as F
root=Path('/home/featurize/work/mambavision_seg_rebaseline_20261116/official_full_uper512_fcn_20261120'); dest=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
import official
torch.cuda.set_per_process_memory_fraction(.093,0)
torch.set_num_threads(2);cv2.setNumThreads(1)
def dice(pred,gt):
 n=pred.sum()+gt.sum()
 return float(2*np.count_nonzero(pred&gt)/n) if n else 1.
def summarize(rows):
 names=['main','aux','mean_logit_main_aux','flip_only','mean_logit_flip','center','rim_ideal','far_ideal','rimfn_ideal','farfn_ideal','rimfp_ideal','farfp_ideal']
 stats={k:float(np.mean([r[k] for r in rows])) for k in names}
 cnt={k:sum(r[k] for r in rows) for k in ('fn','fp','farfn','farfp','conf_farfn','aux_correct_fn','aux_correct_farfn','aux_added_fp','disagree_flip','disagree_aux')}
 cnt['pixels']=len(rows)*256**2
 for k in ('mean_logit_flip','mean_logit_main_aux'):
  diff=np.array([r[k]-r['main'] for r in rows])
  stats[k+'_mean_delta_pp']=round(100*float(diff.mean()),4)
  stats[k+'_winloss']=[int((diff>0).sum()),int((diff<0).sum())]
 return dict(n=len(rows),macro=stats,counts=cnt)
def go(ds):
 folder=root/ds
 fit=json.loads((folder/'fit_complete.json').read_text());checkpoint=folder/fit['best_checkpoint']
 assert official.sha(checkpoint)==fit['sha256']
 official.recipe.setup(42)
 net=official.Segmenter();net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);net.cuda().eval()
 data=official.dataset(ds,['val']);ids=data.ids['val']
 hist=json.loads((folder/'history.json').read_text());ref=hist[fit['best_epoch']-1]['validation_macro_dice']
 rows=[]; shapes={}
 with torch.inference_mode():
  for i in range(0,len(ids),4):
   indices=torch.arange(i,min(i+4,len(ids)),device='cuda')
   x,y=data.batch('val',indices)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    features=net.backbone(x); main0=net.probe.decode_head(features);aux0=net.probe.auxiliary_head(features)
    up=lambda z:F.interpolate(z,size=x.shape[-2:],mode='bilinear',align_corners=False)
    main=up(main0).float();aux=up(aux0).float();mir=net(x.flip([-1]))['main'].flip([-1]).float()
   if i==0:shapes=dict(features=[list(f.shape[1:]) for f in features],main=list(main0.shape[1:]),aux=list(aux0.shape[1:]))
   outputs=[o[:,0].cpu().numpy() for o in (main,aux,mir)]
   ys=y[:,0].cpu().numpy()
   for j,(m,a,z,g) in enumerate(zip(*outputs,ys)):
    assert np.isfinite(m).all() and np.isfinite(a).all() and np.isfinite(z).all()
    g=g>=.5;pred=m>=0;auxp=a>=0;flip=z>=0
    fn=g&~pred;fp=~g&pred
    din=cv2.distanceTransform(g.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    dout=cv2.distanceTransform((~g).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    far=np.where(g,din,dout)>8
    def ideal(where):return dice(np.where(where,g,pred),g)
    rows.append(dict(id=ids[i+j],area=int(g.sum()),main=dice(pred,g),aux=dice(auxp,g),mean_logit_main_aux=dice((m+a)>=0,g),
       flip_only=dice(flip,g),mean_logit_flip=dice((m+z)>=0,g),center=float(g.mean()),
       rim_ideal=ideal(~far),far_ideal=ideal(far),rimfn_ideal=ideal(g&~far),farfn_ideal=ideal(g&far),
       rimfp_ideal=ideal(~g&~far),farfp_ideal=ideal(~g&far),fn=int(fn.sum()),fp=int(fp.sum()),farfn=int((fn&far).sum()),
       farfp=int((fp&far).sum()),conf_farfn=int((fn&far&(m<math.log(.1/.9))).sum()),
       aux_correct_fn=int((fn&auxp).sum()),aux_correct_farfn=int((fn&far&auxp).sum()),aux_added_fp=int((~g&~pred&auxp).sum()),
       disagree_flip=int((flip!=pred).sum()),disagree_aux=int((auxp!=pred).sum())))
 assert len(rows)==len(ids)
 result=summarize(rows); result['shapes']=shapes;result['mean_reference_diff_pp']=(result['macro']['main']-ref)*100
 assert abs(result['mean_reference_diff_pp'])<.1
 result['protocol']='Fixed checkpoints, validation only; equal logits before threshold 0.5. Oracle diagnostic uses GT pixels, never for inference.'
 (dest/(ds+'.json')).write_text(json.dumps(result,indent=2))
 with (dest/(ds+'_cases.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('DONE',ds,result['n'],result['macro']['mean_logit_flip_mean_delta_pp'],result['macro']['mean_logit_main_aux_mean_delta_pp'],flush=True)
if __name__=='__main__':
 for ds in ['PH2','Waterloo','ISIC2018']:go(ds)

