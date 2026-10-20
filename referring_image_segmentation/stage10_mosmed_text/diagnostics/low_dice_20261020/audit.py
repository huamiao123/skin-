"""Read-only seed42 train/validation audit. No test inference and no training."""
from pathlib import Path
import sys
STAGE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(STAGE))
from common import *
import collections, numpy as np, torch, cv2
from PIL import Image,ImageDraw
from models import build_probes,FrozenEncoder
from data import MedicalData,FeatureCache
from engine import predict,score,restore,native_gt

OUT=Path(__file__).resolve().parent

def dice(a,b):
    denominator=int(a.sum())+int(b.sum())
    return 2*int(np.logical_and(a,b).sum())/denominator if denominator else 1.

def sample_records(low,metadata,records):
    independent=[];areas=[];empty=0;bygroup=collections.defaultdict(list)
    for z,r,old in zip(low,metadata,records):
        p=restore(z,r)>=0;g=native_gt(r['mask_path'])
        value=dice(p,g);independent.append(value);areas.append(float(g.mean()));empty+=int(not p.any())
        bygroup[r['group_id']].append(value)
        assert abs(value-old['dice'])<1e-12
    independent=np.array(independent);areas=np.array(areas)
    bins={}
    for name,lo,hi in [('below_0.1pct',0,.001),('0.1_to_1pct',.001,.01),('at_least_1pct',.01,1.01)]:
        keep=(areas>=lo)&(areas<hi)
        bins[name]=dict(n=int(keep.sum()),macro_dice=float(independent[keep].mean()) if keep.any() else None,
                       zero_dice=int((independent[keep]==0).sum()))
    return dict(independent_macro_dice=float(independent.mean()),zero_dice=int((independent==0).sum()),empty_predictions=empty,
        area_bins=bins,per_group={g:dict(n=len(v),macro_dice=float(np.mean(v))) for g,v in bygroup.items()})

def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data=MedicalData();enc=FrozenEncoder('dinov3').cuda();cache=FeatureCache(enc,data)
    for split in ('train','val'):cache.encode(split)
    done=json.loads((ROOT/'results/seed42/DONE.json').read_text());assert done['passed']
    models={};predictions={};all_records={};hashes={}
    for n,p in build_probes(42).items():
        path=ROOT/'results/seed42'/n/'best_dice.pth';hashes[str(path)]=sha(path)
        state=torch.load(path,map_location='cpu');p=p.cuda();p.load_state_dict(state['probe']);p.eval()
        report=dict(selected_epoch=state['epoch'],splits={});all_records[n]={}
        for split in ('train','val'):
            low,loss=predict(p,cache,split);metrics,records=score(low,data,split)
            # Independent Dice implementation, separate from the project's count aggregation.
            direct=[]
            for z,r in zip(low,data.rows[split]):direct.append(dice(restore(z,r)>=0,native_gt(r['mask_path'])))
            assert abs(float(np.mean(direct))-metrics['macro']['dice'])<1e-12
            for r,d in zip(records,direct):r['dice']=d
            report['splits'][split]=dict(metrics=metrics,loss=loss)
            if split=='val':
                report['splits'][split]['diagnostics']=sample_records(low,data.rows[split],records)
                expected=state['metrics']['macro']['dice'];assert abs(metrics['macro']['dice']-expected)<1e-10
                report['saved_checkpoint_validation_match']=True;predictions[n]=low
            dump(OUT/f'{n}_{split}_per_image.json',records);all_records[n][split]=records
            print('AUDIT',n,split,'macro',metrics['macro']['dice'],'pooled',metrics['pooled']['dice'],flush=True)
        models[n]=report;del p
    geometry=[];mask_exact=True
    cached=np.load(ASSETS/'masks256.npy',mmap_mode='r')
    for r in data.rows['val']:
        g=native_gt(r['mask_path']);mask=Image.fromarray(g.astype(np.uint8))
        roundtrips={}
        for side in (256,64,16):
            coarse=mask.resize((side,side),Image.Resampling.NEAREST)
            back=np.asarray(coarse.resize((r['width'],r['height']),Image.Resampling.NEAREST))>0
            roundtrips[str(side)]=dice(back,g)
        expected=np.asarray(mask.resize((r['new_width'],r['new_height']),Image.Resampling.NEAREST))
        t,l,h,w=[r[k] for k in ('top','left','new_height','new_width')]
        mask_exact=mask_exact and np.array_equal(expected,cached[r['index'],t:t+h,l:l+w])
        geometry.append(dict(id=r['id'],roundtrip_dice=roundtrips))
    assert mask_exact
    geometry_summary={side:dict(macro_dice=float(np.mean([r['roundtrip_dice'][side] for r in geometry])),
                               zero_dice=sum(r['roundtrip_dice'][side]==0 for r in geometry)) for side in ('256','64','16')}
    dump(OUT/'mask_resolution_roundtrip.json',geometry)
    # Show a fixed representative from every validation group, plus small/large lesions.
    rows=data.rows['val'];chosen=[]
    for g in sorted({r['group_id'] for r in rows}):
        members=[i for i,r in enumerate(rows) if r['group_id']==g]
        chosen.append(members[len(members)//2])
    areas=np.array([r['gt_pixels']/(r['width']*r['height']) for r in rows])
    chosen=list(dict.fromkeys(chosen+np.argsort(areas)[:2].tolist()+np.argsort(-areas)[:2].tolist()))
    panel=Image.new('RGB',(5*256,len(chosen)*282),(15,15,15));draw=ImageDraw.Draw(panel)
    for k,i in enumerate(chosen):
        r=rows[i];base=Image.open(r['image_path']).convert('RGB').resize((256,256));g=native_gt(r['mask_path'])
        panel.paste(base,(0,k*282+26));draw.text((3,k*282+2),r['id'],fill='white')
        panel.paste(Image.fromarray((g*255).astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST).convert('RGB'),(256,k*282+26))
        draw.text((260,k*282+2),f"GT area {areas[i]*100:.3f}%",fill='white')
        for c,n in enumerate(NAMES,2):
            p=restore(predictions[n][i],r)>=0;image=Image.fromarray((p*255).astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST)
            panel.paste(image.convert('RGB'),(c*256,k*282+26));draw.text((c*256+4,k*282+2),f"{n} Dice {dice(p,g):.3f}",fill='white')
    panel.save(OUT/'validation_prediction_panel.png')
    report=dict(passed=True,seed=42,read_only=True,test_predictions_not_computed=True,
        checkpoint_sha256=hashes,models=models,independent_dice_matches_common_metrics=True,
        saved_best_validation_scores_reproduced=True,original_to_cached_mask_matches=True,
        mask_resolution_roundtrip=geometry_summary,roundtrip_note='Nearest sampled GT then upsampled GT diagnostic, not a formal architectural upper bound.',
        validation_image_count=len(rows),validation_source_group_count=len({r['group_id'] for r in rows}),
        validation_below_1pct=int((areas<.01).sum()),prediction_panel='validation_prediction_panel.png')
    assert all(sha(p)==h for p,h in hashes.items());dump(OUT/'summary.json',report)
    print('AUDIT_PASSED','geometry',json.dumps(geometry_summary),flush=True)

if __name__=='__main__':main()
