"""Read-only reference data and denominator-aware, per-image error accounting."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
COLUMNS=('pixels','dino_wrong','dino_correct','corrected','introduced','candidate_wrong','complementary','complementary_repaired')
BINS=('[0,.1)','[.1,.4)','[.4,.6)','[.6,.9)','[.9,1]')
REGIONS=('boundary_2px','foreground_interior','background_exterior')
CORRECTNESS=('both_correct','dino_correct_vmamba_wrong','dino_wrong_vmamba_correct','both_wrong')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def context():
    ids=json.loads((PROJECT/'stage4_fusion/results/seed42/dinov3_swin/best_dice/ids.json').read_text())
    masks={p.stem.removesuffix('_segmentation').removesuffix('_mask'):p for p in (PROJECT/'data/isic2018/val/masks').iterdir() if p.is_file()}
    manifest=json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())
    for m in manifest['members']:
        if m['split']=='val':assert sha(masks[m['id']])==m['mask_sha256']
    gt=np.stack([np.asarray(Image.open(masks[s]).convert('L').resize((256,256),Image.Resampling.NEAREST))>=128 for s in ids])
    regions=np.empty_like(gt,dtype=np.uint8)
    for i,g in enumerate(gt):
        boundary=distance_transform_edt(~(g^binary_erosion(g)))<=2
        regions[i]=np.where(boundary,0,np.where(g,1,2))
    return ids,gt,regions


def references(criterion,ids):
    out={}
    for name in ('dinov3','vmamba','fusion'):
        p=PROJECT/f'stage4_fusion/results/seed42/{name}_swin'/criterion
        assert json.loads((p/'ids.json').read_text())==ids
        out[name]=np.load(p/'probabilities.npy',mmap_mode='r')
    out['mean']=(out['dinov3']+out['vmamba'])*.5
    return out


def record(counts):
    d={k:int(v) for k,v in zip(COLUMNS,counts)}
    d.update(net_errors_removed=d['corrected']-d['introduced'],
        repair_rate=d['corrected']/d['dino_wrong'] if d['dino_wrong'] else None,
        damage_rate=d['introduced']/d['dino_correct'] if d['dino_correct'] else None,
        candidate_error_rate=d['candidate_wrong']/d['pixels'] if d['pixels'] else None,
        complementary_repair_rate=d['complementary_repaired']/d['complementary'] if d['complementary'] else None)
    assert d['dino_wrong']-d['candidate_wrong']==d['net_errors_removed']
    return d


def pixel_diagnostics(ids,gt,regions,dino,vmamba,candidate):
    joint=np.zeros((5,3,4,8),dtype=np.int64);class_counts=np.zeros((2,8),dtype=np.int64);rows=[]
    for i,sid in enumerate(ids):
        p=dino[i];q=candidate[i];assert np.isfinite(q).all() and q.min()>=0 and q.max()<=1
        de=(p>=.5)!=gt[i];me=(vmamba[i]>=.5)!=gt[i];ce=(q>=.5)!=gt[i]
        bins=np.clip(np.searchsorted([0.,.1,.4,.6,.9,1.],p,side='right')-1,0,4)
        category=de.astype(np.int64)*2+me.astype(np.int64)
        indices=bins*12+regions[i]*4+category
        comp=de&~me
        masks=(np.ones_like(de),de,~de,de&~ce,~de&ce,ce,comp,comp&~ce)
        per=np.array([int(m.sum()) for m in masks])
        for col,m in enumerate(masks):
            joint[:,:,:,col]+=np.bincount(indices[m],minlength=60).reshape(5,3,4)
            class_counts[0,col]+=(m&~gt[i]).sum();class_counts[1,col]+=(m&gt[i]).sum()
        rows.append(dict(id=sid,**record(per)))
    total=joint.sum((0,1,2))
    assert total[0]==len(ids)*256*256
    assert np.array_equal(class_counts.sum(0),total)
    report=dict(all=record(total),probability={name:record(joint[j].sum((0,1))) for j,name in enumerate(BINS)},
        regions={name:record(joint[:,j].sum((0,1))) for j,name in enumerate(REGIONS)},
        correctness={name:record(joint[:,:,j].sum((0,1))) for j,name in enumerate(CORRECTNESS)},
        gt_class={name:record(class_counts[j]) for j,name in enumerate(('background','foreground'))},
        joint_counts=joint.tolist(),joint_axes=[list(BINS),list(REGIONS),list(CORRECTNESS),list(COLUMNS)])
    return report,rows
