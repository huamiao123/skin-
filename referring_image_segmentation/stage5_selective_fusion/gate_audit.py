"""Fixed-checkpoint gate interventions; descriptive, never a new model selector."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from models import ROOT,EncoderPair,ResidualProbe,tensor_hash
from stage3_confirmation.data import ImageData
from stage3_confirmation.engine import encode,evaluate
from stage1_visual.analyze import metrics
from evaluation import summarize_rows
from diagnostics import dump


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    base=ROOT/('smoke' if args.smoke else 'results')/'seed42'
    out=ROOT/('smoke_analysis' if args.smoke else 'analysis')/'gate_audit';out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    encoder=EncoderPair().cuda();hashes={n:tensor_hash(getattr(encoder,n)) for n in ('dinov3','vmamba')}
    data=ImageData(42)
    assert data.audit==json.loads((ROOT.parent/'stage1_visual/data_manifest.json').read_text())['members']
    features,targets=encode(encoder,data,'val',64)
    gt=targets[:,0].cpu().numpy()>=.5
    old=np.load(ROOT.parent/'stage4_fusion/results/seed42/dinov3_swin/best_dice/probabilities.npy',mmap_mode='r')
    dino_error=(old>=.5)!=gt
    checkpoint=torch.load(base/'dynamic_gate/best_dice.pth',map_location='cpu')
    probe=ResidualProbe(42,True).cuda().eval();probe.load_state_dict(checkpoint['probe'])
    summaries={};per_gate=[];values=[[] for _ in range(4)]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(targets),64):
            logits,gates=probe([f[start:start+64] for f in features],return_gates=True)
            arrays=[g[:,0].cpu().numpy() for g in gates]
            for level,a in enumerate(arrays):values[level].append(a.reshape(-1))
            for j in range(len(arrays[0])):
                per_gate.append(dict(id=data.ids['val'][start+j],scales=[dict(mean=float(a[j].mean()),spatial_std=float(a[j].std()),
                    min=float(a[j].min()),max=float(a[j].max())) for a in arrays]))
    distributions=[]
    for level,parts in enumerate(values):
        a=np.concatenate(parts)
        distributions.append(dict(side=(64,32,16,8)[level],mean=float(a.mean()),std=float(a.std()),
            percentiles=dict(zip(['p0','p5','p25','p50','p75','p95','p100'],np.percentile(a,[0,5,25,50,75,95,100]).tolist()))))
    for mode in ('learned','zero','spatial_mean','shift'):
        probe.gate_mode=mode
        region,probabilities,_=evaluate(probe,features,targets,export=True)
        if mode=='learned':assert region==checkpoint['metrics']
        pred=probabilities>=.5
        def sample(i):return dict(id=data.ids['val'][i],**metrics(pred[i],gt[i]))
        with ThreadPoolExecutor(8) as pool:rows=list(pool.map(sample,range(len(targets))))
        full=summarize_rows(rows);assert full['counts']==region['counts']
        error=pred!=gt;corrected=int((dino_error&~error).sum());introduced=int((~dino_error&error).sum())
        full['repair_vs_archived_dino']=dict(corrected=corrected,introduced=introduced,net_errors_removed=corrected-introduced,
            repair_rate=corrected/int(dino_error.sum()),damage_rate=introduced/int((~dino_error).sum()))
        summaries[mode]=full;dump(out/f'{mode}_per_image.json',rows)
        np.save(out/f'{mode}_probabilities.npy',probabilities)
        print('GATE_AUDIT',mode,full['macro']['dice'],flush=True)
    assert all(tensor_hash(getattr(encoder,n))==h for n,h in hashes.items())
    dump(out/'per_image_gate_statistics.json',per_gate)
    dump(out/'summary.json',dict(passed=True,smoke_only=args.smoke,selected_epoch=checkpoint['epoch'],gate_distributions=distributions,
        interventions=summaries,note='Same trained dynamic checkpoint; no refit/reselection. Zero gate is the co-adapted DINO path, not the archived DINO model. Gate magnitudes are not probabilities of correctness.'))
    lines=['# 固定权重门控诊断','', '使用dynamic主checkpoint，不重训、不重新选模。零门控使用当前共同训练后的DINO路径，不能当作原始DINO baseline。','',
        '| Gate | Mean Dice | Pooled Dice | Boundary F1 | HD95 px | 净纠错像素 |','|---|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        lines.append(f"| {name} | {s['macro']['dice']:.6f} | {s['pooled']['dice']:.6f} | {s['macro']['boundary_f1']:.6f} | {s['macro']['hd95_pixels']:.3f} | {s['repair_vs_archived_dino']['net_errors_removed']} |")
    lines+=['','全部统一指标及每尺度gate分布见summary.json；逐图门控均值/空间标准差见per_image_gate_statistics.json。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print('GATE AUDIT PASSED',flush=True)


if __name__=='__main__':main()
