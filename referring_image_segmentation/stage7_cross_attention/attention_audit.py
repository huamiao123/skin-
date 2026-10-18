"""Fixed-checkpoint retrieval interventions, without retraining or reselection."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from models import ROOT,PROJECT,EncoderPair,InteractionProbe,tensor_hash
from stage3_confirmation.data import ImageData
from stage3_confirmation.engine import encode,evaluate
from stage1_visual.analyze import metrics
from evaluation import summarize_rows
from stage5_selective_fusion.diagnostics import dump,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    base=ROOT/('smoke' if args.smoke else 'results')/'seed42'
    out=ROOT/('smoke_analysis' if args.smoke else 'analysis')/'attention_audit';out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    encoder=EncoderPair().cuda();hashes={n:tensor_hash(getattr(encoder,n)) for n in ('dinov3','vmamba')}
    data=ImageData(42);assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    features,targets=encode(encoder,data,'val',64);gt=targets[:,0].cpu().numpy()>=.5
    ref=PROJECT/'stage4_fusion/results/seed42/dinov3_swin/best_dice'
    assert json.loads((ref/'ids.json').read_text())==data.ids['val']
    dino_error=(np.load(ref/'probabilities.npy',mmap_mode='r')>=.5)!=gt
    path=base/'window_cross_attention/best_dice.pth';checkpoint=torch.load(path,map_location='cpu')
    probe=InteractionProbe(42,'window_cross_attention').cuda().eval();probe.load_state_dict(checkpoint['probe'])
    original=tensor_hash(probe);summaries={}
    for mode in ('learned','uniform','same_position'):
        for layer in probe.interaction:layer.mode=mode
        region,probabilities,_=evaluate(probe,features,targets,export=True)
        if mode=='learned':
            assert region==checkpoint['metrics']
            assert np.array_equal(probabilities,np.load(base/'window_cross_attention/best_dice/probabilities.npy',mmap_mode='r'))
        pred=probabilities>=.5
        def sample(i):return dict(id=data.ids['val'][i],**metrics(pred[i],gt[i]))
        with ThreadPoolExecutor(8) as pool:rows=list(pool.map(sample,range(len(targets))))
        full=summarize_rows(rows);assert full['counts']==region['counts']
        error=pred!=gt;corrected=int((dino_error&~error).sum());introduced=int((~dino_error&error).sum())
        full['repair_vs_archived_dino']=dict(corrected=corrected,introduced=introduced,net_errors_removed=corrected-introduced,
            repair_rate=corrected/int(dino_error.sum()),damage_rate=introduced/int((~dino_error).sum()))
        summaries[mode]=full;dump(out/f'{mode}_per_image.json',rows);np.save(out/f'{mode}_probabilities.npy',probabilities)
        print('ATTENTION_AUDIT',mode,full['macro']['dice'],flush=True)
    assert tensor_hash(probe)==original
    assert all(tensor_hash(getattr(encoder,n))==h for n,h in hashes.items())
    dump(out/'summary.json',dict(passed=True,smoke_only=args.smoke,selected_epoch=checkpoint['epoch'],checkpoint_sha256=sha(path),
        interventions=summaries,note='Same trained Cross checkpoint; no refit or model selection. Uniform replaces content/position attention by mean V; same_position retrieves only colocated V. Distribution shifts mean this does not replace separately trained controls.'))
    lines=['# 固定权重空间检索诊断','',
        '使用Cross主checkpoint，不重训、不重新选模。uniform为窗口内平均V，same_position为同位置V；保留原输出投影和残差。干预改变输入/计算分布，不能单凭下降认定注意力优于独立训练的卷积。','',
        '| Mode | Mean Dice | Pooled Dice | mIoU | oIoU | Boundary F1 | HD95 px | 净纠错像素 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,s in summaries.items():
        vals=[s['macro']['dice'],s['pooled']['dice'],s['macro']['iou'],s['pooled']['iou'],s['macro']['boundary_f1'],s['macro']['hd95_pixels']]
        lines.append('| '+n+' | '+' | '.join(f'{v:.6f}' for v in vals)+f" | {s['repair_vs_archived_dino']['net_errors_removed']} |")
    lines+=['','全部Precision/Recall、Prec@X、HD95归一化、Threshold Jaccard见summary.json；逐图记录与概率已保存。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print('ATTENTION AUDIT PASSED',flush=True)


if __name__=='__main__':main()
