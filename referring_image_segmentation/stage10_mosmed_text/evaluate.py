"""Evaluate the locked nine checkpoints; paired seed and source-group comparisons."""
from common import *
import argparse, csv, time
import numpy as np, torch
from torch.nn import functional as F
from models import build_probes, FrozenEncoder
from data import MedicalData, FeatureCache
from engine import predict, score
from evaluation import summarize_rows

@torch.no_grad()
def latency(probe,encoder,data,te=None,tokenizer=None):
    r=data.rows['val'][0];idx=torch.tensor([r['index']],device='cuda');x=data.inputs(idx)
    tokens=None
    if probe.name!=NAMES[0]:
        text=FIXED_TEXT if probe.name==NAMES[1] else r['text']
        tokens={k:v.cuda() for k,v in tokenizer([text],padding='max_length',max_length=77,return_tensors='pt').items()}
    probe.eval()
    def forward():
        with torch.autocast('cuda',dtype=torch.bfloat16):
            t=None if tokens is None else te(**tokens).pooler_output.float()
            return F.interpolate(probe(encoder(x),t),(256,256),mode='bilinear',align_corners=False)
    for _ in range(10):forward()
    torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(30):
        t=time.perf_counter();forward();torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
    return dict(batch=1,mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),
        additional_peak_mib=(torch.cuda.max_memory_allocated()-base)/1024**2,
        scope='Required DINO + optional CLIP text forward + adapter/Swin/head + 256 output; BF16; excludes tokenizer, I/O, CPU native restore; 10 warmups/30 timed')

def mean_std(values):return dict(mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)))

def paired(rows,a,b):
    reference=rows[42][a];ids=[r['id'] for r in reference];groups=np.array([r['group_id'] for r in reference])
    deltas=[];by_seed={}
    for seed in SEEDS:
        aa,bb=rows[seed][a],rows[seed][b]
        assert [r['id'] for r in aa]==[r['id'] for r in bb]==ids
        d=np.array([x['dice']-y['dice'] for x,y in zip(aa,bb)]);deltas.append(d);by_seed[str(seed)]=float(d.mean())
    d=np.mean(deltas,axis=0);unique=sorted(set(groups));sums=np.array([d[groups==g].sum() for g in unique]);counts=np.array([(groups==g).sum() for g in unique])
    rng=np.random.default_rng(42);choice=rng.integers(len(unique),size=(2000,len(unique)))
    boot=sums[choice].sum(1)/counts[choice].sum(1)
    return dict(per_seed_macro_dice_delta=by_seed,across_seeds=mean_std(list(by_seed.values())),
        source_group_cluster_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),groups=len(unique),
        bootstrap_note='2000 source-group resamples; sample-weighted delta averaged over the three fitted seeds; conditional on these seeds. Groups include one unresolved bjorke source bucket; not a verified patient-population CI.',
        average_seed_per_image_wins=int((d>0).sum()),average_seed_per_image_losses=int((d<0).sum()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    seeds=(42,) if args.smoke else SEEDS
    out=ROOT/('smoke_analysis' if args.smoke else 'analysis');out.mkdir(exist_ok=True)
    locks={}
    for seed in seeds:
        run=ROOT/('smoke' if args.smoke else 'results')/f'seed{seed}'
        done=json.loads((run/'DONE.json').read_text());assert done['passed'] and done['checkpoint_selection_locked']
        for n in NAMES:
            p=run/n/'best_dice.pth';locks[str(p)]=sha(p)
    if not args.smoke:
        locked=ROOT/'selection_lock.json'
        if locked.exists():assert json.loads(locked.read_text())['checkpoint_sha256']==locks
        else:dump(locked,dict(checkpoint_sha256=locks,all_nine_runs_complete_before_any_test_evaluation=True))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data=MedicalData();encoder=FrozenEncoder('dinov3').cuda();splits=('val',) if args.smoke else ('val','test')
    limits={'val':16} if args.smoke else None;cache=FeatureCache(encoder,data,splits=splits,limits=limits)
    for split in splits:cache.encode(split,limit=16 if args.smoke else None)
    results={};test_rows={};table=[]
    for seed in seeds:
        run=ROOT/('smoke' if args.smoke else 'results')/f'seed{seed}'
        done=json.loads((run/'DONE.json').read_text());probes={n:p.cuda() for n,p in build_probes(seed).items()}
        results[str(seed)]={};test_rows[seed]={}
        for n,p in probes.items():
            state=torch.load(run/n/'best_dice.pth',map_location='cpu');p.load_state_dict(state['probe'])
            assert state['epoch']==done['best'][n]['epoch']
            record=dict(selected_epoch=state['epoch'],splits={});results[str(seed)][n]=record
            folder=out/f'seed{seed}'/n;folder.mkdir(parents=True,exist_ok=True)
            for split in splits:
                low,loss=predict(p,cache,split,limit=16 if args.smoke else None)
                metrics,rows=score(low,data,split,full=True);metrics['loss']=loss
                positive=[r for r in rows if r['tp']+r['fn']>0];negative=[r for r in rows if r['tp']+r['fn']==0]
                metrics['positive_gt_subset']=summarize_rows(positive) if positive else None
                metrics['negative_gt_subset']=dict(n=len(negative),images_with_false_positive=sum(r['fp']>0 for r in negative),false_positive_pixels=sum(r['fp'] for r in negative))
                group_records={}
                for g in sorted({r['group_id'] for r in rows}):
                    gm=summarize_rows([r for r in rows if r['group_id']==g]);gm.pop('threshold_jaccard_065',None);group_records[g]=gm
                record['splits'][split]=metrics
                np.save(folder/f'{split}_logits64.npy',low);dump(folder/f'{split}_per_image.json',rows);dump(folder/f'{split}_per_group.json',group_records)
                if split=='test':test_rows[seed][n]=rows
                table.append(dict(seed=seed,model=n,split=split,epoch=state['epoch'],macro_dice=metrics['macro']['dice'],pooled_dice=metrics['pooled']['dice'],
                    macro_iou=metrics['macro']['iou'],pooled_iou=metrics['pooled']['iou'],boundary_f1=metrics['macro']['boundary_f1'],hd95_pixels=metrics['macro']['hd95_pixels']))
                print('EVALUATED',seed,n,split,metrics['macro']['dice'],flush=True)
        del probes
    report=dict(models=results,seeds=list(seeds),dataset='MosMedData+; repaired traceable source-group split',
        checkpoint_sha256=locks,parameters=json.loads((ROOT/'preflight.json').read_text())['parameters'],
        data_manifest_sha256=sha(ROOT/'data_manifest.json'),no_test_checkpoint_selection=True,
        patient_disjointness=json.loads((ROOT/'data_manifest.json').read_text())['patient_disjointness'])
    if not args.smoke:
        report['paired_test_comparisons']={'E2_minus_E1':paired(test_rows,NAMES[2],NAMES[1]),'E2_minus_E0':paired(test_rows,NAMES[2],NAMES[0]),'E1_minus_E0':paired(test_rows,NAMES[1],NAMES[0])}
        aggregates={}
        for n in NAMES:
            aggregates[n]={}
            for section in ('macro','pooled','prec_at'):
                aggregates[n][section]={k:mean_std([results[str(s)][n]['splits']['test'][section][k] for s in SEEDS]) for k in results['42'][n]['splits']['test'][section]}
        report['test_across_seed_statistics']=aggregates
        del cache;torch.cuda.empty_cache()
        from transformers import CLIPTextModel,CLIPTokenizer
        folder=PROJECT/'stage8_text_baselines/assets/clip-vit-base-patch32'
        te=CLIPTextModel.from_pretrained(folder,local_files_only=True).cuda().eval().requires_grad_(False)
        tok=CLIPTokenizer.from_pretrained(folder,local_files_only=True)
        probes={n:p.cuda() for n,p in build_probes(42).items()}
        for n,p in probes.items():p.load_state_dict(torch.load(ROOT/'results/seed42'/n/'best_dice.pth',map_location='cpu')['probe'])
        report['inference_latency']={n:latency(p,encoder,data,te,tok) for n,p in probes.items()}
        lines=['# MosMedData+ 三组文本增益实验','',
            'seeds 42/43/44；冻结 DINOv3 和 CLIP，重新初始化 Adapter/Swin/Head。按来源病例组修复公开切片划分，非原论文直接复现。',
            '文本包含额外病灶数量/位置标注；作者流程用 Mask 质检。当前使用公开文本镜像，未核验与作者原始二进制完全一致。','',
            '| Model | Test macro Dice mean ± SD | Test pooled Dice mean ± SD | BF1 mean ± SD | HD95 px mean ± SD |',
            '|---|---:|---:|---:|---:|']
        for n,a in aggregates.items():
            vals=[a['macro']['dice'],a['pooled']['dice'],a['macro']['boundary_f1'],a['macro']['hd95_pixels']]
            lines.append('| '+n+' | '+' | '.join(f"{v['mean']:.6f} ± {v['sample_std']:.6f}" for v in vals)+' |')
        lines+=['','主比较 E2-E1；E2-E0 为整体收益。E1-E0 同时包含预测头结构差异。',
            '逐 seed 配对差值、来源组聚类区间、macro/pooled 全部指标、阴性切片及推理开销见 summary.json；三个种子不自动证明显著性。',
            '可追溯 CT/来源组无重叠；bjorke 患者映射和跨来源患者身份仍未核实。']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (out/'all_models.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    dump(out/'summary.json',report)
    assert all(sha(path)==h for path,h in locks.items())
    dump(out/'validation.json',dict(passed=True,smoke_only=args.smoke,seeds=list(seeds),models=list(NAMES),native_full_metrics=True,checkpoint_unchanged=True))
    print('EVALUATION_PASSED',flush=True)

if __name__=='__main__':main()
