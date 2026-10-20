"""One seed, three matched groups; historical E1/E2 are explicitly identified."""
from common import *
import argparse,csv,time
import numpy as np,torch
from torch.nn import functional as F
from models import build_probes,FrozenEncoder
from data import MedicalData,FeatureCache
from engine import predict,score

def paired(a,b):
    assert [r['id'] for r in a]==[r['id'] for r in b]
    d=np.array([x['dice']-y['dice'] for x,y in zip(a,b)]);g=np.random.default_rng(SEED)
    boot=d[g.integers(len(d),size=(2000,len(d)))].mean(1)
    return dict(macro_dice_delta=float(d.mean()),paired_image_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),
        wins=int((d>0).sum()),losses=int((d<0).sum()),ties=int((d==0).sum()),
        caveat='Single seed; image resampling does not estimate training-seed or patient-level uncertainty. Test set already used during historical development.')

@torch.no_grad()
def latency(p,enc,data,te,tok):
    r=data.rows['val'][0];idx=torch.tensor([r['index']],device='cuda');valid=data.valid(idx)
    x=data.rgb[idx].permute(0,3,1,2).float()/255.;x=(x*valid+data.mean*(1-valid)-data.mean)/data.std
    tokens=None
    if p.name!=NAMES[0]:
        text=FIXED_TEXT if p.name==NAMES[1] else r['text']
        tokens={k:v.cuda() for k,v in tok([text],padding='max_length',max_length=77,return_tensors='pt').items()}
    def forward():
        with torch.autocast('cuda',dtype=torch.bfloat16):
            t=None if tokens is None else te(**tokens).pooler_output.float()
            return F.interpolate(p(enc(x),t),(256,256),mode='bilinear',align_corners=False)
    p.eval()
    for _ in range(5):forward()
    torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(20):
        start=time.perf_counter();forward();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    return dict(batch=1,mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),
        additional_peak_mib=(torch.cuda.max_memory_allocated()-base)/1024**2,
        scope='Required DINO + optional CLIP text forward + adapter/Swin/head +256 output; BF16; excludes tokenizer, I/O, native CPU restore; 5 warmups/20 timed')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run=ROOT/('smoke' if args.smoke else 'results')/'seed42';out=ROOT/('smoke_analysis' if args.smoke else 'analysis');out.mkdir(exist_ok=True)
    done=json.loads((run/'DONE.json').read_text());assert done['passed'] and done['checkpoint_selection_locked']
    lineage=json.loads((ROOT/'lineage.json').read_text());checkpoint={NAMES[0]:run/NAMES[0]/'best_dice.pth'}
    if not args.smoke:
        checkpoint.update({n:Path(v['path']) for n,v in lineage['reused_checkpoints'].items()})
    locked={n:dict(path=str(p),sha256=sha(p)) for n,p in checkpoint.items()}
    if not args.smoke:dump(ROOT/'selection_lock.json',dict(checkpoints=locked,new_E0_selection_locked_before_its_test_evaluation=True,historical_test_already_seen=True))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data=MedicalData();enc=FrozenEncoder('dinov3').cuda();splits=('val',) if args.smoke else ('val','test')
    cache=FeatureCache(enc,data,splits=splits)
    for split in splits:cache.encode(split,limit=16 if args.smoke else None)
    probes=build_probes();models={};records={};table=[]
    for n,path in checkpoint.items():
        state=torch.load(path,map_location='cpu');p=probes[n].cuda();p.load_state_dict(state['probe'])
        report=dict(selected_epoch=state['epoch'],newly_trained=n==NAMES[0],source_checkpoint=str(path),splits={});models[n]=report
        folder=out/n;folder.mkdir(exist_ok=True)
        for split in splits:
            low,loss=predict(p,cache,split,limit=16 if args.smoke else None);m,rows=score(low,data,split,full=True);m['loss']=loss
            report['splits'][split]=m;np.save(folder/f'{split}_logits64.npy',low);dump(folder/f'{split}_per_image.json',rows)
            if split=='test':records[n]=rows
            if n in REUSED:
                expected=lineage['reused_metrics'][n]['splits'][split]['macro']['dice']
                assert abs(m['macro']['dice']-expected)<1e-10,(n,split,m['macro']['dice'],expected)
            table.append(dict(model=n,split=split,epoch=state['epoch'],newly_trained=n==NAMES[0],macro_dice=m['macro']['dice'],
                pooled_dice=m['pooled']['dice'],macro_iou=m['macro']['iou'],pooled_iou=m['pooled']['iou'],
                boundary_f1=m['macro']['boundary_f1'],hd95_pixels=m['macro']['hd95_pixels']))
            print('EVALUATED',n,split,m['macro']['dice'],'pooled',m['pooled']['dice'],flush=True)
    report=dict(seed=SEED,dataset='ISIC2016; unchanged deduplicated stage9 split 806/90/379',models=models,checkpoints=locked,
        newly_trained=[NAMES[0]],historical_reused=list(REUSED),historical_scores_reproduced=True,
        parameters=json.loads((ROOT/'preflight.json').read_text())['parameters'],
        patient_disjointness='not established',test_set_status=lineage['test_set_status'],
        text_scope='VQA-generated color/shape attributes; no GT size/count/location; not original clinical reports or multi-target RIS')
    if not args.smoke:
        report['paired_test_comparisons']={'E2_minus_E1':paired(records[NAMES[2]],records[NAMES[1]]),
            'E2_minus_E0':paired(records[NAMES[2]],records[NAMES[0]]),'E1_minus_E0':paired(records[NAMES[1]],records[NAMES[0]])}
        del cache;torch.cuda.empty_cache()
        from transformers import CLIPTextModel,CLIPTokenizer
        folder=PROJECT/'stage8_text_baselines/assets/clip-vit-base-patch32'
        te=CLIPTextModel.from_pretrained(folder,local_files_only=True).cuda().eval().requires_grad_(False)
        tok=CLIPTokenizer.from_pretrained(folder,local_files_only=True)
        report['inference_latency']={n:latency(p.cuda(),enc,data,te,tok) for n,p in probes.items()}
        lines=['# ISIC2016：seed42 三组文本对照','',
            'E0无文本为本次新训练；E1固定文本与E2逐图属性文本复用stage9已完成的同条件权重，并重新核验原尺寸验证/测试分数。',
            '806训练/90验证/379测试，100轮、batch16、原亮度/对比度增强，选模仅按验证macro Dice。','',
            '| Model | 来源 | 最佳轮 | Val macro Dice | Test macro Dice | Test pooled Dice | Test BF1 | Test HD95 px |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for n,m in models.items():
            a=m['splits']['test'];v=m['splits']['val']['macro']['dice'];label='本次训练' if n==NAMES[0] else '历史权重重放'
            lines.append(f"| {n} | {label} | {m['selected_epoch']} | {v:.6f} | {a['macro']['dice']:.6f} | {a['pooled']['dice']:.6f} | {a['macro']['boundary_f1']:.6f} | {a['macro']['hd95_pixels']:.3f} |")
        lines+=['','主要文本比较E2-E1；E2-E0衡量整体收益。E1-E0含输出头结构差异。单seed，不将小差异自动解释为稳定提升。',
            '完整macro/pooled区域指标、Prec@各阈值、ISIC threshold Jaccard、边界指标和配对差值见summary.json。',
            '文本仅颜色/形状属性；患者独立性未验证，测试集曾用于stage9开发，因此本次不称新的外部验证。']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (out/'all_models.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    dump(out/'summary.json',report)
    assert all(sha(v['path'])==v['sha256'] for v in locked.values())
    dump(out/'validation.json',dict(passed=True,smoke_only=args.smoke,models=list(models),native_full_metrics=True,historical_replay_matches=True,checkpoints_unchanged=True))
    print('EVALUATION_PASSED',flush=True)

if __name__=='__main__':main()
