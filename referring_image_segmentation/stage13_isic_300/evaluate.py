"""Evaluate five fresh 300-epoch models only after all selections are locked."""
from common import *
import argparse,csv,time
import numpy as np,torch
from torch.nn import functional as F
from models import build_probes,FrozenEncoder
from data import MedicalData,FeatureCache
from engine import predict,score

def paired(a,b):
    assert [r['id'] for r in a]==[r['id'] for r in b]
    d=np.array([x['dice']-y['dice'] for x,y in zip(a,b)]);rng=np.random.default_rng(SEED)
    boot=d[rng.integers(len(d),size=(2000,len(d)))].mean(1)
    return dict(macro_dice_delta=float(d.mean()),paired_image_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),
        wins=int((d>0).sum()),losses=int((d<0).sum()),ties=int((d==0).sum()),
        caveat='Single seed; image resampling does not estimate patient or training-seed uncertainty; test set used in historical development')

@torch.no_grad()
def latency(p,enc,data,te,tok):
    r=data.rows['val'][0];idx=torch.tensor([r['index']],device='cuda');valid=data.valid(idx)
    x=data.rgb[idx].permute(0,3,1,2).float()/255.;x=(x*valid+data.mean*(1-valid)-data.mean)/data.std
    tokens=content=None
    if p.name!=NAMES[0]:
        prompt=FIXED_TEXT if p.name in FIXED_NAMES else r['text'];tokens=tok([prompt],padding='max_length',max_length=77,return_tensors='pt')
        content=tokens['attention_mask'].bool()&(tokens['input_ids']!=tok.bos_token_id)&(tokens['input_ids']!=tok.eos_token_id)
        content=content.cuda();tokens={k:v.cuda() for k,v in tokens.items()}
    def forward():
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if tokens is None:text=None
            else:
                output=te(**tokens)
                text=dict(tokens=output.last_hidden_state.float(),valid=content) if p.name in PRE_NAMES else output.pooler_output.float()
            return F.interpolate(p(enc(x),text),(256,256),mode='bilinear',align_corners=False)
    p.eval()
    for _ in range(5):forward()
    torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(20):
        start=time.perf_counter();forward();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    return dict(batch=1,mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),
        additional_peak_mib=(torch.cuda.max_memory_allocated()-base)/1024**2,
        scope='Required DINO + optional CLIP + adapters/fusion/Swin/head +256 output; BF16; excludes tokenizer, I/O, native CPU restore; 5 warmups/20 timed')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run=ROOT/('smoke' if args.smoke else 'results')/'seed42';out=ROOT/('smoke_analysis' if args.smoke else 'analysis');out.mkdir(exist_ok=True)
    done=json.loads((run/'DONE.json').read_text());assert done['passed'] and done['checkpoint_selection_locked']
    locks={n:dict(path=str(run/n/'best_dice.pth'),sha256=sha(run/n/'best_dice.pth')) for n in NAMES}
    if not args.smoke:
        assert done['epochs']==300 and done['updates_per_model']==15300
        dump(ROOT/'selection_lock.json',dict(checkpoints=locks,all_five_300_epoch_runs_complete_before_test=True))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data=MedicalData();enc=FrozenEncoder('dinov3').cuda();splits=('val',) if args.smoke else ('val','test')
    cache=FeatureCache(enc,data,splits=splits)
    for split in splits:cache.encode(split,limit=16 if args.smoke else None)
    probes={n:p.cuda() for n,p in build_probes().items()};models={};records={};table=[]
    for n,p in probes.items():
        state=torch.load(locks[n]['path'],map_location='cpu');p.load_state_dict(state['probe'])
        assert state['epoch']==done['best'][n]['epoch']
        model=dict(selected_epoch=state['epoch'],training_epochs=done['epochs'],fresh_initialization=True,splits={});models[n]=model
        folder=out/n;folder.mkdir(exist_ok=True)
        for split in splits:
            low,loss=predict(p,cache,split,limit=16 if args.smoke else None);metric,rows=score(low,data,split,full=True);metric['loss']=loss
            model['splits'][split]=metric
            if split=='val':assert abs(metric['macro']['dice']-state['metrics']['macro']['dice'])<1e-10
            np.save(folder/f'{split}_logits64.npy',low);dump(folder/f'{split}_per_image.json',rows)
            if split=='test':records[n]=rows
            table.append(dict(model=n,training_epochs=done['epochs'],split=split,selected_epoch=state['epoch'],macro_dice=metric['macro']['dice'],
                pooled_dice=metric['pooled']['dice'],macro_iou=metric['macro']['iou'],pooled_iou=metric['pooled']['iou'],
                boundary_f1=metric['macro']['boundary_f1'],hd95_pixels=metric['macro']['hd95_pixels']))
            print('EVALUATED',n,split,metric['macro']['dice'],'pooled',metric['pooled']['dice'],flush=True)
    lineage=json.loads((ROOT/'lineage.json').read_text())
    report=dict(seed=SEED,training_epochs=done['epochs'],models=models,checkpoints=locks,
        all_models_trained_fresh=True,training_updates_per_model=done['updates_per_model'],
        parameters=json.loads((ROOT/'preflight.json').read_text())['parameters'],dataset='ISIC2016;806/90/379 unchanged',
        patient_disjointness=lineage['patient_disjointness'],test_set_status=lineage['test_set_status'])
    if not args.smoke:
        report['paired_test_comparisons']={
            'late_image_minus_fixed':paired(records[NAMES[2]],records[NAMES[1]]),
            'predecoder_image_minus_fixed':paired(records[NAMES[4]],records[NAMES[3]]),
            'predecoder_image_minus_late_image':paired(records[NAMES[4]],records[NAMES[2]]),
            'late_image_minus_visual':paired(records[NAMES[2]],records[NAMES[0]]),
            'predecoder_image_minus_visual':paired(records[NAMES[4]],records[NAMES[0]])}
        report['reference_100_epoch_results']=lineage['previous_100_epoch_metrics'];budget={}
        for n in NAMES:
            folder=EARLY if n in PRE_NAMES else CONTROLS
            old=json.loads((folder/'analysis'/n/'test_per_image.json').read_text())
            comparison=paired(records[n],old);comparison['reference_path']=str(folder/'analysis'/n/'test_per_image.json');budget[n]=comparison
        report['budget_300_minus_100']=budget
        del cache;torch.cuda.empty_cache()
        from transformers import CLIPTextModel,CLIPTokenizer
        folder=PROJECT/'stage8_text_baselines/assets/clip-vit-base-patch32'
        te=CLIPTextModel.from_pretrained(folder,local_files_only=True).cuda().eval().requires_grad_(False)
        tok=CLIPTokenizer.from_pretrained(folder,local_files_only=True)
        report['inference_latency']={n:latency(p,enc,data,te,tok) for n,p in probes.items()}
        lines=['# ISIC2016：seed42五组统一300轮','',
            '全部五组新初始化训练300轮/15300步，warmup255步和完整300轮cosine。未载入100轮分割权重。','',
            '| Model | 最佳轮 | Val macro Dice | Test macro Dice | Test pooled Dice | BF1 | HD95 px |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for n,m in models.items():
            a=m['splits']['test'];v=m['splits']['val']['macro']['dice']
            lines.append(f"| {n} | {m['selected_epoch']} | {v:.6f} | {a['macro']['dice']:.6f} | {a['pooled']['dice']:.6f} | {a['macro']['boundary_f1']:.6f} | {a['macro']['hd95_pixels']:.3f} |")
        lines+=['','## 历史100轮预算背景','','| Model | 100轮 Test macro Dice | 300轮 Test macro Dice | 差值百分点 |','|---|---:|---:|---:|']
        for n,m in models.items():
            old=lineage['previous_100_epoch_metrics'][n]['splits']['test']['macro']['dice'];new=m['splits']['test']['macro']['dice']
            lines.append(f'| {n} | {old:.6f} | {new:.6f} | {(new-old)*100:+.3f} |')
        lines+=['','相同300轮预算内的固定/逐图配对是主要比较；前后融合比较包含位置、词级表示与输出头差异。',
            '100/300采用不同总长度cosine计划，300轮结果并非从100轮结果续训；只改变预算和对应调度总步数。',
            '单seed、既往开发测试集且患者独立性未验证；不宣称稳定提升或新的外部验证。完整指标、参数、区间和延迟见summary.json。']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (out/'all_models.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    dump(out/'summary.json',report)
    assert all(sha(v['path'])==v['sha256'] for v in locks.values())
    dump(out/'validation.json',dict(passed=True,smoke_only=args.smoke,models=list(NAMES),native_full_metrics=True,
        checkpoints_unchanged=True,new_validation_scores_reproduced=True))
    print('EVALUATION_PASSED',flush=True)

if __name__=='__main__':main()
