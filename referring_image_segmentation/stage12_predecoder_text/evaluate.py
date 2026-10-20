"""Locked evaluation of the two new predecoder models against historical controls."""
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
    bootstrap=d[g.integers(len(d),size=(2000,len(d)))].mean(1)
    return dict(macro_dice_delta=float(d.mean()),paired_image_bootstrap95=np.percentile(bootstrap,[2.5,97.5]).tolist(),
        wins=int((d>0).sum()),losses=int((d<0).sum()),ties=int((d==0).sum()),
        caveat='Single exploratory seed; image resampling does not measure seed/patient uncertainty; test set used in past development')

@torch.no_grad()
def latency(p,enc,data,te,tok):
    r=data.rows['val'][0];idx=torch.tensor([r['index']],device='cuda');valid=data.valid(idx)
    x=data.rgb[idx].permute(0,3,1,2).float()/255.;x=(x*valid+data.mean*(1-valid)-data.mean)/data.std
    prompt=FIXED_TEXT if p.name==NAMES[0] else r['text']
    tokens=tok([prompt],padding='max_length',max_length=77,return_tensors='pt')
    content=tokens['attention_mask'].bool()&(tokens['input_ids']!=tok.bos_token_id)&(tokens['input_ids']!=tok.eos_token_id)
    content=content.cuda();tokens={k:v.cuda() for k,v in tokens.items()};p.eval()
    def forward():
        with torch.autocast('cuda',dtype=torch.bfloat16):
            words=te(**tokens).last_hidden_state.float()
            return F.interpolate(p(enc(x),dict(tokens=words,valid=content)),(256,256),mode='bilinear',align_corners=False)
    for _ in range(5):forward()
    torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(20):
        start=time.perf_counter();forward();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    return dict(batch=1,mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),
        additional_peak_mib=(torch.cuda.max_memory_allocated()-base)/1024**2,
        scope='CLIP word forward + DINO + adapters/predecoder cross-attention/Swin/ordinary mask head +256 output; BF16; excludes tokenizer, I/O, native CPU restore; 5 warmups/20 timed')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run=ROOT/('smoke' if args.smoke else 'results')/'seed42';out=ROOT/('smoke_analysis' if args.smoke else 'analysis');out.mkdir(exist_ok=True)
    done=json.loads((run/'DONE.json').read_text());assert done['passed'] and done['checkpoint_selection_locked']
    locks={n:dict(path=str(run/n/'best_dice.pth'),sha256=sha(run/n/'best_dice.pth')) for n in NAMES}
    lineage=json.loads((ROOT/'lineage.json').read_text())
    if not args.smoke:dump(ROOT/'selection_lock.json',dict(new_checkpoints=locks,previous_checkpoints=lineage['previous_checkpoints'],all_new_models_complete_before_test=True))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data=MedicalData();enc=FrozenEncoder('dinov3').cuda();splits=('val',) if args.smoke else ('val','test')
    cache=FeatureCache(enc,data,splits=splits)
    for s in splits:cache.encode(s,limit=16 if args.smoke else None)
    probes={n:p.cuda() for n,p in build_probes().items()};models={};records={};table=[]
    for n,p in probes.items():
        state=torch.load(locks[n]['path'],map_location='cpu');p.load_state_dict(state['probe'])
        assert state['epoch']==done['best'][n]['epoch']
        model=dict(selected_epoch=state['epoch'],newly_trained=True,interaction='word cross-attention before Swin; ordinary visual-only mask head',splits={});models[n]=model
        folder=out/n;folder.mkdir(exist_ok=True)
        for split in splits:
            low,loss=predict(p,cache,split,limit=16 if args.smoke else None);m,rows=score(low,data,split,full=True);m['loss']=loss
            model['splits'][split]=m
            if split=='val':assert abs(m['macro']['dice']-state['metrics']['macro']['dice'])<1e-10
            np.save(folder/f'{split}_logits64.npy',low);dump(folder/f'{split}_per_image.json',rows)
            if split=='test':records[n]=rows
            table.append(dict(model=n,newly_trained=True,split=split,epoch=state['epoch'],macro_dice=m['macro']['dice'],pooled_dice=m['pooled']['dice'],
                macro_iou=m['macro']['iou'],pooled_iou=m['pooled']['iou'],boundary_f1=m['macro']['boundary_f1'],hd95_pixels=m['macro']['hd95_pixels']))
            print('EVALUATED',n,split,m['macro']['dice'],'pooled',m['pooled']['dice'],flush=True)
    report=dict(seed=SEED,dataset='ISIC2016; unchanged 806/90/379 split and P3 color/shape text',models=models,new_checkpoints=locks,
        parameters=json.loads((ROOT/'preflight.json').read_text())['parameters'],no_postdecoder_text_fusion=True,
        patient_disjointness=lineage['patient_disjointness'],test_set_status=lineage['test_set_status'],
        comparison_limit='Relative to historical late fusion, interaction location, token granularity and output head all change. Only the new fixed/image-text pair has matched structure and capacity.')
    if not args.smoke:
        old_records={}
        for n,m in lineage['previous_metrics'].items():
            model=json.loads(json.dumps(m));model['newly_trained']=False;model['source_stage']=str(CONTROLS);models[n]=model
            old_records[n]=json.loads((CONTROLS/'analysis'/n/'test_per_image.json').read_text())
            assert [r['id'] for r in old_records[n]]==[r['id'] for r in records[NAMES[0]]]
            for split in ('val','test'):
                a=m['splits'][split];table.append(dict(model=n,newly_trained=False,split=split,epoch=m['selected_epoch'],macro_dice=a['macro']['dice'],
                    pooled_dice=a['pooled']['dice'],macro_iou=a['macro']['iou'],pooled_iou=a['pooled']['iou'],boundary_f1=a['macro']['boundary_f1'],hd95_pixels=a['macro']['hd95_pixels']))
        report['paired_test_comparisons']={
            'predecoder_image_minus_fixed':paired(records[NAMES[1]],records[NAMES[0]]),
            'predecoder_image_minus_late_image':paired(records[NAMES[1]],old_records['E2_image_text']),
            'predecoder_image_minus_visual':paired(records[NAMES[1]],old_records['E0_visual']),
            'predecoder_fixed_minus_late_fixed':paired(records[NAMES[0]],old_records['E1_fixed_text'])}
        report['historical_checkpoints']=lineage['previous_checkpoints']
        del cache;torch.cuda.empty_cache()
        from transformers import CLIPTextModel,CLIPTokenizer
        folder=PROJECT/'stage8_text_baselines/assets/clip-vit-base-patch32'
        te=CLIPTextModel.from_pretrained(folder,local_files_only=True).cuda().eval().requires_grad_(False);tok=CLIPTokenizer.from_pretrained(folder,local_files_only=True)
        report['inference_latency']={n:latency(p,enc,data,te,tok) for n,p in probes.items()}
        lines=['# ISIC2016 seed42：Decoder之前的词级交互','',
            '文本只进入Decoder之前的四层Cross-Attention，Decoder之后是普通Mask Head。固定文本和逐图文本均新训练100轮；另外三个历史对照只读复用。','',
            '| Model | 新训练 | 最佳轮 | Val macro Dice | Test macro Dice | Test pooled Dice | Test BF1 | Test HD95 px |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for n,m in models.items():
            a=m['splits']['test'];v=m['splits']['val']['macro']['dice'];source='是' if m['newly_trained'] else '历史'
            lines.append(f"| {n} | {source} | {m['selected_epoch']} | {v:.6f} | {a['macro']['dice']:.6f} | {a['pooled']['dice']:.6f} | {a['macro']['boundary_f1']:.6f} | {a['macro']['hd95_pixels']:.3f} |")
        lines+=['','主要比较是predecoder_image_text - predecoder_fixed_text，解释同容量下逐图属性的额外价值。',
            '与旧late-image比较是整体机制替换，包含位置/词级表示/输出头差异，不作为单独位置收益。',
            '只有一个seed；既往使用过的测试集、未验证患者独立性，不宣称稳定提升或全新泛化验证。',
            '完整macro/pooled区域指标、Prec@、threshold Jaccard、边界指标、参数量与延迟见summary.json。']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (out/'all_models.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    dump(out/'summary.json',report)
    assert all(sha(v['path'])==v['sha256'] for v in locks.values())
    for v in lineage['previous_checkpoints'].values():assert sha(v['path'])==v['sha256']
    dump(out/'validation.json',dict(passed=True,smoke_only=args.smoke,new_models=list(NAMES),native_full_metrics=True,
        checkpoints_unchanged=True,no_postdecoder_text_fusion=True,new_validation_scores_reproduced=True))
    print('EVALUATION_PASSED',flush=True)

if __name__=='__main__':main()
