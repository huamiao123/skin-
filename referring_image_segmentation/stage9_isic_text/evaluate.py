"""One locked-checkpoint test evaluation with text interventions and paired analysis."""
from common import *
import argparse,time,csv
import numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from models import build_probes,EncoderPair
from data import MedicalData,FeatureCache
from engine import predict,score,restore

def paired(a,b):
    assert [x['id'] for x in a]==[x['id'] for x in b]
    delta=np.array([x['dice']-y['dice'] for x,y in zip(a,b)])
    g=np.random.default_rng(SEED)
    boot=delta[g.integers(len(delta),size=(2000,len(delta)))].mean(1)
    return dict(macro_dice_delta=float(delta.mean()),image_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),
        wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()),
        note='Single seed; paired image bootstrap, not seed uncertainty or verified patient-level confidence')

@torch.no_grad()
def latency(probe,encoder,data,text_encoder,tokenizer):
    r=data.rows['val'][0];idx=torch.tensor([r['index']],device='cuda')
    x=data.rgb[idx].permute(0,3,1,2).float()/255.;valid=data.valid(idx)
    x=(x*valid+data.mean*(1-valid)-data.mean)/data.std
    text=FIXED_TEXT if probe.name=='dinov3_fixed_text' else r['text']
    tokens={k:v.cuda() for k,v in tokenizer([text],padding='max_length',max_length=77,return_tensors='pt').items()}
    probe.eval()
    def forward():
        with torch.autocast('cuda',dtype=torch.bfloat16):
            t=text_encoder(**tokens).pooler_output.float()
            if probe.name=='fusion_text':fs=encoder(x)
            elif probe.name=='vmamba_text':fs=[None]*4+encoder.vmamba(x)
            else:fs=encoder.dinov3(x)
            return F.interpolate(probe(fs,t),(256,256),mode='bilinear',align_corners=False)
    for _ in range(5):forward()
    torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(20):
        start=time.perf_counter();forward();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    return dict(batch=1,mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),
        additional_forward_peak_mib=(torch.cuda.max_memory_allocated()-baseline)/1024**2,
        scope='CLIP text forward + required visual encoder(s) + adapters/fusion/Swin/matching + 256 output; BF16; excludes tokenization, disk I/O, CPU native-size restoration; 5 warmups/20 timed')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run=ROOT/('smoke' if args.smoke else 'results')/'seed42'
    out=ROOT/('smoke_analysis' if args.smoke else 'analysis');out.mkdir(parents=True,exist_ok=True)
    done=json.loads((run/'DONE.json').read_text());assert done['passed'] and done['checkpoint_selection_locked']
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data=MedicalData();encoder=EncoderPair().cuda()
    eval_splits=('val',) if args.smoke else ('val','test')
    cache=FeatureCache(encoder,data,splits=eval_splits)
    for split in eval_splits:cache.encode(split,limit=16 if args.smoke else None)
    probes={n:p.cuda() for n,p in build_probes().items()}
    results={};test_rows={};predictions={};checkpoint_hashes={}
    for n,p in probes.items():
        path=run/n/'best_dice.pth';checkpoint_hashes[n]=sha(path)
        checkpoint=torch.load(path,map_location='cpu');p.load_state_dict(checkpoint['probe'])
        assert checkpoint['epoch']==done['best'][n]['epoch']
        results[n]=dict(selected_epoch=checkpoint['epoch'],splits={})
        folder=out/n;folder.mkdir(exist_ok=True)
        for split in eval_splits:
            logits,loss=predict(p,cache,split,limit=16 if args.smoke else None)
            metrics,rows=score(logits,data,split,full=True);metrics['loss']=loss
            results[n]['splits'][split]=metrics
            np.save(folder/f'{split}_logits64.npy',logits.astype(np.float16));dump(folder/f'{split}_per_image.json',rows)
            if split=='test':test_rows[n]=rows;predictions[n]=logits
            print('EVALUATED',n,split,metrics['macro']['dice'],flush=True)
        if args.smoke:continue
        results[n]['text_diagnostics']={}
        if n!='dinov3_fixed_text':
            for intervention in ('fixed','shuffle'):
                logits,_=predict(p,cache,'test',intervention=intervention)
                metrics,rows=score(logits,data,'test')
                changes=total=0
                for i,r in enumerate(data.rows['test']):
                    changed=(restore(logits[i],r)>=0)!=(restore(predictions[n][i],r)>=0)
                    changes+=int(changed.sum());total+=changed.size
                metrics['changed_pixel_fraction_vs_correct_text']=changes/total
                metrics['macro_dice_drop_vs_correct_text']=results[n]['splits']['test']['macro']['dice']-metrics['macro']['dice']
                results[n]['text_diagnostics'][intervention]=metrics
                dump(folder/f'test_{intervention}_per_image.json',rows)
    report=dict(seed=SEED,dataset='ISIC2016; deduplicated MedVLSM split; normalized P3 color/shape text',
        single_seed_exploratory=True,models=results,checkpoint_sha256=checkpoint_hashes,
        parameters=json.loads((ROOT/'preflight.json').read_text())['parameters'],
        no_test_checkpoint_selection=True,patient_disjointness='not established',
        prompt_limit='Template attributes supplied at inference; color/shape may be imperfect; no GT-derived size/count/location; not multi-object natural-language RIS')
    if not args.smoke:
        report['paired_test_comparisons']={
            'fusion_minus_dinov3':paired(test_rows['fusion_text'],test_rows['dinov3_text']),
            'fusion_minus_vmamba':paired(test_rows['fusion_text'],test_rows['vmamba_text']),
            'dinov3_text_minus_fixed_text':paired(test_rows['dinov3_text'],test_rows['dinov3_fixed_text'])}
        # Stable ID-order examples, never used for checkpoint or protocol selection.
        examples=out/'examples';examples.mkdir(exist_ok=True)
        for i,r in enumerate(data.rows['test'][:8]):
            panel=Image.new('RGB',(256*6,256))
            panel.paste(Image.open(r['image_path']).convert('RGB').resize((256,256)),(0,0))
            panel.paste(Image.open(r['mask_path']).convert('RGB').resize((256,256),Image.Resampling.NEAREST),(256,0))
            for col,n in enumerate(NAMES,2):
                image=Image.fromarray((restore(predictions[n][i],r)>=0).astype(np.uint8)*255)
                panel.paste(image.resize((256,256),Image.Resampling.NEAREST).convert('RGB'),(256*col,0))
            panel.save(examples/f"{r['id']}.png")
        del cache;torch.cuda.empty_cache()
        from transformers import CLIPTextModel,CLIPTokenizer
        folder=PROJECT/'stage8_text_baselines/assets/clip-vit-base-patch32'
        te=CLIPTextModel.from_pretrained(folder,local_files_only=True).cuda().eval().requires_grad_(False)
        tok=CLIPTokenizer.from_pretrained(folder,local_files_only=True)
        report['inference_latency']={n:latency(p,encoder,data,te,tok) for n,p in probes.items()}
        lines=['# ISIC2016 医学文本引导分割：seed 42','',
            '独立于已停止的 RefCOCO 实验。三视觉分支均冻结，P3 仅颜色/形状；类别词统一为 skin lesion。',
            '本次为去重后的作者划分，图像级无精确重复，不宣称患者级独立。最佳权重仅由验证集原尺寸 macro Dice 选择。','',
            '| Model | Epoch | Test macro Dice | Test pooled Dice | Test macro IoU | Boundary F1 | HD95 px |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for n,r in results.items():
            m=r['splits']['test'];a=m['macro'];lines.append(f"| {n} | {r['selected_epoch']} | {a['dice']:.6f} | {m['pooled']['dice']:.6f} | {a['iou']:.6f} | {a['boundary_f1']:.6f} | {a['hd95_pixels']:.3f} |")
        lines+=['','固定文本训练对照只与 DINOv3 架构匹配；三主组的临时固定/错配文本结果是诊断，不等同于额外训练的对照。',
            '完整 macro/pooled Dice/IoU/Precision/Recall、Prec@0.5/0.6/0.7/0.8/0.9、threshold Jaccard、BF1、HD95 及配对差值见 summary.json。',
            '单 seed 不能估计训练随机性；图像 bootstrap 区间也不代表患者级泛化。']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(out/'summary.json',report)
    assert all(sha(run/n/'best_dice.pth')==h for n,h in checkpoint_hashes.items())
    dump(out/'validation.json',dict(passed=True,smoke_only=args.smoke,models=list(NAMES),splits=list(eval_splits),
        checkpoint_unchanged=True,native_full_metrics=True,per_image_records=True))
    print('EVALUATION_PASSED',flush=True)

if __name__=='__main__':main()
