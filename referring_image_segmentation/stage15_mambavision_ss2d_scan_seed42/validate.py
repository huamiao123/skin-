"""Independent completion audit: full budget, selectors, pixels, metrics and report."""
from common import *
from analyze import load_gt
from evaluation import summarize_rows
import csv

def main():
    verify_frozen_sources();out=ROOT/'analysis'
    config=json.loads((ROOT/'config.json').read_text());done=json.loads((ROOT/'results/DONE.json').read_text())
    history=json.loads((ROOT/'results/history.json').read_text());summary=json.loads((out/'summary.json').read_text())
    preflight=json.loads((ROOT/'preflight.json').read_text());evaluation=json.loads((out/'evaluation_validation.json').read_text())
    assert config['seed']==42 and config['epochs']==300 and config['total_updates']==9000
    assert done['seed']==42 and done['epochs']==300 and done['updates']==9000 and not done['smoke_only']
    assert len(history)==300 and [r['epoch'] for r in history]==list(range(1,301))
    assert all(r['epoch_updates']==30 and r['optimizer_updates']==r['epoch']*30 for r in history)
    assert all(abs(r['lr']-(1e-5+(.001-1e-5)*(1+np.cos(np.pi*(r['epoch']-1)/300))/2))<1e-14 for r in history)
    assert abs(history[-1]['next_lr']-1e-5)<1e-14
    assert done['initial_encoder_hash']==done['final_encoder_hash']==preflight['encoder_hash']
    assert done['encoder_unchanged_including_bn_buffers'] and evaluation['passed'] and preflight['passed']
    assert summary['seed']==42 and summary['epochs']==300 and summary['optimizer_updates']==9000
    assert sha(ROOT/'official/hf/model.safetensors')==WEIGHT_SHA256
    assert summary['parameters']==config['parameters'] and summary['pretrained']==config['pretrained']
    assert json.loads((ROOT/'routing_preflight.json').read_text())['whole_encoder_single_route_exact']
    assert json.loads((ROOT/'baseline_reuse.json').read_text())['passed']
    assert config['pretrained']['added_parameters']==0 and config['pretrained']['full_vmamba_ss2d_module'] is False
    assert 'mambavision_original' in summary['comparisons']
    expected=dict(best_dice=max(history,key=lambda r:r['val']['macro']['dice'])['epoch'],
        best_loss=min(history,key=lambda r:r['val']['loss'])['epoch'],last=300)
    report=(out/'REPORT.md').read_text();checks=[]
    ids=json.loads((ROOT/'results/best_dice/ids.json').read_text());gt=load_gt(ids)
    assert len(ids)==808 and len(set(ids))==808
    manifest=json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())
    assert ids==[r['id'] for r in manifest['members'] if r['split']=='val']
    for criterion in CRITERIA:
        model=summary['models'][criterion];rows=json.loads((out/f'per_image_{criterion}.json').read_text())
        assert len(rows)==808 and [r['id'] for r in rows]==ids
        assert model['selected_epoch']==expected[criterion]
        ckpt=torch.load(ROOT/'results'/f'{criterion}.pth',map_location='cpu')
        assert ckpt['epoch']==expected[criterion] and sha(ROOT/'results'/f'{criterion}.pth')==done['checkpoints'][criterion]['sha256']
        assert model['checkpoint_sha256']==done['checkpoints'][criterion]['sha256']
        pred=np.load(ROOT/'results'/criterion/'masks.npy',mmap_mode='r')
        probability=np.load(ROOT/'results'/criterion/'probabilities.npy',mmap_mode='r')
        assert pred.shape==probability.shape==(808,256,256)
        assert sha(ROOT/'results'/criterion/'probabilities.npy')==model['probability_sha256']
        for i,row in enumerate(rows):
            a=pred[i].astype(bool);b=gt[i];assert np.array_equal(a,probability[i]>=.5)
            tp=int((a&b).sum());fp=int((a&~b).sum());fn=int((~a&b).sum());tn=int((~a&~b).sum())
            assert (tp,fp,fn,tn)==tuple(row[k] for k in ('tp','fp','fn','tn'))
            assert abs(row['dice']-(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1))<1e-12
            assert abs(row['iou']-(tp/(tp+fp+fn) if tp+fp+fn else 1))<1e-12
        aggregate=summarize_rows(rows)
        for group in ('macro','pooled','prec_at','counts'):
            for key,value in aggregate[group].items():assert abs(value-model[group][key])<1e-12
        assert abs(aggregate['threshold_jaccard_065']-model['threshold_jaccard_065'])<1e-12
        assert f'| {criterion} | {model["selected_epoch"]} | {model["macro"]["dice"]:.6f} | {model["pooled"]["dice"]:.6f} |' in report
        checks.append(dict(criterion=criterion,epoch=expected[criterion],n=808,
            pixel_counts_match_saved_predictions=True,independent_dice_iou_passed=True,
            all_aggregates_match=True,report_matches_json=True))
    with (out/'per_image.csv').open() as source:assert len(list(csv.DictReader(source)))==808
    for name in ('learning_curves.png','per_image_distributions.png','historical_comparison_per_image.csv','inference.json'):
        assert (out/name).stat().st_size>0
    assert json.loads((ROOT/'smoke_analysis/evaluation_validation.json').read_text())['passed']
    assert json.loads((ROOT/'smoke/resume_audit.json').read_text())['python_numpy_cpu_cuda_rng_exact']
    audit=dict(passed=True,seed=42,epochs=300,optimizer_updates=9000,validation_n=808,
        official_weight_identity_verified=True,encoder_unchanged_including_bn_buffers=True,
        all_three_selectors_match_full_history=True,all_three_checkpoints_retained=True,
        complete_rng_optimizer_scheduler_smoke_resume_passed=True,checks=checks,
        historical_and_new_frozen_sources_unchanged=True,report_and_json_consistent=True,
        evaluation_split='development validation, not official independent test')
    dump(ROOT/'validation.json',audit);dump(out/'independent_report_check.json',audit)
    # Convenience aliases, preserving original historical paths.
    dump(ROOT/'history.json',history)
    readme=(ROOT/'README.md').read_text()
    if '\n## 完成结果\n' not in readme:
        readme+='\n## 完成结果\n\n300轮/9000更新及全部独立校验已通过。主权重按验证macro Dice选模，完整结果见 [报告](analysis/REPORT.md) 与 [summary.json](analysis/summary.json)。\n\n'
        m=summary['models']['best_dice'];readme+=f'best_dice 第{m["selected_epoch"]}轮，开发验证 macro Dice {m["macro"]["dice"]*100:.4f}%，pooled Dice {m["pooled"]["dice"]*100:.4f}%。单seed开发验证结果，不作独立测试或架构机制归因。\n'
        (ROOT/'README.md').write_text(readme)
    print('INDEPENDENT_COMPLETION_AUDIT_PASSED',flush=True)

if __name__=='__main__':main()
