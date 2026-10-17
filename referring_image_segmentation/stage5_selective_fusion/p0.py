"""P0: existing predictions only; no fitting, threshold tuning or GT routing."""
import json
from diagnostics import ROOT,PROJECT,context,references,pixel_diagnostics,dump,sha


def main():
    out=ROOT/'p0';out.mkdir(exist_ok=True)
    ids,gt,regions=context();results={};inputs={}
    old=json.loads((PROJECT/'stage4_fusion/analysis/summary.json').read_text())
    for criterion in ('best_dice','best_loss','last'):
        ref=references(criterion,ids);results[criterion]={}
        for name in ('fusion','mean'):
            summary,rows=pixel_diagnostics(ids,gt,regions,ref['dinov3'],ref['vmamba'],ref[name])
            expected=old['models'][criterion][name+'_swin']['counts']
            assert summary['all']['candidate_wrong']==expected['fp']+expected['fn']
            results[criterion][name]=summary;dump(out/f'{criterion}_{name}_per_image.json',rows)
        for name in ('dinov3','vmamba','fusion'):
            p=PROJECT/f'stage4_fusion/results/seed42/{name}_swin/{criterion}/probabilities.npy'
            inputs[str(p)]=sha(p)
        print('P0',criterion,results[criterion]['fusion']['all'],flush=True)
    strata=json.loads((PROJECT/'stage4_fusion/analysis/strata.json').read_text())
    dump(out/'reference_strata.json',[s for s in strata if s['decoder']=='swin'])
    report=dict(passed=True,seed=42,split='ISIC2018 legacy development validation',results=results,input_sha256=inputs,
        caveats=['Foreground probability is not calibrated confidence.','GT groups describe errors only; no GT enters inference.',
                 'No labels for blurred boundaries/hair/reflections.','Baseline performance strata are selected on baseline outcomes.'])
    dump(out/'summary.json',report)
    s=results['best_dice']['fusion']
    lines=['# P0：已有Swin融合的错误来源','',
        '使用第四轮已保存预测，未重训、未调阈值。下表是最高验证mean Dice checkpoint；全部三种checkpoint及概率平均见summary.json。','',
        '| 区域 | 像素数 | DINO错误 | 纠正 | 新增错误 | 修复率 | 破坏率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n,d in dict(all=s['all'],**s['regions']).items():
        lines.append(f"| {n} | {d['pixels']} | {d['dino_wrong']} | {d['corrected']} | {d['introduced']} | {d['repair_rate']:.4%} | {d['damage_rate']:.4%} |")
    lines+=['','## DINO前景概率区间','', '| 区间 | 像素数 | 纠正 | 新增 | 修复率 | 破坏率 |','|---|---:|---:|---:|---:|---:|']
    for n,d in s['probability'].items():
        lines.append(f"| {n} | {d['pixels']} | {d['corrected']} | {d['introduced']} | {d['repair_rate']:.4%} | {d['damage_rate']:.4%} |")
    both=s['correctness']['both_correct'];other=s['correctness']['dino_correct_vmamba_wrong']
    lines+=['',f"两个单模型都正确的位置产生{both['introduced']}个新增错误，占全部新增错误{both['introduced']/s['all']['introduced']:.2%}。该区域破坏率{both['damage_rate']:.4%}；DINO对而VMamba错的区域破坏率{other['damage_rate']:.4%}。",
        '', '修复率分母是原错误；破坏率分母是原正确，不能直接相减。大量背景像素会影响绝对计数，应同时看区域比例和逐图指标。',
        '结果支持同时检查容量、残差主路径及动态选择；不直接证明某种特征是错误原因。模糊边界等缺少标签，不作自动病例归因。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('P0 PASSED',flush=True)


if __name__=='__main__':main()
