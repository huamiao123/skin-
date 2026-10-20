"""Independently check the report counts and write a concise factual interpretation."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT.parent/'stage9_isic_text'

def main():
    report=json.loads((ROOT/'analysis/summary.json').read_text())
    old=json.loads((PREVIOUS/'analysis/summary.json').read_text())
    history=json.loads((ROOT/'results/seed42/history.json').read_text())
    selected=max(history,key=lambda r:r['val']['macro']['dice'])
    assert selected['epoch']==report['models']['E0_visual']['selected_epoch']
    checks=[]
    for name,model in report['models'].items():
        for split,metrics in model['splits'].items():
            rows=json.loads((ROOT/'analysis'/name/f'{split}_per_image.json').read_text())
            tp=np.array([r['tp'] for r in rows],np.int64);fp=np.array([r['fp'] for r in rows],np.int64);fn=np.array([r['fn'] for r in rows],np.int64)
            mean=float(np.mean(2*tp/(2*tp+fp+fn)));pooled=float(2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum()))
            assert abs(mean-metrics['macro']['dice'])<1e-12
            assert abs(pooled-metrics['pooled']['dice'])<1e-12
            if name!='E0_visual':
                source='dinov3_fixed_text' if name=='E1_fixed_text' else 'dinov3_text'
                expected=old['models'][source]['splits'][split]
                for section in ('macro','pooled','prec_at'):
                    for key,value in metrics[section].items():assert abs(value-expected[section][key])<1e-10,(name,split,section,key)
            checks.append(dict(model=name,split=split,n=len(rows),independent_macro_dice=mean,independent_pooled_dice=pooled))
    values={n:m['splits']['test']['macro']['dice'] for n,m in report['models'].items()}
    delta=report['paired_test_comparisons']
    lines=['# ISIC2016 三组对照的结果解释','',
        '本次只新增seed42无文本E0，训练100轮5100次更新。E1/E2为历史同配置已完成实验，重新推理核验全部macro/pooled与Prec@指标一致。六张模型/划分指标表由逐图像素计数独立重算Dice通过，E0最佳轮与完整验证历史最大值一致。','',
        f"E0测试macro Dice {values['E0_visual']*100:.3f}%；固定文本E1 {values['E1_fixed_text']*100:.3f}%；逐图属性文本E2 {values['E2_image_text']*100:.3f}%。",'',
        f"主要文本比较E2-E1：{delta['E2_minus_E1']['macro_dice_delta']*100:+.3f}个百分点；图像配对bootstrap95%区间 {[round(x*100,3) for x in delta['E2_minus_E1']['paired_image_bootstrap95']]} 个百分点。",
        f"整体比较E2-E0：{delta['E2_minus_E0']['macro_dice_delta']*100:+.3f}个百分点；区间 {[round(x*100,3) for x in delta['E2_minus_E0']['paired_image_bootstrap95']]} 个百分点。",'',
        'E0与E1/E2的输出头设计不同，所以二者差异不能单独归因于文字内容；严格的逐图文本额外价值由E2-E1衡量。单seed图像区间不估计患者或训练随机性。','',
        'ISIC2016采用806/90/379去重清单，源患者独立性未验证；测试集曾用于历史开发，不称本次全新外部验证。E2仅VQA颜色/形状属性，描述重复，不能将其等同于医生报告或证明复杂指代理解。','',
        '与MosMed的对照只说明不同任务/数据上的表现差异；ISIC沿用batch16和原亮度/对比度增强，MosMed是batch64且无增强，因此两轮不是只改变数据集的严格消融。',
        '', '三组模型的验证曲线、原始指标与固定ID预测图一并保留。']
    (ROOT/'analysis/FINDINGS.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'analysis/independent_report_check.json').write_text(json.dumps(dict(passed=True,checks=checks,historical_metrics_reproduced=True,selected_epoch_matches_history=True),indent=2)+'\n')
    print('INDEPENDENT_REPORT_CHECK_PASSED',flush=True)

if __name__=='__main__':main()
