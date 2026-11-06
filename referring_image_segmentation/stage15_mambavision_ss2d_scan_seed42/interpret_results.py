"""Post-training result interpretation; does not alter training or selection."""
from common import *

def main():
    out=ROOT/'analysis';summary=json.loads((out/'summary.json').read_text())
    comparison=summary['comparisons']['mambavision_original'];best=summary['models']['best_dice']
    reference=comparison['reference'];rows=json.loads((out/'per_image_best_dice.json').read_text())
    original=json.loads((BASELINE/'analysis/summary.json').read_text())
    interpretation=dict(post_training_only=True,no_new_experiment=True,no_reselection=True,
        macro_dice_delta_percentage_points=comparison['macro_delta']['dice']*100,
        pooled_dice_delta_percentage_points=comparison['pooled_delta']['dice']*100,
        boundary_f1_delta_percentage_points=comparison['macro_delta']['boundary_f1']*100,
        hd95_pixels_delta=comparison['macro_delta']['hd95_pixels'],
        image_bootstrap95_percentage_points=[v*100 for v in comparison['paired_image_bootstrap95']],
        pixel_errors=comparison['pixel_errors'],
        fp_delta=best['counts']['fp']-reference['counts']['fp'],fn_delta=best['counts']['fn']-reference['counts']['fn'],
        area_dice_delta_percentage_points={name:(g['mambavision_macro']['dice']-g['historical']['mambavision_original']['dice'])*100 for name,g in summary['lesion_area_groups'].items()},
        latency_ms=summary['inference']['mean_ms'],previous_baseline_latency_ms=original['inference']['mean_ms'],
        latency_caveat='Two separately measured sessions, descriptive only',
        empty_predictions=sum(r['tp']+r['fp']==0 for r in rows),
        conclusion='No clear benefit for this shared-weight, frozen-encoder, window-local four-route variant; does not evaluate full VMamba SS2D block replacement or fine-tuning')
    dump(out/'interpretation.json',interpretation)
    report=out/'REPORT.md';text=report.read_text()
    text=text.replace('四路版本优于原始冻结MambaVision，可作为后续复验候选；仍需结合边界和效率及独立验证。',
        '四路版本仅macro Dice微高0.0114个百分点、图像区间跨0，pooled Dice及平均边界指标下降；当前没有明确实际优势，不据此选为新基线。')
    marker='\n## 完成后的综合判断\n'
    if marker in text:text=text.split(marker)[0]
    area=interpretation['area_dice_delta_percentage_points'];error=comparison['pixel_errors']
    text+=marker+'\n'
    text+=f'主权重macro Dice差为{interpretation["macro_dice_delta_percentage_points"]:+.4f}个百分点，图像配对95%区间{[round(x,4) for x in interpretation["image_bootstrap95_percentage_points"]]}个百分点；单seed不能据此声称稳定改进。pooled Dice下降0.1974个百分点，Boundary F1下降0.9599个百分点，HD95增加0.1996px。\n\n'
    text+=f'相对原始模型，纠正{error["corrected"]}个错误像素，同时新增{error["introduced"]}个，净增加{-error["net_errors_reduced"]}个错误。FP增加{interpretation["fp_delta"]}，FN增加{interpretation["fn_delta"]}，不支持错误总体减少。\n\n'
    text+=f'既定面积组Dice差为：GT<10% {area["lt_0.1"]:+.3f}、GT10–30% {area["0.1_to_0.3"]:+.3f}、GT>=30% {area["ge_0.3"]:+.3f}个百分点。中等面积有改善，大面积下降；均值几乎相同掩盖了分组取舍，不能据此作面积的因果归因。\n\n'
    text+='完整单图推理14.11ms，原始基线此前12.47ms，两次独立会话仅作描述；此次未体现精度与效率的整体优势。代码、权重、数据、冻结状态和全指标复核通过，当前差异不是通过少跑epoch或改变batch获得。\n\n'
    text+='结论只针对本轮：四方向共享原单路预训练参数、保持窗口、四路均值、Encoder冻结。这不是完整VMamba SS2D模块替换，也没有用完整SS2D的预训练或微调；不能把本结果写成“VMamba/SS2D无效”。完整模块替换及匹配微调需要另立实验，本轮没有自动追加。\n'
    report.write_text(text)
    dump(out/'interpretation_provenance.json',dict(post_training_only=True,script_sha256=sha(ROOT/'interpret_results.py'),
        summary_sha256=sha(out/'summary.json'),report_sha256=sha(report),training_sources_unchanged=True))
    print('POST_TRAINING_INTERPRETATION_WRITTEN',flush=True)

if __name__=='__main__':main()
