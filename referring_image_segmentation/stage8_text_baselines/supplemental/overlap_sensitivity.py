"""Read-only postprocessing of the locked final validation expression records."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from common import NAMES,dump
from evaluation import summarize_rows

def main():
    audit=json.loads((ROOT/'data_overlap_audit.json').read_text());excluded=set(audit['validation_image_ids_to_exclude_in_supplement'])
    while not (ROOT/'analysis/validation.json').exists():
        state=json.loads((ROOT/'run_status.json').read_text())
        if state['state']=='failed':
            dump(ROOT/'supplemental_status.json',dict(state='waiting_for_failed_main_to_resume'));return
        time.sleep(20)
    assert json.loads((ROOT/'analysis/validation.json').read_text())['passed']
    result={}
    for n in NAMES:
        rows=json.loads((ROOT/f'analysis/{n}/val/per_expression.json').read_text())
        filtered=[r for r in rows if r['image_id'] not in excluded]
        result[n]=dict(primary=summarize_rows(rows),excluding_training_duplicate_images=summarize_rows(filtered),
            removed_expressions=len(rows)-len(filtered),removed_image_ids=sorted(excluded))
    dump(ROOT/'analysis/overlap_sensitivity.json',dict(passed=True,models=result,
        note='Official UNC is primary. Same locked checkpoints; sensitivity excludes known exact training/validation file duplicates, without reselection. This is not a new test set.'))
    lines=['# 训练/验证重复图敏感性分析','',
        '保留官方UNC主划分及同一已锁定checkpoint，补充剔除与训练图字节一致的验证图后的指标。不重新选模，不把该子集称为新测试集。','',
        '| Model | Official mIoU | Excluding duplicates mIoU | Official overall IoU | Excluding duplicates overall IoU | Removed expressions |','|---|---:|---:|---:|---:|---:|']
    for n,r in result.items():
        a,b=r['primary'],r['excluding_training_duplicate_images']
        lines.append(f"| {n} | {a['macro']['iou']:.6f} | {b['macro']['iou']:.6f} | {a['pooled']['iou']:.6f} | {b['pooled']['iou']:.6f} | {r['removed_expressions']} |")
    (ROOT/'analysis/OVERLAP_SENSITIVITY.md').write_text('\n'.join(lines)+'\n')
    dump(ROOT/'supplemental_status.json',dict(state='complete',passed=True))

if __name__=='__main__':main()
