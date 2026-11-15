"""Condense diagnostics without choosing a method or modifying selected scores."""
from pathlib import Path
import argparse
import csv
import datetime
import json
import statistics
import time

ROOT=Path(__file__).resolve().parent
STAGES=('stage22_partial_finetune','stage23_encoder_lr3e5','stage24_seed43_confirmation')


def write(path, obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def summarize(stage):
    root=ROOT/'results'/stage
    s=json.loads((root/'report.json').read_text())
    m=s['models']
    rows={name:{r['id']:r for r in csv.DictReader((root/(name+'_per_image.csv')).open())}
          for name in ('frozen','finetune34')}
    changes=[]
    for sid,base in rows['frozen'].items():
        fine=rows['finetune34'][sid]
        if int(base['area_quartile'])==4:
            changes.append(dict(id=sid,delta_dice=float(fine['dice'])-float(base['dice']),
                                recall_frozen=float(base['recall']),recall_finetune=float(fine['recall'])))
    d=sorted(r['delta_dice'] for r in changes)
    output=dict(stage=stage,seed=43 if stage==STAGES[2] else 42,
        encoder_lr=3e-5 if stage==STAGES[1] else 1e-4,
        macro_dice={name:v['groups']['all']['macro_dice'] for name,v in m.items()},
        largest_quartile={name:v['groups']['area_q4'] for name,v in m.items()},
        largest_quartile_checkpoint_deltas={criterion:
            m['finetune34']['existing_checkpoint_criteria'][criterion]['area_q4']['macro_dice']-
            m['frozen']['existing_checkpoint_criteria'][criterion]['area_q4']['macro_dice']
            for criterion in ('best_dice','best_loss','last')},
        largest_quartile_tail=dict(n=len(d),mean_delta=statistics.mean(d),
            median_delta=statistics.median(d),improved=sum(x>0 for x in d),worsened=sum(x<0 for x in d),
            losses_over_5pp=sum(x<-.05 for x in d),losses_over_20pp=sum(x<-.2 for x in d),
            worst5=sorted(changes,key=lambda r:r['delta_dice'])[:5],
            sensitivity_only_mean_excluding_worst5=statistics.mean(d[5:]),
            note='All cases remain in official metrics; exclusion is a post hoc concentration diagnostic only.'),
        best_grid_threshold={name:dict(threshold=(r:=max(v['threshold_curve'],key=lambda z:z['groups']['all']['macro_dice']))['threshold'],
            macro_dice=r['groups']['all']['macro_dice'],
            gain_over_point5=r['groups']['all']['macro_dice']-v['groups']['all']['macro_dice']) for name,v in m.items()})
    return output


def main(wait):
    deadline=time.time()+7200
    if wait:
        while not (ROOT/'results'/STAGES[2]/'DONE.json').exists():
            queue=json.loads((ROOT/'queue_status.json').read_text())
            if queue['state']=='failed':
                raise RuntimeError('Upstream diagnosis queue failed: '+queue.get('error',''))
            if time.time()>deadline:
                raise TimeoutError('Diagnostic report did not finish within two hours')
            time.sleep(15)
    stages=[summarize(stage) for stage in STAGES if (ROOT/'results'/stage/'DONE.json').exists()]
    result=dict(state='complete' if len(stages)==3 else 'partial',
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stages=stages,
        decision='No module selected automatically. Distinguish a reproducible error from its unproven cause.',
        limitations='Development-only, same images across seeds; no threshold, checkpoint, or sample exclusion changes to official results.',
        test_evaluated=False)
    original=[s for s in stages if s['encoder_lr']==1e-4]
    if len(original)==2:
        result['seed_replication']=dict(
            overall_dice_deltas=[s['macro_dice']['finetune34']-s['macro_dice']['frozen'] for s in original],
            largest_quartile_deltas=[s['largest_quartile_tail']['mean_delta'] for s in original],
            largest_quartile_severe_regression_counts=[s['largest_quartile_tail']['losses_over_20pp'] for s in original],
            largest_quartile_low_recall_counts=[s['largest_quartile']['finetune34']['severe_undercoverage_n'] for s in original],
            worst5_id_overlap=sorted(set(r['id'] for r in original[0]['largest_quartile_tail']['worst5']) &
                                    set(r['id'] for r in original[1]['largest_quartile_tail']['worst5'])))
    write(ROOT/'evidence_summary.json',result)
    print(json.dumps(dict(state=result['state'],stages=len(stages),seed_replication=result.get('seed_replication'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait',action='store_true')
    main(parser.parse_args().wait)
