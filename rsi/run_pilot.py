"""Run the pre-registered seed17 stages; obey the T1 research gate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .datasets import IMAMultiReferenceDataset
from .export import export_checkpoint
from .models import ControlledModel
from .runtime import PROJECT, atomic_json, load_config, sha256_file
from .summarize import summarize_pilot
from .train import train_stage, prepare_q, evaluate


def read_rows(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))


def run(cfg,seed=17):
    report_path=PROJECT/'outputs'/f'pilot_progress_seed{seed}.json'
    completed=[]
    def progress(stage,**extra):
        atomic_json(report_path,dict(seed=seed,active=stage,completed=completed,test_scoring_locked=True,**extra))
    progress('A_CNN')
    cnn=train_stage(cfg,seed,'A_CNN');completed.append('A_CNN')
    progress('A_T')
    transformer=train_stage(cfg,seed,'A_T',init=cnn);completed.append('A_T')
    progress('B')
    base=train_stage(cfg,seed,'B',init=transformer);completed.append('B')
    model=ControlledModel(pretrained=False).cuda()
    model.load_state_dict(torch.load(base,map_location='cpu')['model']);model.set_stage('B');model.eval()
    val=IMAMultiReferenceDataset(cfg['manifest'],split='val',subset='M',cache_dir=cfg['cache_dir'])
    bm,_=evaluate(model,val,cfg,'B')
    t1_pass=bm['mean_gain']>0 and bm['G_plus']>0
    atomic_json(PROJECT/'outputs'/f'T1_decision_seed{seed}.json',dict(stage='T1',metrics=bm,
        ordinary_message_average_gain_positive=t1_pass,
        reference_conflict_observed=bm['J']>0 and bm['kappa_pm']>0,
        test_scoring_locked=True,decision='continue_T2' if t1_pass else 'inspect_baseline_stop_T2'))
    del model;torch.cuda.empty_cache()
    rows=export_checkpoint(base,cfg,run_id='B',include_anchor=True,seed=seed)
    if not t1_pass:
        summarize_pilot(rows,PROJECT/'outputs'/f'seed{seed}'/'T1',bootstrap_repeats=cfg['bootstrap_repeats'])
        (PROJECT/'outputs'/f'seed{seed}'/'T1'/'decision_log.md').write_text(
            f'# seed{seed} 首轮决策\n\nT0 工程及全量文件验收通过。完整固定预算 T1 的 B '
            f'平均 Dice 收益为 {bm["mean_gain"]:.6f}，G+ 为 {bm["G_plus"]:.6f}。'
            '普通消息未通过正平均收益门槛，停止 T2；先检查基线及收敛曲线。'
            '不据此判定 RSI 无效，不启动 seed29 或正式 test 评分。\n',encoding='utf-8')
        progress('stopped_at_T1',reason='ordinary_message_has_no_positive_mean_gain',metrics=bm)
        return
    q=prepare_q(base,cfg)
    progress('D0',q=q)
    d0=train_stage(cfg,seed,'D',init=base,method='d0',weight=0.,q=q,run_name='D0');completed.append('D0')
    rows=export_checkpoint(base,cfg,run_id='B',include_anchor=True,d0_checkpoint=d0,seed=seed)
    rows+=export_checkpoint(d0,cfg,run_id='D0',d0_checkpoint=d0,seed=seed)
    for method,name in [('rsi','RSI-1'),('mean_hinge','MeanHinge-1'),('abs_hard','AbsHard-1')]:
        progress(name,q=q)
        ck=train_stage(cfg,seed,'D',init=base,method=method,weight=1.,q=q,run_name=name)
        completed.append(name)
        rows+=export_checkpoint(ck,cfg,run_id=name,weight=1.,method=method,d0_checkpoint=d0,seed=seed)
    for alpha in [0.,1/3,2/3,1.]:
        run_id=f'Shrink-D0-{alpha:.6f}'
        progress(run_id,q=q)
        rows+=export_checkpoint(d0,cfg,run_id=run_id,alpha=alpha,method='fixed_shrink',d0_checkpoint=d0,seed=seed)
        completed.append(run_id)
    output=PROJECT/'outputs'/f'seed{seed}'
    table=summarize_pilot(rows,output,bootstrap_repeats=cfg['bootstrap_repeats'],bootstrap_seed=seed)
    epoch_files=[]
    for name in ['A_CNN','A_T','B','D0','RSI-1','MeanHinge-1','AbsHard-1']:
        epoch_files.append(Path(cfg['run_root'])/f'seed{seed}'/name/'epoch_diagnostics.csv')
    epoch_rows=[]
    for p in epoch_files:epoch_rows+=read_rows(p)
    fields=sorted(set().union(*(set(x) for x in epoch_rows)))
    with (output/'epoch_diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(epoch_rows)
    atomic_json(output/'decision_inputs.json',dict(seed=seed,scope='exploratory_validation_selected_results',
        q=q,T1=bm,manifest_hash=sha256_file(cfg['manifest']),test_scoring_locked=True,
        conditional_weight_grid=[.3,1.,3.],extra_grid_has_not_run=True,
        seed29_has_not_run=True,formal_seeds_have_not_run=True))
    (output/'decision_log.md').write_text(
        f'# seed{seed} 首轮决策\n\nT0 及 T1 通过，lambda=1 的四种 D 目标均已跑满固定预算；'
        'Fixed-Shrink 来自选定 D0，逐 alpha 实际重算 tail。\n\n'
        '本轮表格及配对区间来自用于选择 checkpoint 的开发 val，尚不能作独立确认。'
        '需按 pilot_main_table.csv 与工作点表检查 Dice 非劣容差 0.002、H_epsilon、G+、'
        '负收益尾部及简单对照后决定同步补权重网格或 seed29。'
        '当前尚未补网格、运行 seed29 或释放正式 test；所有失败 run 均保留。\n',encoding='utf-8')
    progress('pilot_lambda1_complete',q=q,output=str(output))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--seed',type=int,default=17)
    a=p.parse_args();run(load_config(a.config),a.seed)


if __name__=='__main__':main()
