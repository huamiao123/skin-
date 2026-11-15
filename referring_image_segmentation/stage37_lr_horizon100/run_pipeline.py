"""Serial Stage37 pipeline: validated restart -> both new trainings -> fixed full test evaluations."""
from pathlib import Path
import datetime,fcntl,json,os,shutil,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
RESEARCH=PROJECT.parent.parent
ARMS=['swin_control', 'uct_cca']
ACTIVE=None
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False));t.replace(p)
def current(state,**kw):
    obj=dict(stage=ROOT.name,state=state,arms=ARMS,epochs_per_arm=100,lr_schedule_horizon=100,
             pipeline_pid=os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             status_file=str(ROOT/'pipeline_status.json'),resume_command=[sys.executable,str(ROOT/'run_pipeline.py')])
    obj.update(kw);dump(ROOT/'pipeline_status.json',obj);dump(RESEARCH/'current_experiment.json',obj)
def run(args,log,**status):
    global ACTIVE
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        process=subprocess.Popen([sys.executable]+args,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        ACTIVE=process.pid
        current(status.pop('state','running'),training_pid=process.pid,**status)
        code=process.wait();ACTIVE=None
        if code:raise RuntimeError(f'Child exited {code}; inspect {log}')
def retain_best(arm):
    folder=(ROOT/arm).resolve();assert folder.is_relative_to(ROOT.resolve())
    removed=[]
    targets=[folder/'results/latest.pth']+list((folder/'smoke').glob('*.pth'))
    for path in targets:
        assert path.resolve().is_relative_to(folder)
        if path.is_file():removed.append(dict(path=str(path),bytes=path.stat().st_size));path.unlink()
    dump(folder/'CHECKPOINT_RETENTION.json',dict(best_only_after_full_test=True,removed=removed))
def main():
    f=(ROOT/'pipeline.lock').open('w');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'PIPELINE_DONE.json').exists():print('ALREADY_COMPLETE',flush=True);return
    old=RESEARCH/'current_experiment.json'
    if old.exists() and not (ROOT/'previous_experiment_status.json').exists():shutil.copy2(old,ROOT/'previous_experiment_status.json')
    if not (ROOT/'INTEGRATION_VERIFIED.json').exists():
        run([str(ROOT/'verify_integration.py')],ROOT/'integration_verification.log',state='integration_verification')
    if not (ROOT/'REFERENCE_REPLAY_VERIFIED.json').exists():
        run([str(ROOT/'verify_reference.py')],ROOT/'reference_replay.log',state='reference_validation_replay')
    for arm in ARMS:
        s=ROOT/arm/'smoke';s.mkdir(parents=True,exist_ok=True)
        if (s/'RESUME_AUDIT.json').exists():continue
        if not (s/'latest.pth').exists():
            run([str(ROOT/'train.py'),'--arm',arm,'--smoke','--stop-after-epoch','1'],s/'first.log',state='smoke_training',arm=arm)
        before=json.loads((s/'history.json').read_text())
        if len(before)==1:
            run([str(ROOT/'train.py'),'--arm',arm,'--smoke'],s/'resume.log',state='smoke_resume',arm=arm)
        after=json.loads((s/'history.json').read_text());assert after[:len(before)]==before and len(after)==2
        load=json.loads((s/'RESUME_LOAD_AUDIT.json').read_text());assert load['passed'] and load['start_epoch']==2
        import torch
        ck=torch.load(s/'latest.pth',map_location='cpu');assert ck['epoch']==2
        for opt in ck['optimizers'].values():assert {int(x['step']) for x in opt['state'].values() if 'step' in x}=={60}
        del ck
        run([str(ROOT/'evaluate.py'),'--arm',arm,'--smoke'],s/'evaluation.log',state='smoke_validation_replay',arm=arm)
        dump(s/'RESUME_AUDIT.json',dict(passed=True,actual_process_restart=True,second_epoch=2,optimizer_steps=60,
             validation_replay_verified=True,test_read=False,first_epoch_preserved=True))
    for arm in ARMS:
        if (ROOT/arm/'results/TRAINING_DONE.json').exists():continue
        run([str(ROOT/'train.py'),'--arm',arm],ROOT/arm/'formal.log',state='formal_training',arm=arm,
            status_file=str(ROOT/arm/'results/run_status.json'),completed_trainings=[a for a in ARMS if (ROOT/a/'results/TRAINING_DONE.json').exists()])
    for arm in ARMS:
        if not (ROOT/arm/'results/DONE.json').exists():
            run([str(ROOT/'evaluate.py'),'--arm',arm],ROOT/arm/'evaluation.log',state='fixed_validation_and_test',arm=arm)
        done=json.loads((ROOT/arm/'results/DONE.json').read_text());assert done['complete'] and done['test_evaluated'] and done['test_n']==867
        retain_best(arm)
    summaries={a:json.loads((ROOT/a/'results/summary.json').read_text()) for a in ARMS}
    from evaluate import refrows,compare
    import numpy as np
    partition=json.loads((ROOT/'identity_partition.json').read_text())
    references=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models']
    paired={}
    for split,key in [('val','validation'),('test','test')]:
        ids=partition['splits'][split]
        rows={a:refrows(ROOT/a/'results'/f'{a}_{key}_per_image.csv',ids) for a in ARMS}
        rows.update({r['arm']:refrows(r['val_rows' if split=='val' else 'test_rows'],ids) for r in references})
        pairs=[('swin_control','uct_cca'),('stage36_swin_control','swin_control'),('stage36_uct_cca','uct_cca'),('stage36_swin_control','stage36_uct_cca')]
        paired[key]={b+'_minus_'+a:compare(rows[a],rows[b]) for a,b in pairs}
        delta=np.array([(n1['dice']-n0['dice'])-(o1['dice']-o0['dice']) for n1,n0,o1,o0 in
          zip(rows['uct_cca'],rows['swin_control'],rows['stage36_uct_cca'],rows['stage36_swin_control'])])
        rng=np.random.default_rng(42)
        boot=np.array([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(2000)])
        paired[key]['schedule_by_module_interaction']=dict(macro_dice_delta=float(delta.mean()),image_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
          definition='(CCA100-Swin100)-(CCA300first100-Swin300first100)',note='Single seed; exploratory paired image bootstrap, not seed/patient uncertainty.')
    dump(ROOT/'FINAL_COMPARISON.json',dict(primary_comparison='Matched2x2 architecture and learning-rate-horizon comparison',paired=paired,seed=42,test_n_per_arm=867,complete=True))
    table=[]
    for r in references:
        table.append(dict(stage=36,arm=r['source_arm'],lr_horizon=300,best_epoch=r['epoch'],
          val_macro_dice=r['validation']['macro']['dice'],test_macro_dice=r['test']['macro']['dice'],test_pooled_dice=r['test']['pooled']['dice']))
    for arm,summary in summaries.items():
        table.append(dict(stage=37,arm=arm,lr_horizon=100,best_epoch=summary['selection']['epoch'],
          val_macro_dice=summary['validation'][arm]['macro']['dice'],test_macro_dice=summary['test'][arm]['macro']['dice'],test_pooled_dice=summary['test'][arm]['pooled']['dice']))
    dump(ROOT/'RESULTS_TABLE.json',table)
    report=['Stage37完成：新2组各100轮，复用Stage36旧2组；仅改学习率衰减分母300到100。最佳权重按验证集逐图宏平均Dice选择。','']
    for r in table:
        report.append(f"stage{r['stage']}/{r['arm']}, horizon={r['lr_horizon']}, best_epoch={r['best_epoch']}, val_macro_Dice={r['val_macro_dice']:.8f}, test_macro_Dice={r['test_macro_dice']:.8f}, test_pooled_Dice={r['test_pooled_dice']:.8f}")
    report.extend(['','分别报告训练曲线收益、同配方下CCA收益，以及两者交互。单seed；既有测试集已暴露，不能视作全新独立确认。'])
    (ROOT/'RESULTS_zh.txt').write_text('\n'.join(report))
    dump(ROOT/'summary.json',dict(complete=True,arms=summaries,paired=paired,test_n_per_arm=867))

    dump(ROOT/'PIPELINE_DONE.json',dict(complete=True,trained=ARMS,test_evaluated=ARMS))
    current('complete',training_pid=None,pipeline_pid=None,results_summary=str(ROOT/'summary.json'))
    latest=RESEARCH/'LATEST_RESEARCH_20261112.json'
    if latest.exists():
        record=json.loads(latest.read_text())
        if record.get('active_experiment')==str(ROOT):
            record.update(active_experiment=None,active_pipeline_pid=None,active_arms=[],latest_completed_experiment=str(ROOT))
            dump(latest,record)
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'PIPELINE_FAILURE.json',dict(traceback=traceback.format_exc(),active_child=ACTIVE))
        current('failed',training_pid=ACTIVE,error_record=str(ROOT/'PIPELINE_FAILURE.json'))
        raise

