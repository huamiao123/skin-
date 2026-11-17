from pathlib import Path
import subprocess,sys,json,datetime,fcntl,traceback,hashlib,os
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports'
def dump(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def run(args,log,phase):
 with log.open('a') as f:
  p=subprocess.Popen([sys.executable]+args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  dump(REPORT/'STAGEB_HEADLR_STATUS.json',dict(phase=phase,arm='headlr3e4',pipeline_pid=os.getpid(),child_pid=p.pid,
   at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  if p.wait()!=0:raise RuntimeError('Child failed: '+str(log))
def main():
 lock=(REPORT/'STAGEB_HEADLR.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((REPORT/'STAGEB_HEADLR_CPU_TESTS.json').read_text())['passed']
 assert json.loads((REPORT/'STAGEB_DECISION.json').read_text())['passed']==False
 assert json.loads((REPORT/'STAGEB_SCALE_DECISION.json').read_text())['passed']==False
 proto=json.loads((REPORT/'STAGEB_HEADLR_PROTOCOL_20261117.json').read_text())
 assert hashlib.sha256((ROOT/'src/train_stageB_headlr.py').read_bytes()).hexdigest()==proto['source_sha256']
 assert hashlib.sha256(Path(proto['reference_checkpoint']).read_bytes()).hexdigest()==proto['reference_sha256']
 out=ROOT/'results/stageB_headlr_20261117/headlr3e4'
 if not out.exists():
  run(['src/train_stageB_headlr.py','--arm','headlr3e4','--micro-batch','64','--stop-after-epoch','1'],
   REPORT/'stageB_headlr_first.log','first_epoch_and_preflight')
 else:
  assert (out/'latest.pth').exists() and (out/'history.json').exists(),'Existing incomplete attempt needs manual audit'
 before=json.loads((out/'history.json').read_text());assert len(before)==1
 run(['src/train_stageB_headlr.py','--arm','headlr3e4','--micro-batch','64','--resume'],
  REPORT/'stageB_headlr_resume.log','formal_training_after_restart')
 after=json.loads((out/'history.json').read_text());assert len(after)==100 and before[0]==after[0]
 assert json.loads((out/'resume_load_audit.json').read_text())['passed']
 (ROOT/'evaluations/stageB_headlr_20261117').mkdir(exist_ok=True)
 run(['src/evaluate_stageB_headlr.py'],REPORT/'stageB_headlr_evaluate.log','validation_and_fixed_test')
 s=json.loads((ROOT/'evaluations/stageB_headlr_20261117/SUMMARY.json').read_text());assert s['complete']
 control=s['models']['control']['validation'];scale=s['models']['headlr3e4']['validation']
 decision=dict(validation_only=True,test_used=False,macro_dice_gain_pp=100*(scale['macro']['dice']-control['macro']['dice']),
  boundary_F1_delta=scale['macro']['boundary_f1']-control['macro']['boundary_f1'],seed=42)
 decision['passed']=decision['macro_dice_gain_pp']>=.1 and decision['boundary_F1_delta']>=0
 dump(REPORT/'STAGEB_HEADLR_DECISION.json',decision)
 dump(REPORT/'STAGEB_HEADLR_STATUS.json',dict(phase='complete',arm='headlr3e4',test_n=867,decision=decision))
if __name__=='__main__':
 try:main()
 except Exception:
  dump(REPORT/'STAGEB_HEADLR_FAILURE.json',dict(traceback=traceback.format_exc()))
  raise
