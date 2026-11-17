from pathlib import Path
import subprocess,sys,json,datetime,fcntl,traceback,hashlib
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports';ARMS=['control','color']
def dump(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def run(args,log,phase,arm=None):
 with log.open('a') as f:
  p=subprocess.Popen([sys.executable]+args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  dump(REPORT/'STAGEB_STATUS.json',dict(phase=phase,arm=arm,pipeline_pid=__import__('os').getpid(),child_pid=p.pid,
   at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  if p.wait()!=0:raise RuntimeError('Child failed: '+str(log))
def main():
 lock=(REPORT/'STAGEB.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((REPORT/'STAGEB_COLOR_CPU_TESTS.json').read_text())['passed']
 assert json.loads((ROOT/'evaluations/stageA_20261117/SUMMARY.json').read_text())['complete']
 proto=json.loads((REPORT/'STAGEB_COLOR_PROTOCOL_20261117.json').read_text())
 assert hashlib.sha256((ROOT/'src/train_stageB_color.py').read_bytes()).hexdigest()==proto['source_sha256']
 assert hashlib.sha256((ROOT/'src/color_augmentation.py').read_bytes()).hexdigest()==proto['augmentation_sha256']
 for arm in ARMS:
  out=ROOT/'results/stageB_color_20261117'/arm
  assert not out.exists(),'Refuse to overwrite existing experiment'
  run(['src/train_stageB_color.py','--arm',arm,'--micro-batch','64','--stop-after-epoch','1'],
      REPORT/('stageB_'+arm+'_first.log'),'first_epoch_and_preflight',arm)
  before=json.loads((out/'history.json').read_text());assert len(before)==1
  run(['src/train_stageB_color.py','--arm',arm,'--micro-batch','64','--resume'],
      REPORT/('stageB_'+arm+'_resume.log'),'formal_training_after_restart',arm)
  after=json.loads((out/'history.json').read_text())
  assert len(after)==100 and before[0]==after[0]
  assert json.loads((out/'resume_load_audit.json').read_text())['passed']
 (ROOT/'evaluations/stageB_color_20261117').mkdir(exist_ok=True)
 run(['src/evaluate_stageB_color.py'],REPORT/'stageB_evaluate.log','validation_and_fixed_test')
 s=json.loads((ROOT/'evaluations/stageB_color_20261117/SUMMARY.json').read_text());assert s['complete']
 control=s['models']['control']['validation'];color=s['models']['color']['validation']
 decision=dict(validation_only=True,test_used=False,macro_dice_gain_pp=100*(color['macro']['dice']-control['macro']['dice']),
  boundary_F1_delta=color['macro']['boundary_f1']-control['macro']['boundary_f1'],seed=42,
  interpretation='Single seed validation screen, not significance evidence or new independent test.')
 decision['passed']=decision['macro_dice_gain_pp']>=.1 and decision['boundary_F1_delta']>=0
 dump(REPORT/'STAGEB_DECISION.json',decision)
 dump(REPORT/'STAGEB_STATUS.json',dict(phase='complete',arms=ARMS,test_n_each=867,decision=decision))
if __name__=='__main__':
 try:main()
 except Exception:
  dump(REPORT/'STAGEB_FAILURE.json',dict(traceback=traceback.format_exc()))
  raise
