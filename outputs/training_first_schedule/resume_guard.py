import pathlib,json,time,os,signal,datetime
base=pathlib.Path('/home/featurize/work/RSI_Implementation_20261003/outputs')
target=103068
record=base/'training_first_schedule/resume_guard_result.json'
def save(state,**extra):
 data=dict(state=state,export_pid=target,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
 tmp=record.with_suffix('.tmp.json');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(record)
save('waiting_for_three_training_stages')
while True:
 audit=base/'remaining_training_seed17.json'
 if audit.exists():
  try:d=json.loads(audit.read_text())
  except (OSError,json.JSONDecodeError):time.sleep(10);continue
  if d.get('status')=='failed':save('training_failed_export_left_stopped',failure=d.get('failure'));break
  if d.get('status')=='complete':
   assert len(d['completed'])==3 and d['source_bundle_unchanged'] and d['scientific_config_unchanged']
   for name in ['RSI-1','MeanHinge-1','AbsHard-1']:
    x=json.loads(pathlib.Path('/home/featurize/rsi_runs/P2-IMA-M-v2/seed17',name,'DONE.json').read_text())
    assert x['epochs']==40 and x['seed']==17 and x['stage']=='D' and x['test_scoring_locked']
    assert x['signature']['source_bundle_hash']=='36825cbb7a90064f3db3cfd97eb16ec62b29aa1947e17e88c7ad4759024268fb'
   cmd=pathlib.Path(f'/proc/{target}/cmdline')
   if not cmd.exists():save('exporter_gone_root_restart_needed');break
   assert b'cached_pilot_runner.py' in cmd.read_bytes() and os.getpgid(target)==target
   os.killpg(target,signal.SIGCONT);save('exporter_resumed_after_all_three_DONE',signal='SIGCONT',completed_stages=['RSI-1','MeanHinge-1','AbsHard-1']);break
 time.sleep(10)
