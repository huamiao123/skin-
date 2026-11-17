import json,sys,runpy,torch,hashlib,datetime
from pathlib import Path
root=Path(__file__).resolve().parents[1];original=root/'src/train_mb64.py';state=root/'results/formal_mb64/B-512-noaux/config.json'
c=json.loads(state.read_text());source=original.read_text();needle="'device':torch.cuda.get_device_name(0)";assert source.count(needle)==1
assert c['device']=='NVIDIA GeForce RTX 3090' and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 4090'
assert hashlib.sha256(original.read_bytes()).hexdigest()==c['source_sha256']
record={'old_gpu':c['device'],'new_gpu':torch.cuda.get_device_name(0),'reason':'resume unchanged config and checkpoint on new GPU; actual numeric reproducibility is not guaranteed','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_source_sha256':c['source_sha256'],'physical_batch':64,'effective_batch':64}
(root/'reports/B_resume_gpu_change_20261117.json').write_text(json.dumps(record,indent=2))
source=source.replace(needle,"'device':"+repr(c['device']))
sys.argv=[str(original),'--arm','B-512-noaux','--micro-batch','64','--resume']
sys.path.insert(0,str(original.parent));scope={'__name__':'__main__','__file__':str(original)}
exec(compile(source,str(original),'exec'),scope)
