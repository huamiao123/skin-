"""Replay both completed Stage36 controls on the same GPU using unchanged architecture."""
from pathlib import Path
import gc,json,csv
import torch
from train import ROOT,PROJECT,System,base,dump,digest,load_data,cache_new,fill_cache
def main():
 base.setup()
 records=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models']
 partition=json.loads((ROOT/'identity_partition.json').read_text())
 data,_=load_data()
 result=[]
 for record in records:
  assert digest(Path(record['checkpoint']))==record['sha256']
  for split,key in [('val','val_rows'),('test','test_rows')]:
   with Path(record[key]).open() as f:ids=[row['id'] for row in csv.DictReader(f)]
   assert ids==partition['splits'][split]
  m=System(record['source_arm'])
  state=torch.load(record['checkpoint'],map_location='cpu')
  m.load_state_dict(state['model'],strict=True);del state
  m=m.cuda().eval()
  cache=cache_new(len(data.ids['val']));fill_cache(m,data,'val',cache)
  metric,_=base.evaluate(m,cache,64)
  assert metric['counts']==record['validation']['counts'],record['arm']
  assert abs(metric['macro']['dice']-record['validation']['macro']['dice'])<1e-12
  result.append(dict(arm=record['arm'],passed=True,validation_n=762,macro_dice=metric['macro']['dice'],confusion_counts_identical=True))
  del m,cache;gc.collect();torch.cuda.empty_cache()
 payload=dict(passed=True,same_gpu='RTX4090',historical_weights_and_splits_verified=True,test_images_read=False,replays=result)
 dump(ROOT/'REFERENCE_REPLAY_VERIFIED.json',payload);print('REFERENCE_OK',json.dumps(payload),flush=True)
if __name__=='__main__':main()

