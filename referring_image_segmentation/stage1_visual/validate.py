"""Small independent checks for paired metrics and loss reduction."""
import json
from pathlib import Path
import numpy as np
import torch
from analyze import metrics
from models import segmentation_loss

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(4)
mask=np.zeros((32,32),dtype=bool);mask[8:24,8:24]=True
perfect=metrics(mask,mask)
assert perfect['dice']==perfect['iou']==perfect['boundary_f1']==1.
assert perfect['hd95_pixels']==0.
empty=metrics(np.zeros_like(mask),mask)
assert empty['dice']==empty['iou']==empty['recall']==empty['boundary_f1']==0.
assert np.isclose(empty['hd95_pixels'],np.hypot(32,32))
assert metrics(np.zeros_like(mask),np.zeros_like(mask))['dice']==1.
shift=metrics(np.roll(mask,1,axis=0),mask)
assert 0<shift['dice']<1 and shift['hd95_pixels']==1.
torch.manual_seed(42)
x=torch.randn(7,1,16,16);y=torch.rand_like(x)
whole=segmentation_loss(x,y)
parts=(segmentation_loss(x[:3],y[:3])*3+segmentation_loss(x[3:],y[3:])*4)/7
assert torch.allclose(whole,parts,atol=1e-6)
a=json.loads((ROOT/'benchmark_dinov3.json').read_text())
b=json.loads((ROOT/'benchmark_vmamba.json').read_text())
assert a['decoder_initial_hash']==b['decoder_initial_hash']
for n in ['dinov3','vmamba']:
    d=json.loads((ROOT/'smoke'/n/'DONE.json').read_text())
    assert d['smoke_only'] and d['epochs']==2 and d['encoder_unchanged']
report=dict(passed=True,metric_edge_cases=True,sample_weighted_loss_invariant=True,
            equal_decoder_initialization=True,both_real_data_smoke_runs_passed=True)
(ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
