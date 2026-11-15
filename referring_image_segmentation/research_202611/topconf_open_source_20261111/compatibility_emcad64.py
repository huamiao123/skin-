from pathlib import Path
import torch,importlib.util,json
R=Path(__file__).resolve().parent
p=R.parent/'module_design_20261111/official_code/SLDGroup__EMCAD/lib/decoders.py'
spec=importlib.util.spec_from_file_location('emcad_original',p)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
d=json.loads((R/'compatibility_results.json').read_text())
d['cases']=[x for x in d['cases'] if x['module']!='EMCAD_full_64']
for precision in ['fp32','bf16']:
 torch.manual_seed(42)
 m=mod.EMCAD(channels=[64,64,64,64]).cuda().train()
 xs=[torch.randn(2,64,h,h,device='cuda',requires_grad=True) for h in [8,16,32,64]]
 with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
  ys=m(xs[0],xs[1:]);loss=sum(y.float().square().mean() for y in ys)
 loss.backward()
 finite=all(bool(torch.isfinite(y).all()) for y in ys) and all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in xs) and all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None)
 row=dict(module='EMCAD_full_64',precision=precision,batch=2,status='pass' if finite else 'failed',parameters=sum(p.numel() for p in m.parameters()),output_shapes=[list(y.shape) for y in ys],all_output_inputgrad_paramgrad_finite=finite,parameters_without_gradient=[n for n,p in m.named_parameters() if p.requires_grad and p.grad is None])
 d['cases'].append(row);print(row)
(R/'compatibility_results.json').write_text(json.dumps(d,indent=2))
