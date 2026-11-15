from pathlib import Path
import torch,copy,json,io,traceback,gc
from integration_prototypes import SwinPixelDecoder,AuthorModuleDecoder
R=Path(__file__).resolve().parent
torch.manual_seed(42)
base=SwinPixelDecoder().cpu()
rows=[]
for variant in ['emcad64','swin_cga','swin_mfms']:
 row=dict(variant=variant,purpose='Research wiring check; feature pyramid input, not full encoder or training')
 try:
  m=AuthorModuleDecoder(variant,base).cuda().train()
  inputs=[torch.randn(2,64,h,h,device='cuda',requires_grad=True) for h in [64,32,16,8]]
  with torch.autocast('cuda',dtype=torch.bfloat16):
   y=m(inputs);loss=y.float().square().mean()
  loss.backward()
  assert list(y.shape)==[2,1,256,256] and bool(torch.isfinite(y).all())
  assert all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in inputs)
  missing=[n for n,p in m.named_parameters() if p.requires_grad and p.grad is None]
  assert not missing,missing
  assert all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None)
  m.eval()
  with torch.no_grad():a=m([x.detach() for x in inputs])
  buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0)
  restored=AuthorModuleDecoder(variant,base).cuda().eval()
  restored.load_state_dict(torch.load(buf,map_location='cuda',weights_only=True),strict=True)
  with torch.no_grad():b=restored([x.detach() for x in inputs])
  delta=float((a-b).abs().max());assert delta<1e-6,delta
  row.update(status='pass',shape=list(y.shape),parameters=sum(p.numel() for p in m.parameters()),checkpoint_roundtrip_max_abs=delta)
 except Exception:row.update(status='failed',error=traceback.format_exc())
 rows.append(row);print(row,flush=True)
 (R/'integration_checks.json').write_text(json.dumps(rows,indent=2))
 del m,inputs
 if 'restored' in globals():del restored
 gc.collect();torch.cuda.empty_cache()

