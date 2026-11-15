import ast,json,math,traceback
from pathlib import Path
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import DropPath, trunc_normal_tf_
from timm.models import named_apply
root=Path('/home/waas/mambavision_stage22/research/topconf_mamba_followup_20261112')
r=root/'author_sources'
def defs(path,names=None,ns=None):
    ns={} if ns is None else ns
    tree=ast.parse((r/path).read_text())
    chosen=[x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and (names is None or x.name in names)]
    exec(compile(ast.Module(body=chosen,type_ignores=[]),path,'exec'),ns)
    return ns
ns=dict(torch=torch,nn=nn,F=F,math=math,partial=partial,named_apply=named_apply,trunc_normal_tf_=trunc_normal_tf_,DropPath=DropPath)
defs('xmindflow__cenet/src/networks/cenet/modules/blocks.py',{'SepConvBN','_init_weights'},ns)
defs('xmindflow__cenet/src/networks/cenet/modules/nlb.py',{'Nonlocal'},ns)
defs('xmindflow__cenet/src/networks/cenet/modules/cfam.py',ns=ns)
out={'scope':'Exact author AST definitions; isolated module synthetic forward/backward only. No segmentation training; no full-network validation. Imports unrelated to selected definitions bypassed.','results':[]}
for amp in [False,True]:
    torch.manual_seed(42)
    m=ns['CFAModule'](64,init_value=1e-6,mca_rates=[1,2,4]).cuda().train()
    x=torch.randn(64,64,32,32,device='cuda',requires_grad=True)
    target=torch.randn_like(x);torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        y=m(x);loss=(y-target).square().mean()
    loss.backward()
    finite=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in m.parameters())
    out['results'].append({'module':'CENet CFAModule','amp_bf16':amp,'shape':list(y.shape),'parameters':sum(p.numel() for p in m.parameters()),'output_finite':bool(torch.isfinite(y).all()),'gradients_finite':finite,'max_init_change':float((y-x).abs().max()),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
    del m,x,y,target,loss
    torch.cuda.empty_cache()
n=dict(torch=torch,nn=nn,F=F)
defs('csguoh__MambaIR/basicsr/archs/mambair_arch.py',{'CAB','ChannelAttention'},n)
m=n['CAB'](384).cuda();x=torch.randn(2,384,16,16,device='cuda',requires_grad=True);y=m(x);y.square().mean().backward()
out['results'].append({'module':'MambaIR CAB default','shape':list(y.shape),'parameters':sum(p.numel() for p in m.parameters()),'output_finite':bool(torch.isfinite(y).all()),'gradients_finite':all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())})
x=torch.arange(24).reshape(1,3,2,4)
out['dseb_layout_check']={'nchw_view_b_l_c_matches_correct_permute':torch.equal(x.view(1,-1,3),x.permute(0,2,3,1).reshape(1,-1,3)),'note':'Pinned DSEBlock.apply_diffattn uses view without permute; tokens are not per-pixel channel vectors under the NCHW input contract. Static/reindexing finding, not a reimplementation or paper-result refutation.'}
(root/'MODULE_PREFLIGHT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
