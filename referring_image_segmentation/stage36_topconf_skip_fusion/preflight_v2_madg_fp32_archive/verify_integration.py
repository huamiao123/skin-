"""Check exact shared tensors, matched conv initialization and official runtime block paths."""
import gc,json,torch
from train import ROOT,PROJECT,System,base,dump,digest,setup,objective
from author_modules import AuthorModuleDecoder,ARMS
def main():
 setup()
 for p,sha in json.loads((ROOT/'ARCHIVED_SOURCE_HASHES.json').read_text()).items():
  assert digest(__import__('pathlib').Path(p))==sha,p
 reference=base.System(finetune=True,stochastic_depth=False)
 features=[torch.randn(2,64,h,h) for h in [64,32,16,8]]
 reference.eval()
 with torch.no_grad():expected=reference.probe.decoder(features)
 shared=reference.state_dict()
 decoders={}
 records={}
 for arm in ARMS:
  m=System(arm)
  base.assert_tree_exact(m.encoder.state_dict(),reference.encoder.state_dict())
  base.assert_tree_exact(m.probe.adapter.state_dict(),reference.probe.adapter.state_dict())
  base.assert_tree_exact(m.probe.decoder.baseline.state_dict(),reference.probe.decoder.state_dict())
  opt=base.optimizer(m)
  assert {id(q) for g in opt.param_groups for q in g['params']}=={id(q) for q in m.parameters() if q.requires_grad}
  m.eval()
  visited={}
  hooks=[]
  for key,block in getattr(m.probe.decoder,'author_blocks',{}).items():
   def make_hook(k):
    def hook(mod,args,output):visited[k]=dict(input=list(args[0].shape),skip=list(args[1].shape),output=list(output.shape))
    return hook
   hooks.append(block.register_forward_hook(make_hook(key)))
  with torch.no_grad():out=m.probe.decoder(features)
  for h in hooks:h.remove()
  assert out.shape==(2,1,256,256) and torch.isfinite(out).all()
  if arm=='swin_control':assert torch.equal(out,expected)
  elif arm.startswith('uct_'):
   assert set(visited)=={'0','1','2'}
   assert all(b.up.mode=='nearest' for b in m.probe.decoder.author_blocks.values())
  else:assert set(visited)=={'0'}
  decoders[arm]=base.cpu_tree(m.probe.decoder.state_dict())
  records[arm]=dict(total_parameters=sum(p.numel() for p in m.parameters()),
    trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
    probe_parameters=sum(p.numel() for p in m.probe.parameters()),runtime_blocks=visited)
  del m,opt;gc.collect()
 for ctrl,full,prefix in [('uct_concat','uct_cca','author_blocks.'),('madg_concat','madg_mfms','author_blocks.0.attention_layer.')]:
  a,b=decoders[ctrl],decoders[full]
  assert all(k in b and torch.equal(v,b[k]) for k,v in a.items()),(ctrl,'shared tensors differ')
  extra=set(b)-set(a)
  assert extra and all(k.startswith(prefix) for k in extra)
  if ctrl=='uct_concat':assert all('.coatt.' in k for k in extra)
 # Single-output loss is unchanged, including final sigmoid policy (logits only).
 target=torch.randint(0,2,expected.shape).float()
 assert torch.equal(objective(expected,target),base.segmentation_loss(expected,target))
 result=dict(passed=True,shared_encoder_adapters_swin_readout_exact=True,
  baseline_prediction_exact=True,paired_surrounding_conv_initialization_exact=True,
  optimizer_owns_all_trainable=True,native_author_nodes_unmodified=True,
  retained_single_output_common_loss=True,archived_sources_unchanged=True,test_read=False,arms=records)
 dump(ROOT/'INTEGRATION_VERIFIED.json',result)
 print('INTEGRATION_OK',json.dumps(result),flush=True)
if __name__=='__main__':main()

