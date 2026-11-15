"""Verify unchanged architecture/init and the exact one-factor learning-rate intervention."""
from pathlib import Path
import ast,math,json,gc
import torch
from train import ROOT,PROJECT,System,ARMS,base,dump,digest,setup,lr_epoch,objective
from author_modules import _import
def node(path,name):
 tree=ast.parse(path.read_text())
 return ast.dump(next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name))
def main():
 setup()
 prior=PROJECT/'stage36_topconf_skip_fusion'
 for p,sha in json.loads((ROOT/'ARCHIVED_SOURCE_HASHES.json').read_text()).items():assert digest(Path(p))==sha,p
 assert digest(ROOT/'author_modules.py')==digest(prior/'author_modules.py')
 for name in ['System','setup','objective','update','cache_new','fill_cache','load_data','preflight']:
  assert node(ROOT/'train.py',name)==node(prior/'train.py',name),name
 old_lr=next(n for n in ast.parse((PROJECT/'stage28_drop_path/train.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='lr_epoch')
 class ChangeHorizon(ast.NodeTransformer):
  def visit_Constant(self,n):
   return ast.copy_location(ast.Constant(value=100),n) if n.value==300 else n
 old_lr=ChangeHorizon().visit(old_lr)
 assert ast.dump(old_lr)==node(ROOT/'train.py','lr_epoch')
 author=_import(prior/'author_modules.py','stage36_archived_author_for_init')
 features=[torch.randn(2,64,h,h) for h in [64,32,16,8]]
 records={}
 for arm in ARMS:
  new=System(arm)
  old=base.System(finetune=True,stochastic_depth=False)
  old.probe.decoder=author.AuthorModuleDecoder(arm,old.probe.decoder,seed=42);old.train()
  base.assert_tree_exact(new.state_dict(),old.state_dict())
  new.eval();old.eval()
  with torch.no_grad():
   a=new.probe.decoder(features);b=old.probe.decoder(features)
  assert torch.equal(a,b)
  opt=base.optimizer(new)
  assert {id(q) for g in opt.param_groups for q in g['params']}=={id(q) for q in new.parameters() if q.requires_grad}
  schedule=[]
  for epoch in [1,2,25,50,75,100]:
   lr_epoch(opt,epoch)
   expected=.01+.99*(1+math.cos(math.pi*(epoch-1)/100))/2
   rates={g['name']:g['lr'] for g in opt.param_groups}
   assert all(abs(g['lr']-g['initial_lr']*expected)<1e-15 for g in opt.param_groups)
   schedule.append(dict(epoch=epoch,factor=expected,rates=rates))
  records[arm]=dict(shared_initial_state_exact=True,initial_prediction_exact=True,learning_rates=schedule,
     total_parameters=sum(q.numel() for q in new.parameters()),trainable_parameters=sum(q.numel() for q in new.parameters() if q.requires_grad))
  del old,new,opt;gc.collect()
 result=dict(passed=True,only_lr_formula_constant_changed=True,archived_sources_unchanged=True,unchanged_training_functions_verified=True,
  original_author_wrapper_byte_identical=True,arms=records,test_read=False)
 dump(ROOT/'INTEGRATION_VERIFIED.json',result)
 print('INTEGRATION_OK',json.dumps(result),flush=True)
if __name__=='__main__':main()

