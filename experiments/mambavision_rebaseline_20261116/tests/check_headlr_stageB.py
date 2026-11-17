from pathlib import Path
import ast,copy,json,torch
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'src/train_stageB_headlr.py').read_text()
node=next(n for n in ast.parse(source).body if getattr(n,'name',None)=='make_optimizer')
scope={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'make_optimizer','exec'),scope)
class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__();self.probe=torch.nn.Linear(3,2);self.encoder=torch.nn.Linear(3,3)
torch.manual_seed(42);baseline=Tiny();candidate=copy.deepcopy(baseline)
old=torch.optim.AdamW([
 dict(params=list(baseline.probe.parameters()),lr=.001,initial_lr=.001,name='decoder'),
 dict(params=list(baseline.encoder.parameters()),lr=.0001,initial_lr=.0001,name='encoder')],
 weight_decay=.01,betas=(.9,.999),eps=1e-8)
new=scope['make_optimizer'](candidate)
assert [g['lr'] for g in new.param_groups]==[.0003,.0001]
for g,h in zip(old.param_groups,new.param_groups):
 for key in set(g)-{'params','lr','initial_lr'}:assert g[key]==h[key],key
initial={k:p.detach().clone() for k,p in baseline.named_parameters()}
for (name,p),(name2,q) in zip(baseline.named_parameters(),candidate.named_parameters()):
 assert name==name2
 grad=torch.arange(p.numel()).reshape(p.shape).float()/max(p.numel(),1)+.1
 p.grad=grad.clone();q.grad=grad.clone()
for opt in [old,new]:
 for group in opt.param_groups:torch.nn.utils.clip_grad_norm_(group['params'],1.)
 opt.step()
for (name,p),(name2,q) in zip(baseline.named_parameters(),candidate.named_parameters()):
 if name.startswith('encoder'):assert torch.equal(p,q),name
 else:torch.testing.assert_close(q-initial[name],.3*(p-initial[name]),atol=4e-8,rtol=1e-3)
assert 'color_augmentation import' not in source and 'scale_augmentation import' not in source
result=dict(passed=True,encoder_updates_bitwise_equal=True,decoder_first_update_ratio=.3,
 other_optimizer_options_exact=True,no_new_augmentation=True)
(ROOT/'reports/STAGEB_HEADLR_CPU_TESTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
