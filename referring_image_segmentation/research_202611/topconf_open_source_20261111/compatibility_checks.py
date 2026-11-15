from pathlib import Path
import ast,importlib.util,json,traceback,time,math,gc,hashlib
import torch,torch.nn as nn,torch.nn.functional as F,numpy as np
from itertools import combinations
R=Path(__file__).resolve().parent
OLD=R.parent/'module_design_20261111/official_code'
NEW=R/'official_code'
def loadfile(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def loadnodes(path,names):
 src=path.read_text();tree=ast.parse(src)
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 assert set(names)=={n.name for n in nodes}
 scope=dict(torch=torch,nn=nn,F=F,np=np,math=math,combinations=combinations)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
 return scope
sources={
 'EMCAD':OLD/'SLDGroup__EMCAD/lib/decoders.py',
 'CGA':NEW/'maklachur__MambaLiteUNet/models/MambaLiteUNet.py',
 'MFMS':NEW/'Inha-CVAI__MADGNet/IS2D_models/mfmsnet.py',
 'GAB':NEW/'JCruan519__EGE-UNet/models/egeunet.py',
 'LoMix':NEW/'SLDGroup__LoMix/trainer.py',
}
report={'purpose':'Random-tensor interface/finite-gradient checks only; no accuracy, convergence or fair performance benchmark. Author module source unchanged. Imports isolated for MFMS/LoMix using original AST definitions. Batch 2 is not a full training batch.',
 'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'source_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sources.items()},'cases':[]}
mods={}
for name in ['EMCAD','CGA','GAB']:
 try:mods[name]=loadfile(sources[name],'audit_'+name)
 except Exception as e:report['cases'].append(dict(module=name,status='import_failed',error=traceback.format_exc()))
try:mods['MFMS']=loadnodes(sources['MFMS'],['get_freq_indices','MultiFrequencyChannelAttention','MFMSAttentionBlock'])
except Exception:report['cases'].append(dict(module='MFMS',status='import_failed',error=traceback.format_exc()))
def run(name,h,mode):
 row=dict(module=name,height=h,batch=2,precision=mode)
 torch.manual_seed(20261111);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 try:
  if name=='LGAG':m=mods['EMCAD'].LGAG(64,64,32,kernel_size=3,groups=32)
  elif name=='CGA':m=mods['CGA'].CrossGatedAttentionGate(64,64,32)
  elif name=='MFMS':m=mods['MFMS']['MFMSAttentionBlock'](64)
  elif name=='GAB':m=mods['GAB'].group_aggregation_bridge(64,64)
  elif name=='EMCAD_full':m=mods['EMCAD'].EMCAD(channels=[768,384,192,96])
  m=m.cuda().train();row['parameters']=sum(p.numel() for p in m.parameters())
  if name=='EMCAD_full':
   ins=[torch.randn(2,c,z,z,device='cuda',requires_grad=True) for c,z in [(768,8),(384,16),(192,32),(96,64)]]
  elif name=='GAB':
   ins=[torch.randn(2,64,h//2,h//2,device='cuda',requires_grad=True),torch.randn(2,64,h,h,device='cuda',requires_grad=True),torch.sigmoid(torch.randn(2,1,h,h,device='cuda')).requires_grad_()]
  else:ins=[torch.randn(2,64,h,h,device='cuda',requires_grad=True) for _ in range(2 if name in ['LGAG','CGA'] else 1)]
  start=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mode=='bf16'):
   out=m(ins[0],ins[1:]) if name=='EMCAD_full' else m(*ins)
   outs=list(out) if isinstance(out,(tuple,list)) else [out]
   loss=sum(x.float().square().mean() for x in outs)
  loss.backward();torch.cuda.synchronize()
  row.update(status='pass',output_shapes=[list(x.shape) for x in outs],output_finite=all(bool(torch.isfinite(x).all()) for x in outs),input_grad_finite=all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in ins),parameter_grad_finite=all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None),parameters_without_gradient=[n for n,p in m.named_parameters() if p.requires_grad and p.grad is None],peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,single_unwarmed_check_seconds=time.perf_counter()-start)
  if not all(row[k] for k in ['output_finite','input_grad_finite','parameter_grad_finite']):row['status']='nonfinite_or_missing_gradient'
 except Exception:row.update(status='failed',error=traceback.format_exc())
 report['cases'].append(row);print(name,h,mode,row['status'],row.get('peak_allocated_MiB'),flush=True)
 (R/'compatibility_results.json').write_text(json.dumps(report,indent=2))
 gc.collect();torch.cuda.empty_cache()
for name in ['LGAG','CGA','MFMS','GAB']:
 for h in [16,32,64]:
  for mode in ['fp32','bf16']:run(name,h,mode)
for mode in ['fp32','bf16']:run('EMCAD_full',0,mode)
try:
 lm=loadnodes(sources['LoMix'],['WeightedFusion','CombinatorialMutationsLossModule'])
 row=dict(module='LoMix_concat_repeatability',author_code_unchanged=True)
 for op in ['concat','add']:
  torch.manual_seed(42)
  m=lm['CombinatorialMutationsLossModule'](2,2,2,operations=[op]).eval()
  ins=[torch.randn(2,2,8,8) for _ in range(2)]
  a=m(ins);b=m(ins)
  # Official no-label branch returns fused_logits list.
  def tensors(o):
   if torch.is_tensor(o):return [o]
   if isinstance(o,(list,tuple)):return sum([tensors(v) for v in o],[])
   return []
  aa,bb=tensors(a),tensors(b)
  row[op]=dict(return_type=type(a).__name__,tensor_count=len(aa),max_abs_repeat_difference=max((float((x-y).abs().max()) for x,y in zip(aa,bb)),default=None),state_dict_keys=list(m.state_dict()))
 report['audit_LoMix']=row
except Exception:report['audit_LoMix']=dict(error=traceback.format_exc())
# Demonstrate the actual layout operation used by UACA, without changing its source.
x=torch.arange(1*2*2*3).reshape(1,2,2,3)
report['audit_UACA_context_layout']={'source_expression':'f = x.view(b, h * w, -1)', 'BCHW_example':x.tolist(),'actual_rows':x.view(1,6,-1).tolist(),'spatial_pixel_rows':x.flatten(2).transpose(1,2).tolist(),'equal':bool(torch.equal(x.view(1,6,-1),x.flatten(2).transpose(1,2))),'interpretation':'The direct view does not create per-pixel channel vectors from contiguous BCHW. This is a source-layout concern, not proof that the complete published method is ineffective.'}
(R/'compatibility_results.json').write_text(json.dumps(report,indent=2))
print('DONE',len(report['cases']),report.get('audit_LoMix'),flush=True)
