import sys, argparse,torch,time,json
from pathlib import Path
from model import PROJECT,SegmentationSystem,load_pretrained_encoder
from smoke_train import accumulate,atomic_json
sys.path.insert(0,str(PROJECT))
from stage45_upernet_decoder import train as ref
ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);ap.add_argument('--micro',type=int,required=True);a=ap.parse_args()
ref.setup();torch.use_deterministic_algorithms(True,warn_only=True)
data,_=ref.load_data();m=SegmentationSystem(load_pretrained_encoder(),a.arm,seed=42).cuda()
opt=torch.optim.AdamW([{'params':list(m.probe.parameters()),'lr':1e-3,'name':'decoder'},{'params':[p for p in m.encoder.parameters() if p.requires_grad],'lr':1e-4,'name':'encoder'}],weight_decay=.01)
vc=ref.cache_new(len(data.ids['val']));tc=ref.cache_new(len(data.ids['train']))
ref.fill_cache(m,data,'val',vc);order,g=data.geometry(1);ref.fill_cache(m,data,'train',tc,g)
torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
m.train();loss,norms=accumulate(m,opt,tc,order[:64],a.micro)
torch.cuda.synchronize()
report=dict(arm=a.arm,micro=a.micro,effective=64,real_data=True,train_n=len(data.ids['train']),val_n=len(data.ids['val']),
 loss=loss,norms=norms,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
 peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,seconds_one_effective_batch=time.perf_counter()-t,passed=True)
p=Path('reports')/f'capacity_real_{a.arm}_{a.micro}.json';atomic_json(p,report);print(json.dumps(report),flush=True)
