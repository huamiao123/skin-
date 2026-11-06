"""Strict real-data acceptance before timing or the isolated smoke training."""
from common import *
import copy
from torch.nn import functional as F
from models import (Segmentor, OfficialEncoder, tensor_hash, region_targets,
                    region_numerators, segmentation_loss)
from stage3_confirmation.data import ImageData


def error(a,b):return float((a.float()-b.float()).abs().max().item())


def main():
    setup()
    audit={}
    data=ImageData(42)
    assert data.audit == json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    audit['data_counts']={k:len(v) for k,v in data.ids.items()}
    assert audit['data_counts']=={'train':1886,'val':808}
    x,gt=data.batch('train',torch.arange(4,device='cuda'))
    targets=region_targets(gt)
    assert list(targets.shape)==[4,3,16,16]
    assert error(targets.sum(1),torch.ones_like(targets[:,0])) < 1e-6
    assert bool((targets[:,1].flatten(1).sum(1)>0).all())
    audit['soft_target_sums_to_one'] = True
    a,b=Segmentor('ordered_prior'),Segmentor('region_scan')
    assert tensor_hash(a)==tensor_hash(b)
    audit['two_variants_identical_initial_state'] = True
    historical=json.loads((PROJECT/'stage17_mambavision_decoder_hiram_seed42/preflight.json').read_text())
    audit['stage17_probe_initial_hash_matches'] = tensor_hash(a.probe)==historical['local_initial_hash']
    assert audit['stage17_probe_initial_hash_matches']
    audit['frozen_hash'] = a.encoder.frozen_state_hash()
    assert audit['frozen_hash']==b.encoder.frozen_state_hash()==json.loads((STAGE14/'config.json').read_text())['encoder_initial_hash']
    audit['valid_per_window']=a.encoder.mixer.window_valid.sum(1).tolist()
    assert audit['valid_per_window']==[196,28,28,4]
    grid=torch.arange(256).reshape(1,1,16,16)
    padded=F.pad(grid,(0,12,0,12))
    windows=padded.reshape(1,1,2,14,2,14).permute(0,2,4,3,5,1).reshape(4,196,1)
    assert torch.equal(a.encoder.mixer.window_map(windows),grid)
    audit['window_pack_unpack_exact']=True

    # Native bypass must reconstruct the exact pinned NVIDIA feature outputs.
    a=a.cuda().eval()
    official=OfficialEncoder().cuda().eval()
    a.encoder.mixer.mode='native'
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        reference=official(x)
        actual,route=a.encoder(x)
    assert route is None
    audit['official_native_feature_max_abs']=max(error(u,v) for u,v in zip(reference,actual))
    assert audit['official_native_feature_max_abs']==0
    a.encoder.mixer.mode='ordered_prior'
    del official,reference,actual
    torch.cuda.empty_cache()

    # When all valid tokens choose the same category and all priorities tie,
    # region scan must reduce exactly to ordered valid + ordered padded scans.
    ordered=copy.deepcopy(a.encoder.mixer).eval()
    grouped=copy.deepcopy(a.encoder.mixer).eval()
    ordered.mode='ordered_prior';grouped.mode='region_scan';grouped.stochastic=False
    with torch.no_grad():
        for module in (ordered,grouped):
            for p in module.router.parameters():p.zero_()
            module.region_prior.copy_(torch.arange(24,device='cuda').reshape(3,8)*.001)
        tokens=torch.randn((8,196,384),device='cuda')
        y0=ordered(tokens);y1=grouped(tokens)
        audit['one_region_original_order_max_abs']=error(y0,y1)
        assert audit['one_region_original_order_max_abs'] < 1e-6
        both=grouped(tokens)
        separated=torch.cat([grouped(tokens[i:i+4]) for i in (0,4)])
        audit['independent_image_mixer_max_abs']=error(both,separated)
        assert audit['independent_image_mixer_max_abs'] < 2e-5
    del ordered,grouped,tokens,y0,y1,both,separated
    torch.cuda.empty_cache()

    # Loss accumulated in microbatches uses full-effective-batch denominators.
    fake=torch.randn(7,3,16,16,dtype=torch.float64)
    soft=torch.rand(7,3,16,16,dtype=torch.float64);soft=soft/soft.sum(1,keepdim=True)
    masses=soft.sum((0,2,3))
    numer=-(soft*fake.log_softmax(1)).sum((0,2,3))
    terms=sum((-(soft[j:j+2]*fake[j:j+2].log_softmax(1)).sum((0,2,3))/masses).mean() for j in range(0,7,2))
    audit['accumulated_aux_loss_abs']=abs(float(terms-(numer/masses).mean()))
    assert audit['accumulated_aux_loss_abs']<1e-12

    b=b.cuda()
    parameters={}
    for name,model in [('ordered_prior',a),('region_scan',b)]:
        model.train()
        assert not any(m.training for m in model.encoder.net.modules())
        assert model.encoder.mixer.stochastic
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output,route=model(x)
        assert list(output.shape)==[4,1,256,256] and list(route.shape)==[4,3,16,16]
        loss=segmentation_loss(output,gt)+(region_numerators(route,targets)/(targets.sum((0,2,3))+1e-8)).mean()
        assert torch.isfinite(loss)
        loss.backward()
        failed=[n for n,p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        assert not failed,(name,failed)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        route_grad=sum(float(p.grad.float().abs().sum()) for p in model.encoder.mixer.router.parameters())
        prior_grad=float(model.encoder.mixer.region_prior.grad.float().abs().sum())
        assert route_grad>0 and prior_grad>0,(name,route_grad,prior_grad)
        old_router=model.encoder.mixer.router[0].weight.detach().clone()
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001,weight_decay=.01)
        opt.step()
        assert not torch.equal(old_router,model.encoder.mixer.router[0].weight)
        assert model.encoder.frozen_state_hash()==audit['frozen_hash']
        parameters[name]=dict(total=sum(p.numel() for p in model.parameters()),
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            new_encoder_trainable=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
            loss=float(loss.item()),router_grad_sum=route_grad,prior_grad_sum=prior_grad,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        model.zero_grad(set_to_none=True)
        del output,route,loss,opt,old_router
    audit['models']=parameters
    audit['encoder_original_parameters_and_buffers_frozen']=True
    audit['pretrained_encoder']=a.encoder.audit
    import selective_scan_cuda
    audit['selective_scan_cuda_sha256']=sha(selective_scan_cuda.__file__)
    assert audit['selective_scan_cuda_sha256']=='0829e6a7d729adf47a5fdf24925238ea859f6395985a0db9fee8a7a8954ba929'
    audit['gpu']=torch.cuda.get_device_name()
    audit['gpu_total_gib']=torch.cuda.get_device_properties(0).total_memory/1024**3
    audit['torch']=torch.__version__
    audit['passed']=True
    dump(ROOT/'preflight.json',audit)
    print('PREFLIGHT PASSED',json.dumps(audit),flush=True)


if __name__=='__main__':main()
