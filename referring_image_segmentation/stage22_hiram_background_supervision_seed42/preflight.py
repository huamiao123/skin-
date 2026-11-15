"""Check data, class targets, the proposed loss and matched model initialization."""
from common import *
from models import (NAMES, DecoderProbe, FrozenEncoder, region_targets, historical_targets,
                    auxiliary_numerators, auxiliary_loss, segmentation_loss, tensor_hash)
from stage3_confirmation.data import ImageData
from stage3_confirmation.models import Probe


def gradient_case(boundary, interior, background, mode):
    logits=torch.zeros(1,3,1,1,requires_grad=True)
    pred=logits.log_softmax(1)
    targets=tuple(torch.tensor([[[[x]]]]) for x in (boundary,interior,background))
    nums=auxiliary_numerators([pred],targets)
    denoms=tuple(t.sum()+1e-8 for t in targets)
    loss=auxiliary_loss(mode,nums,denoms)
    loss.backward()
    return dict(loss=float(loss.detach()),gradient=logits.grad.flatten().tolist())


def main():
    setup()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    old_bg=gradient_case(0,0,1,'original')
    new_bg=gradient_case(0,0,1,'background_supervised')
    old_boundary=gradient_case(1,0,0,'original')
    assert old_bg['loss']==0 and max(abs(v) for v in old_bg['gradient'])==0
    assert new_bg['gradient'][0] < 0 and old_boundary['gradient'][0] > 0
    data=ImageData(42)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    assert {k:len(v) for k,v in data.ids.items()}=={'train':1886,'val':808}
    target_stats={}
    for split in ('train','val'):
        minima,maxima,errors=[],[],[]
        for start in range(0,len(data.ids[split]),64):
            _,mask=data.batch(split,torch.arange(start,min(start+64,len(data.ids[split])),device='cuda'))
            boundary,interior,background=region_targets(mask)
            old=historical_targets(mask)
            assert torch.equal(boundary,old[0]) and torch.equal(interior,old[1])
            stack=torch.cat((background,boundary,interior),1)
            minima.append(float(stack.min()));maxima.append(float(stack.max()))
            errors.append(float((stack.sum(1)-1).abs().max()))
        target_stats[split]=dict(min=min(minima),max=max(maxima),max_sum_error=max(errors))
        assert target_stats[split]['min']>=0 and target_stats[split]['max']<=1 and max(errors)<1e-6
    models={n:DecoderProbe('hiram').cuda() for n in NAMES}
    reference=models[NAMES[0]].state_dict()
    assert all(all(torch.equal(reference[k],v) for k,v in m.state_dict().items()) for m in models.values())
    model_hash=tensor_hash(models[NAMES[0]])
    base_hash=tensor_hash(Probe('swin',42,channels=CHANNELS))
    prior=json.loads((PROJECT/'stage17_mambavision_decoder_hiram_seed42/preflight.json').read_text())
    encoder=FrozenEncoder().cuda().eval()
    encoder_hash=tensor_hash(encoder)
    assert encoder_hash==json.loads((STAGE14/'config.json').read_text())['encoder_initial_hash']
    images,target=data.batch('train',torch.arange(4,device='cuda'))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features=encoder(images)
    assert [tuple(f.shape) for f in features]==[(4,c,s,s) for c,s in zip(CHANNELS,(64,32,16,8))]
    targets=region_targets(target);denoms=tuple(t.sum()+1e-8 for t in targets)
    rng=torch.cuda.get_rng_state();outputs={};samples={}
    for name,model in models.items():
        model.train();torch.cuda.set_rng_state(rng)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits,regions=model(features,return_regions=True)
        nums=auxiliary_numerators(regions,targets)
        loss=segmentation_loss(logits,target)+auxiliary_loss(name,nums,denoms)
        assert list(logits.shape)==[4,1,256,256] and torch.isfinite(loss)
        loss.backward()
        failed=[n for n,p in model.named_parameters() if p.requires_grad and
                (p.grad is None or not torch.isfinite(p.grad).all())]
        assert not failed,(name,failed)
        outputs[name]=logits.detach()
        samples[name]=dict(loss=float(loss),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    differences={n:float((outputs[n].float()-outputs[NAMES[0]].float()).abs().max()) for n in NAMES}
    assert max(differences.values())==0
    assert tensor_hash(encoder)==encoder_hash and all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    assert not any(m.training for m in encoder.modules())
    result=dict(passed=True,seed=42,gpu=torch.cuda.get_device_name(),gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
        cuda=torch.version.cuda,torch=torch.__version__,data_counts={'train':1886,'val':808},
        manifest_exact_match=True,target_stats=target_stats,synthetic_gradients=dict(old_bg=old_bg,new_bg=new_bg,old_boundary=old_boundary),
        same_initialization=True,model_initial_sha256=model_hash,initial_outputs_max_abs=differences,
        probe_initial_sha256=base_hash,stage17_probe_initial_sha256=prior['local_initial_hash'],
        stage17_initial_probe_matches=base_hash==prior['local_initial_hash'],
        encoder_sha256=encoder_hash,pretrained=encoder.audit,encoder_frozen_including_buffers=True,
        feature_shapes=[list(f.shape) for f in features],sample_checks=samples)
    dump(ROOT/'preflight.json',result)
    print('PREFLIGHT',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
