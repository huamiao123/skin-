"""Audit official single-image equivalence, batching, data and real GPU memory."""
from common import *
import copy
from torch.nn import functional as F
from models import BaselineProbe, RegionProbe, batched_hiram, region_targets, region_loss_numerators, tensor_hash, segmentation_loss
from stage4_fusion.models import EncoderPair, FusionProbe
from stage3_confirmation.data import ImageData


def tensor_error(a, b):
    return float((a.float() - b.float()).abs().max().item())


def main():
    setup()
    test = {}
    reference = FusionProbe('swin', 42)
    historical = json.loads((PROJECT/'stage4_fusion/preflight.json').read_text())
    test['historical_initial_hash'] = historical['initial_probe_hashes']['fusion_swin']
    test['local_initial_hash'] = tensor_hash(reference)
    test['historical_initial_hash_matches'] = test['historical_initial_hash'] == test['local_initial_hash']
    models = {'paired_baseline': BaselineProbe().cuda(),
              **{name: RegionProbe(name).cuda() for name in ('region_only', 'hiram')}}
    ref = reference.state_dict()
    for name, model in models.items():
        s = model.state_dict()
        assert all(torch.equal(v, s[k].cpu()) for k, v in ref.items()), name
    m0, m1 = models['region_only'], models['hiram']
    assert all(torch.equal(a, b) for a, b in zip(m0.route_dino.parameters(), m1.hiram.mug_sam.prior_generate.route_proj.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(m0.route_vmamba.parameters(), m1.hiram.mug_cnn.prior_generate.route_proj.parameters()))
    test['matched_local_baseline_initialization_and_region_heads'] = True

    # The original HiRAM module is designed for B=1 in its SAM wrapper. Compare
    # exact seeded Gumbel routing and all valid outputs and input gradients.
    official = copy.deepcopy(m1.hiram).cuda().train()
    adapted = copy.deepcopy(m1.hiram).cuda().train()
    a = torch.randn(1, 64, 16, 16, device='cuda', requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    aa, bb = a.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
    torch.cuda.manual_seed_all(2026)
    x = official(a, b)
    torch.cuda.manual_seed_all(2026)
    y = batched_hiram(adapted, aa, bb, True)
    test['official_single_forward_max_abs'] = max(tensor_error(x[i], y[i]) for i in (0, 1))
    test['official_single_region_max_abs'] = max(tensor_error(x[2][i], y[2][i]) for i in (0, 1))
    (x[0].square().mean() + x[1].square().mean() + sum(v.square().mean() for v in x[2])).backward()
    (y[0].square().mean() + y[1].square().mean() + sum(v.square().mean() for v in y[2])).backward()
    test['official_single_input_gradient_max_abs'] = max(tensor_error(a.grad, aa.grad), tensor_error(b.grad, bb.grad))
    grad_diffs = []
    for (n, p), (n2, q) in zip(official.named_parameters(), adapted.named_parameters()):
        assert n == n2
        if p.grad is None or q.grad is None:
            assert p.grad is None and q.grad is None, n
        else:
            grad_diffs.append(tensor_error(p.grad, q.grad))
    test['official_single_parameter_gradient_max_abs'] = max(grad_diffs)
    assert test['official_single_forward_max_abs'] < 2e-5 and test['official_single_region_max_abs'] < 2e-6
    assert test['official_single_input_gradient_max_abs'] < 2e-5 and test['official_single_parameter_gradient_max_abs'] < 2e-5
    del official, adapted, a, b, aa, bb, x, y
    torch.cuda.empty_cache()

    # Deterministic eval: a sample has identical result when a different image
    # is added to its batch, up to floating-point kernel differences.
    module = m1.hiram.eval()
    a = torch.randn(2, 64, 16, 16, device='cuda')
    b = torch.randn_like(a)
    with torch.no_grad():
        batched = batched_hiram(module, a, b, False)
        singles = [batched_hiram(module, a[i:i+1], b[i:i+1], False) for i in range(2)]
        repeat = batched_hiram(module, a, b, False)
    test['independent_batched_vs_single_max_abs'] = max(tensor_error(batched[j], torch.cat([s[j] for s in singles])) for j in (0,1))
    test['repeat_eval_max_abs'] = max(tensor_error(batched[j], repeat[j]) for j in (0,1))
    assert test['independent_batched_vs_single_max_abs'] < 2e-5 and test['repeat_eval_max_abs'] == 0
    del a, b, batched, singles, repeat

    data = ImageData(42)
    assert data.audit == json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    test['data_counts'] = {k: len(v) for k,v in data.ids.items()}
    assert test['data_counts'] == dict(train=1886, val=808)
    test['mask_values'] = {s: torch.unique(data.masks[s]).cpu().tolist() for s in ('train','val')}
    test['region_morphology_uses_binary_threshold'] = 0.5
    test['empty_boundary_targets'] = {}
    for split in ('train','val'):
        empty = 0
        for start in range(0, len(data.ids[split]), 64):
            _, mask = data.batch(split, torch.arange(start, min(start+64,len(data.ids[split])), device='cuda'))
            boundary, interior = region_targets(mask)
            empty += int((boundary.flatten(1).sum(1) == 0).sum())
        test['empty_boundary_targets'][split] = empty

    encoder = EncoderPair().cuda().eval()
    hashes = {n: tensor_hash(getattr(encoder,n)) for n in ('dinov3','vmamba')}
    for n,h in hashes.items():
        assert h == json.loads((PROJECT/f'stage1_visual/benchmark_{n}.json').read_text())['frozen_hash']
    indices = torch.arange(16,device='cuda')
    imgs, target = data.batch('train',indices)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        features = encoder(imgs)
    test['encoder_feature_shapes_batch16'] = [list(x.shape) for x in features]
    test['encoder_peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
    target_regions = region_targets(target)
    denominators = [x.sum() + 1e-8 for x in target_regions]
    params = {}
    for name, model in models.items():
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits, regions = model(features, return_regions=True)
        assert list(logits.shape) == [16,1,256,256]
        loss = segmentation_loss(logits,target)
        if name != 'paired_baseline':
            n0,n1 = region_loss_numerators(regions, target_regions)
            loss = loss + n0/denominators[0] + n1/denominators[1]
        assert torch.isfinite(loss)
        loss.backward()
        failed = [n for n,p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        assert not failed, (name,failed)
        params[name] = dict(total=sum(p.numel() for p in model.parameters()),
                            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                            frozen_unused=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                            sample_loss=float(loss.item()))
        model.zero_grad(set_to_none=True)
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    assert all(tensor_hash(getattr(encoder,n)) == h for n,h in hashes.items())
    test['encoder_hashes'] = hashes
    test['encoder_frozen_including_buffers'] = True
    test['models'] = params
    test['gpu'] = torch.cuda.get_device_name()
    test['gpu_total_gib'] = torch.cuda.get_device_properties(0).total_memory/1024**3
    test['torch'] = torch.__version__
    import selective_scan_cuda
    test['selective_scan_cuda_path'] = selective_scan_cuda.__file__
    test['selective_scan_cuda_sha256'] = sha(selective_scan_cuda.__file__)
    test['passed'] = True
    dump(ROOT/'preflight.json', test)
    print('PREFLIGHT PASSED',json.dumps(test,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
