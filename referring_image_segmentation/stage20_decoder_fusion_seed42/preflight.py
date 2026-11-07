"""Run data, pinned-weight, exact baseline, AXIS and real CUDA checks."""
from common import *
from torch.nn import functional as F
from models import DecoderProbe, FrozenEncoder, UPSTREAM, UPSTREAM_SHA256, tensor_hash, segmentation_loss
from stage3_confirmation.models import Probe
from stage3_confirmation.data import ImageData


def main():
    setup()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    checks = dict(seed=42, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, bf16_supported=torch.cuda.is_bf16_supported(),
                  precision='BF16 autocast with FP32 loss for paired models',
                  axis_source_sha256=sha(UPSTREAM))
    assert checks['axis_source_sha256'] == UPSTREAM_SHA256
    baseline = Probe('swin', 42, channels=CHANNELS).cuda().eval()
    models = {n: DecoderProbe(n).cuda() for n in ('paired_baseline', 'gate_only', 'axis_bridge')}
    for name, model in models.items():
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in baseline.state_dict().items()), name
    gate = models['gate_only'].bridge.context_gate.state_dict()
    full_gate = models['axis_bridge'].bridge.context_gate.state_dict()
    assert all(torch.equal(gate[k], full_gate[k]) for k in gate)
    sample_features = [torch.randn(2, c, s, s, device='cuda') for c, s in zip(CHANNELS, (64,32,16,8))]
    with torch.no_grad():
        expected = baseline(sample_features)
        actual = models['paired_baseline'].eval()(sample_features)
    checks['baseline_forward_max_abs_error'] = (expected-actual).abs().max().item()
    assert checks['baseline_forward_max_abs_error'] == 0
    checks['reference_initial_hash'] = tensor_hash(baseline)
    historic = json.loads((STAGE14/'config.json').read_text())
    checks['stage14_initial_hash'] = historic['probe_initial_hash']
    checks['initialization_matches_stage14'] = checks['reference_initial_hash'] == historic['probe_initial_hash']
    stage17 = json.loads((PROJECT/'stage17_mambavision_decoder_hiram_seed42/preflight.json').read_text())
    checks['initialization_matches_stage17'] = checks['reference_initial_hash'] == stage17['local_initial_hash']
    # The previous environment already differed from stage14. Record any
    # further change and use the exactly matched local baseline for inference.

    # Compare the wrapper with the unmodified upstream module directly in its
    # intended NHWC representation, including backward through all inputs.
    bridge = models['axis_bridge'].bridge.train()
    a = torch.randn(2, 64, 32, 32, device='cuda', requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    with torch.no_grad():
        actual = bridge(a.permute(0,2,3,1), b.permute(0,2,3,1)).permute(0,3,1,2)
        g = bridge.context_gate
        att = torch.sigmoid(g.W_psi(g.relu(g.W_g(a)+g.W_x(b))))
        y = a + b*att
        ca = bridge.axis_refine
        y_h = y.mean(3,keepdim=True)
        y_w = y.mean(2,keepdim=True).permute(0,1,3,2)
        weights = ca.mlp(torch.cat((y_h,y_w),dim=2))
        w_h,w_w = torch.split(weights,[32,32],dim=2)
        expected = y * w_h.sigmoid() * w_w.permute(0,1,3,2).sigmoid()
    checks['axis_explicit_formula_max_abs_error'] = (expected-actual).abs().max().item()
    assert checks['axis_explicit_formula_max_abs_error'] < 1e-6
    del a,b,actual,expected,sample_features,baseline

    data = ImageData(42)
    assert data.audit == json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    checks['data_counts'] = {k:len(v) for k,v in data.ids.items()}
    assert checks['data_counts'] == dict(train=1886, val=808)
    checks['historical_manifest_exact_match'] = True
    encoder = FrozenEncoder().cuda().eval()
    original_hash = tensor_hash(encoder)
    assert original_hash == historic['encoder_initial_hash']
    checks['encoder_sha256'] = original_hash
    checks['pretrained'] = encoder.audit
    checks['encoder_parameters'] = sum(p.numel() for p in encoder.parameters())
    images, target = data.batch('train',torch.arange(4, device='cuda'))
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        features = encoder(images)
    checks['encoder_features'] = [list(f.shape) for f in features]
    assert all(torch.isfinite(f).all() for f in features)
    checks['encoder_peak_gib_including_resident_data'] = torch.cuda.max_memory_allocated()/1024**3
    checks['models'] = {}
    for name, model in models.items():
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(features)
        assert list(logits.shape) == [4,1,256,256]
        loss = segmentation_loss(logits,target)
        assert torch.isfinite(loss)
        loss.backward()
        failed = [n for n,p in model.named_parameters() if p.requires_grad and
                  (p.grad is None or not torch.isfinite(p.grad).all())]
        assert not failed, (name,failed)
        checks['models'][name] = dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                      sample_loss=float(loss), output_shape=list(logits.shape))
        model.zero_grad(set_to_none=True)
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    assert not any(m.training for m in encoder.modules())
    assert tensor_hash(encoder) == original_hash
    checks['encoder_unchanged_including_buffers'] = True
    checks['passed'] = True
    dump(ROOT/'preflight.json',checks)
    print('PREFLIGHT PASSED',json.dumps(checks,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
