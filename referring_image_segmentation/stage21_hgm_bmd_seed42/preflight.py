"""Preflight the adapted HGM model with real data and pretrained encoder."""
from common import *
from hgm_model import MambaVisionHGMProbe, FrozenEncoder, SOURCES, tensor_hash, segmentation_loss
from stage3_confirmation.data import ImageData
from stage3_confirmation.models import Probe


def main():
    setup()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    diag = json.loads((ROOT / 'diagnostic_batch_64.json').read_text())
    assert diag['passed'] and diag['shared_initialization_matches_baseline']
    base = Probe('swin', 42, channels=CHANNELS)
    prior = json.loads((PROJECT / 'stage20_decoder_fusion_seed42/preflight.json').read_text())
    assert tensor_hash(base) == prior['reference_initial_hash']
    data = ImageData(42)
    assert data.audit == json.loads((PROJECT / 'stage1_visual/data_manifest.json').read_text())['members']
    assert {k: len(v) for k, v in data.ids.items()} == {'train': 1886, 'val': 808}
    encoder = FrozenEncoder().cuda().eval()
    old = tensor_hash(encoder)
    assert old == prior['encoder_sha256']
    images, target = data.batch('train', torch.arange(4, device='cuda'))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        features = encoder(images)
    assert [tuple(x.shape) for x in features] == [(4,c,s,s) for c,s in zip(CHANNELS,(64,32,16,8))]
    assert all(torch.isfinite(x).all() for x in features)
    model = MambaVisionHGMProbe().cuda().train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(features)
    assert list(logits.shape) == [4, 1, 256, 256] and torch.isfinite(logits).all()
    loss = segmentation_loss(logits, target)
    loss.backward()
    bad = [n for n,p in model.named_parameters() if p.requires_grad and
           (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not bad, bad
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    assert not any(m.training for m in encoder.modules()) and tensor_hash(encoder) == old
    result = dict(passed=True, seed=42, data_counts={'train':1886, 'val':808},
        historical_manifest_exact_match=True, encoder_sha256=old, encoder_unchanged_including_buffers=True,
        encoder_pretrained_audit=encoder.audit, encoder_parameters=sum(p.numel() for p in encoder.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in encoder.parameters())+sum(p.numel() for p in model.parameters()),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        hgm_revision='49ae05afc9785dbbc5a303d39e63588b068e5328', source_sha256=SOURCES,
        baseline_initial_hash=prior['reference_initial_hash'], shared_initialization_matches_baseline=True,
        features=[list(x.shape) for x in features], output_shape=list(logits.shape), sample_loss=float(loss),
        precision='BF16 autocast with FP32 HGM Mamba and loss', official_hgm_reproduction=False,
        method='partial Swin + adapted HGM BMD; not full HGM')
    dump(ROOT/'preflight.json', result)
    print('PREFLIGHT', json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
