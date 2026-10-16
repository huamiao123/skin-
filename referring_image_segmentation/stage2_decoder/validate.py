"""Preflight for expensive training: metric equivalence, initialization, gradients."""
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import time
import torch
import numpy as np

from models import FrozenEncoder, LightProbe, Probe, segmentation_loss, tensor_hash, DECODER_SPEC
from evaluation import aggregate_counts
from stage1_visual.analyze import metrics

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=['dinov3','vmamba'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Hand-computed large/small target example and empty-mask conventions.
    r = aggregate_counts([90,5],[10,5],[10,5],[0,0])
    assert np.isclose(r['macro']['dice'], .7)
    assert np.isclose(r['pooled']['dice'], 190/220)
    assert r['prec_at']['0.5'] == .5
    e = aggregate_counts([0,0,0],[0,0,1],[0,1,0],[4,3,3])
    assert e['macro']['dice'] == 1/3 and e['macro']['precision'] == 1/3
    exact = aggregate_counts([1],[1],[0],[2])
    assert exact['prec_at']['0.5'] == 1. and exact['threshold_jaccard_065'] == 0.
    old_rows = json.loads((ROOT.parent/'stage1_visual/analysis/per_image.json').read_text())
    old = json.loads((ROOT.parent/'stage1_visual/analysis/summary.json').read_text())
    for name in ('dinov3','vmamba','mean_probability'):
        rows = [r['models'][name] for r in old_rows]
        current = aggregate_counts(*[[r[k] for r in rows] for k in ('tp','fp','fn','tn')])
        for mode in ('macro','pooled'):
            for key in ('dice','iou'):
                assert abs(current[mode][key]-old['models'][name][mode][key]) < 1e-12
    torch.manual_seed(42)
    encoder = FrozenEncoder(args.name).cuda()
    before = tensor_hash(encoder)
    old_benchmark = json.loads((ROOT.parent/f'stage1_visual/benchmark_{args.name}.json').read_text())
    assert before == old_benchmark['frozen_hash']
    encoder.train()
    assert not encoder.training
    torch.manual_seed(42)
    light = LightProbe(encoder.channels)
    torch.manual_seed(42)
    probe = Probe(encoder.channels)
    assert tensor_hash(light.adapter) == tensor_hash(probe.adapter)
    assert tensor_hash(light.decoder.readout) == tensor_hash(probe.decoder.readout)
    decoder_hash = tensor_hash(probe.decoder)
    adapter_hash = tensor_hash(probe.adapter)
    assert all(b.blocks[1].shift_size == (4,4) for b in probe.decoder.blocks[:3])
    assert probe.decoder.blocks[3].blocks[1].shift_size == (0,0)
    light = light.cuda().eval()
    probe = probe.cuda()
    params = dict(encoder=sum(p.numel() for p in encoder.parameters()),
                  adapter=sum(p.numel() for p in probe.adapter.parameters()),
                  decoder=sum(p.numel() for p in probe.decoder.parameters()),
                  trainable=sum(p.numel() for p in probe.parameters()))
    x = torch.randn(64,3,256,256,device='cuda')
    target = (torch.rand(64,1,256,256,device='cuda')>.5).float()
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        features = encoder(x)
    optimizer = torch.optim.AdamW(probe.parameters(),lr=1e-3,weight_decay=.01)
    torch.cuda.reset_peak_memory_stats()
    initial_head = probe.decoder.readout.weight.detach().clone()
    started = time.perf_counter()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits = probe(features)
        assert logits.shape == target.shape
        loss = segmentation_loss(logits,target)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
        torch.nn.utils.clip_grad_norm_(probe.parameters(),1.)
        optimizer.step()
    torch.cuda.synchronize()
    seconds = (time.perf_counter()-started)/2
    peak = torch.cuda.max_memory_allocated()/1024**3
    assert not torch.equal(initial_head,probe.decoder.readout.weight)
    assert tensor_hash(encoder) == before and all(p.grad is None for p in encoder.parameters())
    del optimizer, logits, loss, features, target
    gc.collect(); torch.cuda.empty_cache()
    timings = {}
    # Same process/device/software for both complete encoder+probe pipelines.
    for name, network in [('light',light),('swin',probe)]:
        network.eval()
        timings[name] = {}
        for batch in (1,64):
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                for _ in range(3):
                    result = network(encoder(x[:batch]))
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(10):
                    result = network(encoder(x[:batch]))
                torch.cuda.synchronize()
            elapsed = (time.perf_counter()-start)/10
            timings[name][str(batch)] = dict(seconds_per_batch=elapsed,images_per_second=batch/elapsed,
                                            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    from timm.models.swin_transformer import SwinTransformerBlock
    sources = [ROOT/'models.py', ROOT/'train.py', ROOT/'launch.py', ROOT/'validate.py',
               ROOT/'PROTOCOL.md', ROOT.parent/'evaluation.py',ROOT.parent/'evaluate_runs.py',
               ROOT.parent/'stage1_visual/models.py',ROOT.parent/'stage1_visual/data.py',
               Path(inspect.getfile(SwinTransformerBlock))]
    report = dict(passed=True,model=args.name,encoder_batch=64,decoder_initial_hash=decoder_hash,
        adapter_initial_hash=adapter_hash,encoder_initial_hash=before,parameters=params,
        decoder_spec=DECODER_SPEC,seconds_per_training_update=seconds,training_peak_gib=peak,
        frozen_encoder_verified=True,all_trainable_gradients_finite=True,adapter_readout_match_light=True,
        historical_metrics_match=True,inference=timings,
        inference_note='Synthetic normalized image; encoder+adapter+decoder; no data loading; BF16; 3 warmups, 10 repeats; both models resident',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name())
    (ROOT/f'preflight_{args.name}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
