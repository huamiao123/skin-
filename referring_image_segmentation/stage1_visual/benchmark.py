"""Choose a frozen-encoder throughput batch; optimizer batch remains 64."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from models import FrozenEncoder, Probe, segmentation_loss, tensor_hash

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=['dinov3', 'vmamba'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    encoder = FrozenEncoder(args.name).cuda()
    probe = Probe(encoder.channels).cuda()
    before = tensor_hash(encoder)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=.001, weight_decay=.01)
    initial_head = probe.decoder.readout.weight.detach().clone()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        features = encoder(torch.randn(64, 3, 256, 256, device='cuda'))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = probe(features)
        loss = segmentation_loss(output, torch.rand_like(output))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        optimizer.step()
    assert not torch.equal(initial_head, probe.decoder.readout.weight)
    assert all(p.grad is None for p in encoder.parameters()) and tensor_hash(encoder) == before
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
    shape = [list(f.shape) for f in features]
    decoder_hash = tensor_hash(Probe(encoder.channels).decoder)
    params = dict(encoder=sum(p.numel() for p in encoder.parameters()),
                  trainable=sum(p.numel() for p in probe.parameters()),
                  decoder=sum(p.numel() for p in probe.decoder.parameters()))
    del probe, optimizer, features, output, loss
    gc.collect()
    torch.cuda.empty_cache()
    records = []
    for batch in [64, 128, 256, 512, 1024]:
        try:
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(batch, 3, 256, 256, device='cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                warm = encoder(x)
                del warm
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(3):
                    out = encoder(x)
                    del out
                torch.cuda.synchronize()
            seconds = (time.perf_counter() - t) / 3
            row = dict(batch=batch, seconds=seconds, images_per_second=batch/seconds,
                       peak_gib=torch.cuda.max_memory_allocated()/1024**3, passed=True)
            records.append(row)
            print(args.name, row, flush=True)
            del x
            torch.cuda.empty_cache()
            # Reserve >=12 GiB for epoch/validation feature caches, data and head updates.
            if row['peak_gib'] > 32:
                break
        except torch.cuda.OutOfMemoryError:
            records.append(dict(batch=batch, passed=False, reason='OOM'))
            if 'x' in locals(): del x
            gc.collect(); torch.cuda.empty_cache()
            break
    candidates = [r for r in records if r['passed'] and r['peak_gib'] < 32]
    assert candidates
    best = max(r['images_per_second'] for r in candidates)
    # Within 5% of maximum throughput, use the smaller batch for memory headroom.
    selected = min(r['batch'] for r in candidates if r['images_per_second'] >= .95*best)
    report = dict(model=args.name, encoder_batch=selected, optimizer_batch=64, records=records,
                  pretrained=encoder.audit, frozen_hash=before, decoder_initial_hash=decoder_hash,
                  feature_shapes=shape, parameters=params, frozen_and_backward_checks_passed=True,
                  device=torch.cuda.get_device_name(), total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    (ROOT / f'benchmark_{args.name}.json').write_text(json.dumps(report, indent=2)+'\n')
    print('SELECTED', selected, flush=True)


if __name__ == '__main__':
    main()
