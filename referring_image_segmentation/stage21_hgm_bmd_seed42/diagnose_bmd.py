"""HGM BMD isolated model compatibility and capacity checks; not training."""
import argparse
import json
import time
import traceback
from common import CHANNELS, dump, setup
from hgm_model import MambaVisionHGMProbe
from stage3_confirmation.models import Probe
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=2)
    args = parser.parse_args()
    setup()
    output = {'batch': args.batch, 'gpu': torch.cuda.get_device_name(0),
              'kind': 'synthetic engineering diagnostic; not training or evaluation',
              'passed': False}
    try:
        reference = Probe('swin', 42, channels=CHANNELS)
        model = MambaVisionHGMProbe()
        ref = reference.state_dict()
        model_state = model.state_dict()
        shared = [k for k in model_state if k.startswith('adapter.') or
                  any(k.startswith('decoder.blocks.%d.' % i) for i in (1, 2, 3))]
        output['shared_initialization_matches_baseline'] = all(torch.equal(model_state[k], ref[k]) for k in shared)
        assert output['shared_initialization_matches_baseline']
        output['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        output['probe_parameters'] = sum(p.numel() for p in model.parameters())
        del reference, ref, model_state
        model = model.cuda().train()
        features = [torch.randn(args.batch, c, side, side, device='cuda', dtype=torch.bfloat16)
                    for c, side in zip(CHANNELS, (64, 32, 16, 8))]
        target = torch.zeros(args.batch, 1, 256, 256, device='cuda')
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target)
        assert torch.isfinite(loss)
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        bad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
        output['missing_gradients'] = missing
        output['nonfinite_gradients'] = bad
        assert not missing and not bad
        opt.step()
        torch.cuda.synchronize()
        output.update({'logit_shape': list(logits.shape), 'loss': loss.item(),
                       'elapsed_seconds': time.monotonic() - start,
                       'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                       'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                       'passed': True})
    except Exception as exc:
        output['error'] = str(exc)
        output['exception_type'] = type(exc).__name__
        output['traceback'] = traceback.format_exc()
    dump('diagnostic_batch_%d.json' % args.batch, output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    if not output['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
