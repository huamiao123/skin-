"""Re-evaluate stage19 best checkpoints; requires trusted assets and CUDA."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16',
                        help='fp32 is a compatible diagnostic, not the historical BF16 evaluation')
    parser.add_argument('--strict', action='store_true', help='require identical historical pixel counts')
    args = parser.parse_args()
    project = args.project.resolve()
    stage = project / 'stage19_region_mambavision_encoder_seed42'
    sys.path.insert(0, str(stage))
    from common import setup, torch, np
    from models import Segmentor, segmentation_loss
    from train import load_trainable
    from evaluation import aggregate_counts
    from stage3_confirmation.data import ImageData

    if args.batch < 1 or not torch.cuda.is_available():
        raise RuntimeError('A positive --batch and CUDA are required')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support BF16; use --precision fp32 for a non-equivalent diagnostic')
    setup()
    data = ImageData(42)
    expected_members = json.loads((project / 'stage1_visual/data_manifest.json').read_text())['members']
    if data.audit != expected_members:
        raise RuntimeError('Data content or split differs from the historical manifest')
    results = {}
    for name in ('ordered_prior', 'region_scan'):
        path = stage / 'results' / name / 'best_dice.pth'
        # This checkpoint is a pickle: only load trusted files.
        ckpt = torch.load(path, map_location='cpu')
        model = Segmentor(name).cuda().eval()
        if ckpt['criterion'] != 'best_dice' or ckpt['seed'] != 42:
            raise RuntimeError('Unexpected checkpoint selection or seed')
        if model.encoder.frozen_state_hash() != ckpt['frozen_hash']:
            raise RuntimeError('Frozen encoder is not the one used by this checkpoint')
        load_trainable(model, ckpt['model'])
        # Mirrors the historical engine; the fp32 option only changes autocast.
        counts = []
        loss_sum = 0.
        with torch.no_grad():
            for start in range(0, len(data.ids['val']), args.batch):
                end = min(start + args.batch, len(data.ids['val']))
                images, gt = data.batch('val', torch.arange(start, end, device='cuda'))
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                    logits, _ = model(images)
                loss = segmentation_loss(logits, gt)
                pred, truth = logits.float().sigmoid() >= .5, gt >= .5
                parts = [(pred & truth).sum((1, 2, 3)), (pred & ~truth).sum((1, 2, 3)),
                         (~pred & truth).sum((1, 2, 3)), (~pred & ~truth).sum((1, 2, 3))]
                counts.append(torch.stack(parts, 1).cpu().numpy())
                loss_sum += float(loss.item()) * (end - start)
        counts = np.concatenate(counts, axis=0).astype(np.int64)
        actual = aggregate_counts(*counts.T)
        actual['loss'] = loss_sum / len(data.ids['val'])
        identical = actual['counts'] == ckpt['metrics']['counts']
        results[name] = dict(epoch=ckpt['epoch'], historical=ckpt['metrics'],
                             recomputed=actual, exact_pixel_counts_match=identical)
        if args.strict and not identical:
            raise RuntimeError(f'{name}: count mismatch; check hardware, precision and dependencies')
        del model
        torch.cuda.empty_cache()
    print(json.dumps(dict(seed=42, n=808, precision=args.precision,
                          device=torch.cuda.get_device_name(), results=results), indent=2))


if __name__ == '__main__':
    main()
