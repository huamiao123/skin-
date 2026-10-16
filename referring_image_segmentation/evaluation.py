"""Shared foreground metric definitions; aggregation is always explicit."""
import numpy as np

PREC_THRESHOLDS = (.5, .6, .7, .8, .9)
METRIC_VERSION = 'foreground_v1_20261016'


def divide(num, den, default=1.):
    num, den = np.broadcast_arrays(np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64))
    out = np.broadcast_to(np.asarray(default, dtype=np.float64), num.shape).copy()
    return np.divide(num, den, out=out, where=den != 0)


def per_sample_counts(tp, fp, fn, tn):
    tp, fp, fn, tn = [np.asarray(x, dtype=np.float64) for x in (tp, fp, fn, tn)]
    return dict(dice=divide(2*tp, 2*tp+fp+fn), iou=divide(tp, tp+fp+fn),
                precision=divide(tp, tp+fp, tp+fn == 0), recall=divide(tp, tp+fn))


def aggregate_counts(tp, fp, fn, tn):
    counts = [np.asarray(x, dtype=np.int64) for x in (tp, fp, fn, tn)]
    if not all(x.ndim == 1 and x.shape == counts[0].shape for x in counts) or not len(counts[0]):
        raise ValueError('Expected equally sized, nonempty per-sample count vectors')
    if any(np.any(x < 0) for x in counts):
        raise ValueError('Negative pixel counts')
    samples = per_sample_counts(*counts)
    totals = [int(x.sum()) for x in counts]
    pooled = {k: float(v) for k, v in per_sample_counts(*totals).items()}
    return dict(metric_version=METRIC_VERSION, n=len(counts[0]),
                macro={k: float(v.mean()) for k, v in samples.items()}, pooled=pooled,
                prec_at={f'{t:.1f}': float((samples['iou'] >= t).mean()) for t in PREC_THRESHOLDS},
                threshold_jaccard_065=float(np.where(samples['iou'] < .65, 0., samples['iou']).mean()),
                counts=dict(zip(('tp','fp','fn','tn'), totals)))


def summarize_rows(rows):
    result = aggregate_counts(*[[r[k] for r in rows] for k in ('tp','fp','fn','tn')])
    for key in ('boundary_f1', 'hd95_pixels', 'hd95_diagonal'):
        result['macro'][key] = float(np.mean([r[key] for r in rows]))
    return result
