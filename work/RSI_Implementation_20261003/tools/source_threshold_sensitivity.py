"""CPU-only descriptive sensitivity from immutable per-reference exports.

Original-coordinate hard Dice uses the frozen probability threshold 0.5.
Canonical letterbox/valid-pixel loss supplies soft gains. H/T1 and tool strata
are descriptive and never select checkpoints, methods, or working points.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import glob
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
EPSILONS = (0.0, 0.002, 0.005, 0.01)
FIXED_TWO_SEED = 17
RUN_FIELDS = ('protocol_id', 'seed', 'run_id', 'checkpoint_hash', 'action', 'lambda', 'alpha', 'subset')
REQUIRED = ('run_id', 'checkpoint_hash', 'image_id', 'subset', 'reference_id', 'dice', 'dice_anchor',
            'loss', 'loss_anchor', 'loss_view', 'changed_pixel_fraction', 'tool', 'split')


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def number(row, name):
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{row.get('image_id')}: missing/invalid {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{row.get('image_id')}: nonfinite {name}")
    return value


def read_exports(paths, manifest_hash=None):
    """Parse one byte snapshot per file and reject ambiguous or repeated rows."""
    rows, inputs, seen = [], [], set()
    for path in sorted(set(map(Path, paths))):
        data = path.read_bytes()
        inputs.append({'path': str(path.resolve()), 'bytes': len(data), 'sha256': sha256(data)})
        reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
        if not set(REQUIRED).issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing columns {sorted(set(REQUIRED)-set(reader.fieldnames or []))}')
        count = 0
        for raw in reader:
            row = dict(raw)
            if any(row.get(key) in (None, '') for key in REQUIRED):
                raise ValueError(f'{path}: missing required value')
            if row['split'] != 'val':
                raise ValueError('This pilot sidecar accepts validation exports only; test scoring remains locked')
            if row['subset'] not in {'M', 'H', 'T1'} or row['tool'] not in {'T1', 'T2', 'T3'}:
                raise ValueError('Unknown preregistered subset/tool')
            if row['loss_view'] != 'canonical_letterbox_valid':
                raise ValueError('Soft gain requires explicitly canonical letterbox/valid loss')
            if manifest_hash and row.get('manifest_hash') != manifest_hash:
                raise ValueError(f'{path}: export manifest hash does not match the configured frozen manifest')
            for key, default in [('action', 1), ('alpha', 1), ('lambda', 0)]:
                if row.get(key) in (None, ''):
                    row[key] = default
                row[key] = number(row, key)
            for key in ('dice', 'dice_anchor', 'loss', 'loss_anchor', 'changed_pixel_fraction'):
                row[key] = number(row, key)
            if any(not 0 <= row[key] <= 1 for key in ('dice', 'dice_anchor', 'changed_pixel_fraction')):
                raise ValueError('Dice and changed-pixel fraction must lie in [0,1]')
            row['gain_dice'] = row['dice'] - row['dice_anchor']
            row['gain_loss'] = row['loss_anchor'] - row['loss']
            for gain in ('gain_dice', 'gain_loss'):
                if raw.get(gain) not in (None, '') and not math.isclose(float(raw[gain]), row[gain], abs_tol=1e-7):
                    raise ValueError(f'{path}: {gain} disagrees with its two absolute values')
            key = tuple(str(row.get(k, '')) for k in ('run_id', 'checkpoint_hash', 'image_id', 'subset', 'reference_id', 'action'))
            if key in seen:
                raise ValueError(f'Duplicate complete export key: {key}')
            seen.add(key)
            rows.append(row)
            count += 1
        inputs[-1]['reference_rows'] = count
        if not count:
            raise ValueError(f'{path}: completed export contains no reference rows')
    return rows, inputs


def image_groups(rows):
    images = defaultdict(list)
    for row in rows:
        images[row['image_id']].append(row)
    for image_id, refs in images.items():
        changes = [r['changed_pixel_fraction'] for r in refs]
        if not np.allclose(changes, changes[0], rtol=1e-6, atol=1e-8):
            raise ValueError(f'{image_id}: changed-pixel fraction differs across references of one output')
    return dict(sorted(images.items()))


def average_ranks(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def correlation(x, y, weights=None, *, ranks=False):
    """Descriptive weighted Pearson, or Pearson of average tie ranks."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return None
    if ranks:
        x, y = average_ranks(x), average_ranks(y)
    w = np.ones(len(x), float) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    dx, dy = x - np.dot(w, x), y - np.dot(w, y)
    vx, vy = np.dot(w, dx * dx), np.dot(w, dy * dy)
    if vx <= 0 or vy <= 0:
        return None
    return float(np.clip(np.dot(w, dx * dy) / math.sqrt(vx * vy), -1, 1))


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows, epsilon_d):
    images = image_groups(rows)
    if not images:
        raise ValueError('Cannot summarize an empty stratum')
    gains = [np.array([r['gain_dice'] for r in refs]) for refs in images.values()]
    soft = [np.array([r['gain_loss'] for r in refs]) for refs in images.values()]
    worst = np.array([g.min() for g in gains])
    best = np.array([g.max() for g in gains])
    multi = np.array([len(g) >= 2 for g in gains])
    changed = np.array([refs[0]['changed_pixel_fraction'] > 0 for refs in images.values()])
    exceed = (worst < -epsilon_d) | (best > epsilon_d)
    flip = (worst < -epsilon_d) & (best > epsilon_d) & multi
    n, refs_n = len(gains), sum(map(len, gains))
    canonical_d = [-g for g in soft]
    j_values = [float(np.maximum(d, 0).mean()-max(float(d.mean()), 0)) for d in canonical_d]
    tail_n = max(1, math.ceil(0.1*n))
    out = {
        'epsilon_D': epsilon_d, 'epsilon_L': 0.0, 'n_images': n, 'n_references': refs_n,
        'n_multi_reference_images': int(multi.sum()), 'n_single_reference_images': int((~multi).sum()),
        'mean_gain_dice': float(np.mean([g.mean() for g in gains])),
        'G_plus': float(np.mean([np.maximum(g, 0).mean() for g in gains])),
        'H_minus': float(np.mean([np.maximum(-g, 0).mean() for g in gains])),
        'H_epsilon': float(np.mean([np.maximum(-g-epsilon_d, 0).mean() for g in gains])),
        'any_harm_count': int((worst < -epsilon_d).sum()), 'any_harm_denominator': n,
        'any_harm': float(np.mean(worst < -epsilon_d)),
        'sign_flip_count': int(flip.sum()),
        'sign_flip_all_image_denominator': n, 'sign_flip_all_image_rate': float(flip.mean()),
        'sign_flip_multi_image_denominator': int(multi.sum()),
        'sign_flip': fraction(int(flip.sum()), int(multi.sum())),
        'mask_changed_image_count': int(changed.sum()),
        'sign_flip_changed_denominator': int((multi & changed).sum()),
        'sign_flip_changed_count': int((flip & changed).sum()),
        'sign_flip_changed_rate': fraction(int((flip & changed).sum()), int((multi & changed).sum())),
        'exceed_tolerance_image_count': int(exceed.sum()),
        'sign_flip_exceed_denominator': int((multi & exceed).sum()),
        'sign_flip_exceed_count': int((flip & exceed).sum()),
        'sign_flip_exceed_rate': fraction(int((flip & exceed).sum()), int((multi & exceed).sum())),
        'mean_canonical_loss_gain': float(np.mean([g.mean() for g in soft])),
        'canonical_rho': float(np.mean([(d > 0).mean() for d in canonical_d])),
        'canonical_J_mean': float(np.mean(j_values)),
        'canonical_J_p50': float(np.quantile(j_values, .5)),
        'canonical_J_p90': float(np.quantile(j_values, .9)),
        'canonical_J_p99': float(np.quantile(j_values, .99)),
        'canonical_kappa_pm': float(np.mean([d.min() < -1e-6 and d.max() > 1e-6 for d in canonical_d])),
        'worst_reference_mean': float(worst.mean()),
        'worst_tail_10pct': float(np.sort(worst)[:tail_n].mean()),
        'worst_tail_n': tail_n,
    }
    table_counts, table_macro = Counter(), Counter()
    soft_harm_n = hard_harm_n = both_n = soft_only_n = hard_only_n = discord_images = 0
    discord_macro, flat_dice, flat_soft, weights = [], [], [], []
    classify = lambda value, tolerance: 'positive' if value > tolerance else 'negative' if value < -tolerance else 'neutral'
    for dice, loss in zip(gains, soft):
        sh, hh = loss < 0, dice < -epsilon_d
        disagreement = sh != hh
        soft_harm_n += int(sh.sum()); hard_harm_n += int(hh.sum()); both_n += int((sh & hh).sum())
        soft_only_n += int((sh & ~hh).sum()); hard_only_n += int((~sh & hh).sum())
        discord_images += int(disagreement.any()); discord_macro.append(float(disagreement.mean()))
        for gd, gl in zip(dice, loss):
            key = f'soft_{classify(gl, 0)}_hard_{classify(gd, epsilon_d)}'
            table_counts[key] += 1
            table_macro[key] += 1 / (n * len(dice))
            flat_dice.append(gd); flat_soft.append(gl); weights.append(1 / (n * len(dice)))
    out.update(soft_harm_reference_count=soft_harm_n, hard_harm_reference_count=hard_harm_n,
               both_harm_reference_count=both_n, soft_harm_hard_nonharm_reference_count=soft_only_n,
               soft_nonharm_hard_harm_reference_count=hard_only_n,
               soft_hard_harm_disagreement_reference_count=soft_only_n+hard_only_n,
               soft_hard_harm_disagreement_reference_fraction=(soft_only_n+hard_only_n)/refs_n,
               soft_hard_harm_disagreement_macro_rate=float(np.mean(discord_macro)),
               soft_hard_harm_disagreement_image_count=discord_images,
               soft_hard_harm_disagreement_image_fraction=discord_images/n,
               weighted_reference_gain_pearson=correlation(flat_soft, flat_dice, weights),
               weighted_reference_gain_spearman=correlation(flat_soft, flat_dice, weights, ranks=True),
               image_mean_gain_pearson=correlation([g.mean() for g in soft], [g.mean() for g in gains]),
               image_mean_gain_spearman=correlation([g.mean() for g in soft], [g.mean() for g in gains], ranks=True))
    for loss_sign in ('positive', 'neutral', 'negative'):
        for dice_sign in ('positive', 'neutral', 'negative'):
            key = f'soft_{loss_sign}_hard_{dice_sign}'
            out[key+'_reference_count'] = table_counts[key]
            out[key+'_macro_mass'] = table_macro[key]
    return out


def fixed_two(rows):
    """Exactly reuse rsi.summarize's preregistered seed17 selection rule."""
    selected = []
    for image_id, refs in image_groups(rows).items():
        if len(refs) < 2:
            continue
        refs = sorted(refs, key=lambda r: r['reference_id'])
        digest = hashlib.sha256(f'{FIXED_TWO_SEED}:{image_id}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
        selected.extend(refs[int(index)] for index in rng.choice(len(refs), size=2, replace=False))
    return selected


def analyze(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(k, '')) for k in RUN_FIELDS)].append(row)
    output = []
    for key, refs in sorted(grouped.items()):
        meta = dict(zip(RUN_FIELDS, key))
        strata = [('native_subset', 'all', 'all_references', refs)]
        strata.extend(('tool', tool, 'all_references', [r for r in refs if r['tool'] == tool])
                      for tool in ('T1', 'T2', 'T3') if any(r['tool'] == tool for r in refs))
        two = fixed_two(refs)
        if two:
            strata.append(('native_subset', 'all', 'fixed_two_seed17', two))
        for stratum, tool, selection, selected in strata:
            for epsilon in EPSILONS:
                output.append(dict(meta, stratum=stratum, source_tool=tool, reference_selection=selection,
                                   fixed_two_seed=FIXED_TWO_SEED if selection == 'fixed_two_seed17' else '',
                                   threshold=0.5, selection_use='descriptive_only',
                                   **summarize(selected, epsilon)))
    return output


def write_json(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def run(paths, output_dir, config_path):
    config_path, output_dir = Path(config_path), Path(output_dir)
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if float(config['threshold']) != 0.5 or float(config.get('loss', {}).get('epsilon_L', 0)) != 0:
        raise ValueError('This sidecar freezes threshold0.5 and epsilon_L0; different rules need a preregistered tool revision')
    manifest = Path(config['manifest']) if config.get('manifest') else None
    manifest_hash = sha256(manifest.read_bytes()) if manifest else None
    rows, inputs = read_exports(paths, manifest_hash)
    summaries = analyze(rows)
    for item in inputs:
        if sha256(Path(item['path']).read_bytes()) != item['sha256']:
            raise RuntimeError('Export was replaced during sensitivity computation; rerun after it is complete')
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / 'source_threshold_sensitivity.csv'
    if summaries:
        temporary = output.with_suffix('.tmp.csv')
        with temporary.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader(); writer.writerows(summaries)
        temporary.replace(output)
    metadata = {
        'status': 'complete' if rows else 'awaiting_reference_exports',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'inputs': inputs, 'input_reference_rows': len(rows), 'summary_rows': len(summaries),
        'configuration': {'path': str(config_path.resolve()), 'sha256': sha256(config_bytes),
                          'threshold': 0.5, 'epsilon_D': list(EPSILONS), 'epsilon_L': 0.0,
                          'fixed_two_seed': FIXED_TWO_SEED, 'manifest_sha256': manifest_hash},
        'tool_sha256': sha256(Path(__file__).read_bytes()),
        'output_csv': str(output.resolve()) if summaries else None,
        'output_sha256': sha256(output.read_bytes()) if summaries else None,
        'scope': 'CPU descriptive validation sidecar; no model scoring, training, checkpoint/workpoint selection',
        'source_strata_use': 'M/H/T1 and T1/T2/T3 tool strata describe already selected outputs; H/T1 never select parameters',
        'metric_definitions': {
            'gain_dice': 'original-coordinate dice-current minus dice-anchor at frozen probability threshold0.5',
            'gain_loss': 'canonical_letterbox_valid loss-anchor minus loss-current; positive is improvement',
            'H_epsilon': 'image average of reference mean max(-gain_dice-epsilon_D,0)',
            'G_plus': 'image average of reference mean max(gain_dice,0)',
            'any_harm': 'image min reference gain < -epsilon_D; denominator all stratum images',
            'sign_flip': 'min gain < -epsilon_D and max gain > epsilon_D; primary denominator all eligible multi-reference images',
            'additional_sign_flip_denominators': 'all stratum images, eligible images with actual original-mask change, eligible images with at least one abs gain exceeding tolerance; zero denominator is missing, not0',
            'tool_single_reference_images': 'reported explicitly; cannot exhibit reference sign conflict and are excluded from the eligible multi-reference denominator',
            'soft_harm': 'canonical gain_loss <0 (epsilon_L0); hard harm uses gain_dice < -epsilon_D',
            'canonical_rho_J_kappa': 'd=-canonical gain_loss; rho=Eir I(d>0), J=Ei[Er relu(d)-relu(Er d)], kappa=Ei I(min d<-1e-6 and max d>1e-6); diagnostic, not an added objective',
            'worst_tail_10pct': 'sort each image minimum reference Dice gain; average first max(1,ceil0.1N) images',
            'disagreement_counts': 'raw reference/image descriptive counts, plus image-then-reference macro rates; references are not independent samples',
            'correlations': 'weighted reference Pearson and Pearson of average tie ranks (weights1/N/m); unweighted per-image mean counterparts; undefined constant-vector correlation is missing',
            'fixed_two': 'same sha256(seed17:image_id), first8bytes big-endian NumPy RNG, sorted reference_id and choice2 without replacement as frozen rsi.summarize',
            'native_source_sensitivity': 'existing original summarizer separately compares independently selected M/H/T1 references on the same source-image intersection; this tool does not overwrite those tables',
        },
        'test_scoring_locked': True,
    }
    write_json(output_dir/'source_threshold_sensitivity.json', metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='*', default=None, help='CSV paths or glob patterns')
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/p2_ima_m_v2.yaml')
    parser.add_argument('--output-dir', type=Path, default=PROJECT/'outputs/seed17/sensitivity')
    args = parser.parse_args()
    patterns = args.inputs if args.inputs is not None else [str(PROJECT/'outputs/per_reference/seed17/*.csv')]
    paths = [Path(path) for pattern in patterns for path in glob.glob(pattern)]
    metadata = run(paths, args.output_dir, args.config)
    print(json.dumps({'status': metadata['status'], 'inputs': len(metadata['inputs']),
                      'reference_rows': metadata['input_reference_rows'], 'summary_rows': metadata['summary_rows'],
                      'output': metadata['output_csv']}))


if __name__ == '__main__':
    main()
