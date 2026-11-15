"""CPU-only diagnosis of saved development predictions; no model training."""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
STAGES = ('stage22_partial_finetune', 'stage23_encoder_lr3e5',
          'stage24_seed43_confirmation')
MODELS = ('frozen', 'finetune34')
THRESHOLDS = np.array([.001, .01, .025] + [x/100 for x in range(5, 100, 5)]
                      + [.975, .99, .999], dtype=np.float32)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b, empty=0.):
    return float(a / b) if b else empty


def metrics(tp, fp, fn, tn):
    return dict(dice=ratio(2*tp, 2*tp+fp+fn, 1.),
                iou=ratio(tp, tp+fp+fn, 1.),
                precision=ratio(tp, tp+fp, float(tp+fn == 0)),
                recall=ratio(tp, tp+fn, 1.))


def threshold_counts(prob, gt):
    bins = np.searchsorted(THRESHOLDS, prob.ravel(), side='right')
    truth = gt.ravel()
    pos = np.bincount(bins[truth], minlength=len(THRESHOLDS)+1)
    neg = np.bincount(bins[~truth], minlength=len(THRESHOLDS)+1)
    tp = np.cumsum(pos[::-1])[::-1][1:]
    fp = np.cumsum(neg[::-1])[::-1][1:]
    return np.stack((tp, fp, truth.sum()-tp, (~truth).sum()-fp), axis=1)


def exact_threshold_oracle(prob, gt):
    """Max Dice over distinct threshold masks, never break equal-value ties."""
    if not gt.any():
        return 1., float(np.nextafter(prob.max(), np.float32(np.inf)))
    order = np.argsort(-prob.ravel(), kind='stable')
    values = prob.ravel()[order]
    positive = gt.ravel()[order]
    tp = np.cumsum(positive)
    ends = np.flatnonzero(np.r_[values[:-1] != values[1:], True])
    scores = 2*tp[ends] / (ends+1+int(gt.sum()))
    best = int(np.argmax(scores))
    return float(scores[best]), float(values[ends[best]])


def aggregate(rows):
    if not rows:
        return dict(n=0)
    out = dict(n=len(rows))
    for key in ('dice', 'iou', 'precision', 'recall'):
        out['macro_' + key] = float(np.mean([float(r[key]) for r in rows]))
    total = {k:sum(int(r[k]) for r in rows) for k in ('tp', 'fp', 'fn', 'tn')}
    out['counts'] = total
    out['pooled'] = metrics(**total)
    if 'oracle_dice' in rows[0]:
        out['oracle_macro_dice'] = float(np.mean([r['oracle_dice'] for r in rows]))
        out['oracle_mean_gain_over_point5'] = out['oracle_macro_dice']-out['macro_dice']
        for key in ('fn_p_lt_01', 'fn_interior3', 'fn_interior8', 'fn_interior8_p_lt_01'):
            out[key] = sum(r[key] for r in rows)
            out[key + '_fraction_of_fn'] = ratio(out[key], total['fn'])
        out['interior8_false_negative_rate'] = ratio(
            out['fn_interior8'], sum(r['gt_interior8'] for r in rows))
        out['mean_pred_to_gt_area'] = float(np.mean([r['pred_to_gt_area'] for r in rows]))
        out['severe_undercoverage_n'] = sum(r['recall'] < .5 for r in rows)
    return out


def grouped(rows, metadata):
    groups = {'all': rows}
    for quartile in range(1, 5):
        qrows = [r for r in rows if metadata[r['id']]['area_quartile'] == quartile]
        groups['area_q' + str(quartile)] = qrows
        for touch in (False, True):
            groups[f'area_q{quartile}_frame_{int(touch)}'] = [
                r for r in qrows if metadata[r['id']]['frame_contact'] == touch]
    for touch in (False, True):
        groups['frame_' + str(int(touch))] = [
            r for r in rows if metadata[r['id']]['frame_contact'] == touch]
    return {k:aggregate(v) for k,v in groups.items()}


def prepare_metadata(ids):
    base = PROJECT / STAGES[0] / 'results'
    old = list(csv.DictReader((base/'per_image.csv').open()))
    chosen = {r['id']:r for r in old if r['model']=='frozen' and r['criterion']=='best_dice'}
    cuts = np.quantile([int(chosen[s]['tp'])+int(chosen[s]['fn']) for s in ids], [.25,.5,.75])
    metadata, targets, distances = {}, [], []
    mask_digest = hashlib.sha256()
    for sid in ids:
        path = PROJECT/'data/isic2018/val/masks'/(sid+'.png')
        with Image.open(path) as image:
            gt = np.asarray(image.convert('L').resize((256,256), Image.Resampling.NEAREST)) >= 128
        mask_digest.update(sid.encode()+b'\0'+gt.tobytes())
        area = int(gt.sum())
        assert area == int(chosen[sid]['tp'])+int(chosen[sid]['fn'])
        edges = np.r_[gt[0],gt[-1],gt[1:-1,0],gt[1:-1,-1]]
        metadata[sid] = dict(area=area, area_quartile=int(np.searchsorted(cuts,area,side='right'))+1,
                             frame_contact=bool(edges.any()), frame_foreground_fraction=float(edges.mean()))
        dist = distance_transform_edt(np.pad(gt,1,constant_values=False))[1:-1,1:-1]
        targets.append(gt)
        distances.append(dist.astype(np.float32))
    return metadata, targets, distances, cuts.tolist(), mask_digest.hexdigest()


def diagnose(stage):
    assert stage in STAGES
    source = PROJECT/stage/'results'
    assert (source/'DONE.json').exists(), 'Source experiment not complete'
    out = ROOT/'results'/stage
    assert not (out/'DONE.json').exists(), 'Completed diagnostic is immutable'
    out.mkdir(parents=True, exist_ok=True)
    partition = json.loads((source/'active_partition.json').read_text())
    reference = json.loads((PROJECT/STAGES[0]/'results/active_partition.json').read_text())
    assert partition == reference and partition['counts']['val'] == 762
    ids = partition['splits']['val']
    metadata, targets, distances, cuts, mask_hash = prepare_metadata(ids)
    old = list(csv.DictReader((source/'per_image.csv').open()))
    inputs = ['DONE.json','config.json','summary.json','per_image.csv','active_partition.json']
    inputs += [m+'_probabilities.npy' for m in MODELS]
    provenance = dict(stage=stage, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input_sha256={name:digest(source/name) for name in inputs},
        source_sha256={name:digest(ROOT/name) for name in ('diagnose.py','PROTOCOL.txt')},
        decoded_validation_masks_sha256=mask_hash, area_quartile_cutpoints_pixels=cuts,
        n_validation=len(ids), inference_performed=False, test_evaluated=False)
    dump(out/'provenance.json', provenance)
    outputs = {}
    predictions = {}
    for model in MODELS:
        print(stage, model, 'diagnosing', flush=True)
        probs = np.load(source/(model+'_probabilities.npy'), mmap_mode='r')
        assert probs.shape == (len(ids),256,256)
        selected = [r for r in old if r['model']==model and r['criterion']=='best_dice']
        assert [r['id'] for r in selected] == ids, 'Probability order mismatch'
        rows, curves, binary = [], [], []
        for i,(sid,gt,dist) in enumerate(zip(ids,targets,distances)):
            prob = np.asarray(probs[i])
            assert np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
            pred = prob>=.5
            fn = gt & ~pred
            counts = dict(tp=int((gt&pred).sum()),fp=int((~gt&pred).sum()),
                          fn=int(fn.sum()),tn=int((~gt&~pred).sum()))
            assert all(counts[k] == int(selected[i][k]) for k in counts), (sid,model,'counts differ')
            oracle, threshold = exact_threshold_oracle(prob,gt)
            row = dict(id=sid, **metadata[sid], **counts, **metrics(**counts),
                oracle_dice=oracle, oracle_threshold=threshold,
                pred_to_gt_area=ratio(int(pred.sum()),int(gt.sum())),
                fn_p_lt_01=int((gt&(prob<.1)).sum()),
                fn_interior3=int((fn&(dist>3)).sum()),
                fn_interior8=int((fn&(dist>8)).sum()),
                gt_interior8=int((dist>8).sum()),
                fn_interior8_p_lt_01=int((gt&(dist>8)&(prob<.1)).sum()))
            assert oracle+1e-12 >= row['dice']
            rows.append(row)
            curves.append(threshold_counts(prob,gt))
            binary.append(pred)
        predictions[model] = binary
        write_csv(out/(model+'_per_image.csv'),rows)
        threshold_rows = []
        for t, threshold in enumerate(THRESHOLDS):
            at_t=[]
            for sid, curve in zip(ids,curves):
                counts=dict(zip(('tp','fp','fn','tn'),map(int,curve[t])))
                at_t.append(dict(id=sid, **counts, **metrics(**counts)))
            threshold_rows.append(dict(threshold=float(threshold), groups=grouped(at_t,metadata)))
        selectors = {}
        for criterion in ('best_dice','best_loss','last'):
            selectors[criterion] = grouped([r for r in old if r['model']==model and r['criterion']==criterion],metadata)
        outputs[model] = dict(groups=grouped(rows,metadata), threshold_curve=threshold_rows,
                              existing_checkpoint_criteria=selectors)
    changes=[]
    for i,(sid,gt,dist) in enumerate(zip(ids,targets,distances)):
        base, fine = predictions['frozen'][i], predictions['finetune34'][i]
        changes.append(dict(id=sid, **metadata[sid],
            newly_missed_gt=int((gt&base&~fine).sum()),
            newly_missed_gt_interior8=int((gt&base&~fine&(dist>8)).sum()),
            recovered_gt=int((gt&~base&fine).sum()),
            removed_fp=int((~gt&base&~fine).sum()),
            introduced_fp=int((~gt&~base&fine).sum())))
    write_csv(out/'paired_error_transitions.csv',changes)
    result = dict(**provenance, models=outputs,
                  paired_transition_totals={k:sum(r[k] for r in changes) for k in changes[0]
                    if k not in ('id','area','area_quartile','frame_contact','frame_foreground_fraction')},
                  limitations='Exploratory same-development-image diagnosis. Oracle uses labels; threshold curves are not new official scores. Associations do not identify a mechanism.')
    dump(out/'report.json',result)
    dump(out/'DONE.json',dict(state='complete',report_sha256=digest(out/'report.json')))
    print(stage, 'COMPLETE', flush=True)


def self_test():
    for gt in (np.array([[1,0],[1,0]],bool),np.zeros((2,2),bool),np.ones((2,2),bool)):
        for prob in (np.array([[.5,.5],[.1,.9]],np.float32),np.full((2,2),.4,np.float32)):
            counts = threshold_counts(prob,gt)
            for t,threshold in enumerate(THRESHOLDS):
                pred=prob>=threshold
                assert counts[t].tolist() == [int((pred&gt).sum()),int((pred&~gt).sum()),int((~pred&gt).sum()),int((~pred&~gt).sum())]
            value,_=exact_threshold_oracle(prob,gt)
            candidates=list(np.unique(prob))+[float(np.nextafter(prob.max(),np.float32(np.inf)))]
            brute=max(metrics(int(((prob>=t)&gt).sum()),int(((prob>=t)&~gt).sum()),
                              int(((prob<t)&gt).sum()),int(((prob<t)&~gt).sum()))['dice'] for t in candidates)
            assert abs(value-brute)<1e-12
    print('SELF_TEST_PASSED: threshold equality, ties, empty/full GT, exact oracle',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stages',nargs='+',choices=STAGES)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    for stage in args.stages or []:
        diagnose(stage)
