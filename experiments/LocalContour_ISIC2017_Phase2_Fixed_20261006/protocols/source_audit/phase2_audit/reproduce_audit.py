"""Read-only synthetic audit of the uploaded Phase-2 source.
Usage: python reproduce_audit.py --repo /path/to/skin--main --out audit_results.json
No model checkpoint, dataset, or GPU is required. Does not modify the repository.
It reproduces detected bugs (rather than treating them as correct behavior).
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('audit_results.json'))
    args = parser.parse_args()
    phase2 = args.repo.resolve()/'experiments/LocalContour_ISIC2017_Phase2_20261005'
    if not (phase2/'train_local.py').is_file():
        raise FileNotFoundError(f'Phase-2 directory not found: {phase2}')
    sys.path.insert(0, str(phase2))
    from train_local import loss_and_accuracy
    from train_gated_refiner import losses
    from local_model import StrongLocal, parameter_count
    from refiner_model import NodeRefiner
    from evaluation import full_metrics, repair_metrics, selected_mask
    from radius_sensitivity import dense
    from contour.geometry import extract_geometry, closed_dp, path_objective
    from dataclasses import replace

    torch.set_num_threads(2)
    torch.manual_seed(1705)
    result = {'environment': {'torch':torch.__version__, 'numpy':np.__version__, 'device':'cpu'},
              'scope':'Synthetic source-function tests and archived CSV checks; no training or original-image inference',
              'findings':{}, 'checks':{}}

    # Actual candidate constructor, not an assumed candidate ordering.
    pred=np.zeros((256,256),bool); pred[64:192,64:192]=True
    geometry=replace(extract_geometry(pred,256,16),search_radius=32.)
    field=dense(np.full(pred.shape,0.5),geometry)
    result['checks']['dense_order']={
        'index0_offset':float(field.offsets[0,0]),
        'index32_offset':float(field.offsets[0,32]),
        'index64_offset':float(field.offsets[0,64]),
        'index32_equals_source':bool(np.array_equal(field.points[:,32],geometry.points))}

    # All candidates too far from the GT: the documented label should be KEEP.
    valid=torch.ones((1,1,65),dtype=torch.bool)
    b={'valid':valid,'has_contour':torch.ones(1,dtype=torch.bool),
       'gt_distance':torch.full((1,1,65),10.)}
    logits=torch.zeros((1,1,65),requires_grad=True)
    loss,acc=loss_and_accuracy(logits,b); loss.backward()
    preferred=int(logits.grad[0,0].argmin())
    result['findings']['unreachable_keep_targets_negative32']={
        'bug_reproduced':preferred==0,'expected_target_index':32,'actual_target_index':preferred,
        'actual_target_offset':preferred-32,'loss_at_uniform_logits':float(loss.detach()),
        'gradient_index0':float(logits.grad[0,0,0]),'gradient_index32':float(logits.grad[0,0,32])}

    # The same wrong label can even point to a candidate that is masked invalid.
    invalid_b={k:v.clone() for k,v in b.items()}; invalid_b['valid'][...,0]=False
    invalid_b['gt_distance'][...,0]=1000.
    invalid_logits=torch.zeros((1,1,65),requires_grad=True)
    bad_loss,_=loss_and_accuracy(invalid_logits,invalid_b);bad_loss.backward()
    corrected_target=F.one_hot(torch.full((1,1),32,dtype=torch.long),65).float()
    masked=invalid_logits.detach().masked_fill(~invalid_b['valid'],-1e4)
    correct_loss=-(corrected_target*F.log_softmax(masked,dim=-1)).sum(-1).mean()
    result['findings']['target_on_invalid_candidate']={
        'bug_reproduced':float(bad_loss.detach())>10000.,
        'current_loss':float(bad_loss.detach()),'corrected_keep_loss':float(correct_loss),
        'invalid_target_logit_gradient':float(invalid_logits.grad[0,0,0]),
        'valid_logit_gradient_sum':float(invalid_logits.grad[invalid_b['valid']].sum())}

    # Gate says KEEP because the original boundary distance is zero.
    gate_b={k:v.clone() for k,v in b.items()};gate_b['gt_distance'][...,32]=0.
    s=torch.zeros((1,1,65),requires_grad=True);g=torch.zeros((1,1),requires_grad=True)
    total,c,gl,ga=losses(s,g,gate_b);total.backward()
    result['findings']['gated_keep_targets_negative32']={
        'bug_reproduced':int(s.grad[0,0].argmin())==0,
        'zero_candidate_gt_distance':0.,'expected_candidate_label':32,
        'actual_candidate_label':int(s.grad[0,0].argmin()),
        'gate_gradient':float(g.grad[0,0]),
        'note':'positive gate gradient trains KEEP, but candidate gradient trains index 0 (-32px)'}

    # Global dependence introduced by the NEW GroupNorm layers, holding frozen maps fixed.
    local=StrongLocal(width=64).eval()
    x=torch.randn(1,37,256,256)
    changed=x.clone();changed[:,:,:32,:32]+=10.
    offsets=torch.arange(-32,33).float()
    points=torch.stack([128.+offsets,torch.full_like(offsets,128.)],dim=-1)[None,None]
    source=torch.tensor([[[128.,128.]]]);normals=torch.tensor([[[1.,0.]]])
    def score(m,inp):
        return m(inp[:,:32],inp[:,32:35],inp[:,35:36],inp[:,36:37],points,source,normals)
    no_norm=copy.deepcopy(local)
    for index,layer in enumerate(no_norm.encoder):
        if isinstance(layer,nn.GroupNorm):no_norm.encoder[index]=nn.Identity()
    with torch.no_grad():
        a=score(local,x);c=score(local,changed)
        an=score(no_norm,x);cn=score(no_norm,changed)
    delta=float((a-c).abs().max());delta_no_norm=float((an-cn).abs().max())
    result['findings']['new_local_network_has_global_dependence']={
        'bug_reproduced':delta>1e-7 and delta_no_norm==0.,
        'perturbation':'add 10 to all input-map channels inside remote top-left 32x32 region',
        'candidate_region':'x=96..160, y=128, unchanged across both inputs',
        'max_candidate_logit_change_with_GroupNorm':delta,
        'max_candidate_logit_change_without_GroupNorm':delta_no_norm,
        'zero_candidate_logit_change':float((a-c).abs()[0,0,32])}

    # Attention mask correctness and per-layer vs composed information radius.
    stats={}
    common=None
    for name,ran in [('local8',8),('medium32',32),('global',None)]:
        model=NodeRefiner(ran).eval()
        if common is None:common=copy.deepcopy(model.state_dict())
        else:model.load_state_dict(common)
        allowed=torch.ones(256,256,dtype=torch.bool) if model.attention_mask is None else ~model.attention_mask
        two=(allowed.int()@allowed.int())>0
        stats[name]={'parameters':parameter_count(model),'allowed_in_one_layer':int(allowed[0].sum()),
                     'reachable_in_two_layers':int(two[0].sum()),'wraparound_0_to255_allowed':bool(allowed[0,255])}
    result['checks']['attention_masks']=stats
    result['checks']['total_additional_parameters']={'local_small':parameter_count(StrongLocal(64)),
        'local_large':parameter_count(StrongLocal(96)),
        'refiner_only':stats['global']['parameters'],
        'local_small_plus_refiner':parameter_count(StrongLocal(64))+stats['global']['parameters']}

    # Wrong-move metric calls a move wrong even when it improves an already-close point.
    d=np.full((1,65),20.);d[0,32]=1.5;d[0,33]=0.5
    metrics=repair_metrics(np.array([33]),np.ones((1,65),bool),d)
    result['findings']['wrong_move_metric_is_any_movement_on_near_correct_nodes']={
        'bug_reproduced':metrics['repair_precision']==1. and metrics['wrong_move_rate']==1.,
        'distance_before':1.5,'distance_after':0.5,'metrics':metrics}

    # Reconstruction KEEP is not an exact identity in general.
    yy,xx=np.mgrid[:256,:256]
    radial=np.sqrt((xx-128)**2+(yy-128)**2)
    theta=np.arctan2(yy-128,xx-128)
    p=radial<(65.+5.*np.sin(37.*theta))
    geo=replace(extract_geometry(p,256,16),search_radius=32.)
    f=dense(np.full(p.shape,.5),geo)
    rebuilt=selected_mask(p,f.points,np.full(256,32,dtype=int))
    result['checks']['zero_reconstruction']={'changed_pixels':int(np.count_nonzero(rebuilt!=p)),
        'original_area':int(p.sum()),'rebuilt_area':int(rebuilt.sum()),
        'note':'Expected approximation effect of polygon resampling, not a mislabeled offset.'}

    # Pruning raw scores before applying displacement can remove the best final unary.
    from evaluate_local import choose_dp
    dp_scores=np.zeros((4,65),np.float64)
    dp_scores[:,:8]=2.0
    dp_scores[:,33]=1.9
    dp_valid=np.ones_like(dp_scores,dtype=bool)
    chosen_pruned=choose_dp(dp_scores,dp_valid,lam=0.,displacement=.1)
    chosen_full=np.argmax(dp_scores-.1*np.abs(np.arange(-32,33))[None,:],axis=-1)
    result['findings']['DP_prepenalty_pruning_changes_unary_optimum']={
        'demonstrated':bool(np.all(chosen_pruned!=chosen_full)),
        'top8_raw_score_offsets':list(range(-32,-24)),
        'near_candidate_offset':1, 'near_candidate_score':1.9,
        'far_candidate_scores':2.0,'alpha':.1,'lambda':0.,
        'pruned_selected_offset':(chosen_pruned-32).tolist(),
        'full_selected_offset':(chosen_full-32).tolist(),
        'classification':'Disclosed candidate-pruning approximation, not an error in the cyclic-DP recurrence'}

    # Check producer-consumer metadata mismatch without CNN inference or fake cache approval.
    tree=ast.parse((phase2/'build_feature_cache.py').read_text())
    producer_keys=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='complete' for t in node.targets):
            producer_keys=[key.value for key in node.value.keys if isinstance(key,ast.Constant)]
    result['findings']['fresh_cache_metadata_mismatch']={
        'fresh_feature_manifest_keys':producer_keys,
        'training_requires':'gt_shape_verified_for_all == 2150',
        'missing_required_key':'gt_shape_verified_for_all' not in producer_keys,
        'note':'Historical repair_gt_cache writes this key but intentionally rejects already-correct fresh GT.'}

    # DP can be checked independently by exhaustive enumeration of a tiny cyclic field.
    rng=np.random.default_rng(17);maxerr=0.
    for _ in range(20):
        cost=rng.uniform(0,2,(4,3));off=np.tile([-2.,0.,2.],(4,1));lam=float(rng.uniform(0,1))
        path,value=closed_dp(cost,off,lam)
        brute=min(path_objective(np.array(p),cost,off,lam) for p in itertools.product(range(3),repeat=4))
        maxerr=max(maxerr,abs(value-brute))
    result['checks']['cyclic_dp_vs_bruteforce']={'trials':20,'max_objective_error':float(maxerr),'passed':maxerr<1e-10}

    # Stable metric tests; conventions documented rather than declared universally wrong.
    mask=np.zeros((256,256),bool);mask[80:160,80:160]=True
    identity=full_metrics(mask,mask);empty=full_metrics(np.zeros_like(mask),mask)
    result['checks']['segmentation_metrics_sanity']={'identical':identity,'empty_vs_nonempty':empty,
        'passed':identity['dice']==1. and identity['iou']==1. and identity['bf1']==1. and identity['hd95']==0. and empty['dice']==0.}

    files=sorted(phase2.glob('*.py'))
    for f in files:ast.parse(f.read_text(),filename=str(f))
    result['checks']['source_files']={'parsed':len(files),'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
