import torch

from candidate_layout import CANDIDATE_COUNT,ZERO_INDEX
from local_model import StrongLocal
from train_gated_refiner import losses as gate_losses
from train_local import loss_and_accuracy
from evaluation import repair_metrics
from evaluate_local import choose_dp
import numpy as np


def case(distance, invalid=(), has=True):
    d=torch.full((1,2,CANDIDATE_COUNT),10.0)
    d[:,:,ZERO_INDEX]=distance
    valid=torch.ones_like(d,dtype=torch.bool)
    for i in invalid:valid[:,:,i]=False
    if not has:valid[:]=False
    return {'gt_distance':d,'valid':valid,'has_contour':torch.tensor([has])}


def test_unreachable_keep_with_invalid_negative_radius():
    batch=case(10,invalid=(0,))
    scores=torch.zeros((1,2,CANDIDATE_COUNT),requires_grad=True)
    loss,accuracy=loss_and_accuracy(scores,batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.grad[0,0,ZERO_INDEX]<0
    assert scores.grad[0,0,0]==0


def test_already_correct_and_repair_direction():
    batch=case(0)
    scores=torch.zeros((1,2,CANDIDATE_COUNT),requires_grad=True)
    loss_and_accuracy(scores,batch)[0].backward()
    assert scores.grad[0,0].argmin().item()==ZERO_INDEX
    batch=case(10)
    batch['gt_distance'][:,:,ZERO_INDEX+1]=0
    scores=torch.zeros((1,2,CANDIDATE_COUNT),requires_grad=True)
    loss_and_accuracy(scores,batch)[0].backward()
    assert scores.grad[0,0].argmin().item()==ZERO_INDEX+1


def test_no_contour_and_invalid_zero_guard():
    empty=case(10,has=False)
    scores=torch.zeros((1,2,CANDIDATE_COUNT),requires_grad=True)
    loss,accuracy=loss_and_accuracy(scores,empty)
    assert loss.item()==0 and accuracy.item()==0
    invalid=case(10,invalid=(ZERO_INDEX,))
    try:loss_and_accuracy(scores,invalid)
    except ValueError as exc:assert 'zero-offset' in str(exc)
    else:raise AssertionError('Invalid keep candidate was accepted')


def test_gate_keep_and_repair_gradient():
    batch=case(0)
    scores=torch.zeros((1,2,CANDIDATE_COUNT),requires_grad=True)
    gate=torch.zeros((1,2),requires_grad=True)
    gate_losses(scores,gate,batch)[0].backward()
    assert scores.grad[0,0].argmin().item()==ZERO_INDEX
    assert gate.grad[0,0]>0
    batch=case(10)
    batch['gt_distance'][:,:,ZERO_INDEX+1]=0
    scores=torch.zeros((1,2,CANDIDATE_COUNT),requires_grad=True)
    gate=torch.zeros((1,2),requires_grad=True)
    gate_losses(scores,gate,batch)[0].backward()
    assert scores.grad[0,0].argmin().item()==ZERO_INDEX+1
    assert gate.grad[0,0]<0


def test_new_local_module_ignores_remote_feature_change():
    torch.manual_seed(7)
    model=StrongLocal(width=64,normalization='pixel_group').eval()
    f=torch.randn(1,32,256,256);rgb=torch.randn(1,3,256,256)
    prob=torch.randn(1,1,256,256);boundary=torch.randn(1,1,256,256)
    points=torch.tensor([[[[128.0,128.0]]]])
    source=points[:,:,0,:]
    normal=torch.tensor([[[1.0,0.0]]])
    with torch.no_grad():
        first=model(f,rgb,prob,boundary,points,source,normal)
        modified=f.clone();modified[:,:,:8,:8]+=10
        second=model(modified,rgb,prob,boundary,points,source,normal)
    assert torch.equal(first,second)


def test_movement_metric_distinguishes_beneficial_change():
    d=np.full((1,CANDIDATE_COUNT),10.0)
    d[0,ZERO_INDEX]=1.5;d[0,ZERO_INDEX+1]=0.5
    metrics=repair_metrics(np.array([ZERO_INDEX+1]),np.ones_like(d,bool),d)
    assert metrics['move_rate_on_initially_correct']==1
    assert metrics['distance_increase_rate']==0
    assert metrics['correct_to_incorrect_rate']==0


def test_dp_prunes_after_displacement_cost():
    score=np.zeros((4,CANDIDATE_COUNT))
    score[:,:8]=2.0
    score[:,ZERO_INDEX+1]=1.9
    chosen=choose_dp(score,np.ones_like(score,bool),lam=0.0,displacement=0.1)
    assert np.array_equal(chosen,np.full(4,ZERO_INDEX+1))
