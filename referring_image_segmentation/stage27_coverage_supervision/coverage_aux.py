"""Training-only foreground occupancy supervision on the deepest encoder feature."""
import torch
from torch import nn
from torch.nn import functional as F
WEIGHT=.2

def attach_coverage_head(model):
    # Fork RNG so the original decoder/backbone/shuffle initialization is untouched.
    with torch.random.fork_rng(devices=[]):
        head=nn.Conv2d(768,1,1)
        nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    head.requires_grad_(model.auxiliary)
    model.probe.add_module('coverage_head',head)
    model._capture_coverage=False;model._coverage_logits=None
    def capture(module,inputs,output):
        if model._capture_coverage:
            feature=output[1]
            assert feature.shape[1:]==(768,8,8),feature.shape
            model._coverage_logits=model.probe.coverage_head(feature)
    model.encoder.model.levels[3].register_forward_hook(capture)

def forward_training(model,cache,indices):
    model._capture_coverage=model.auxiliary
    model._coverage_logits=None
    try:
        logits=model.from_cache(cache,indices)
        auxiliary=model._coverage_logits
        if model.auxiliary:assert auxiliary is not None
        return logits,auxiliary
    finally:
        model._capture_coverage=False;model._coverage_logits=None

def occupancy_target(target):
    n,c,h,w=target.shape
    assert c==1 and h==256 and w==256
    binary=(target>=.5).float()
    return binary.reshape(n,1,8,32,8,32).mean((3,5))

def coverage_loss(logits,target):
    if logits is None:return target.new_zeros(())
    return F.binary_cross_entropy_with_logits(logits.float(),occupancy_target(target))

def assert_coverage_gradients(model):
    gradients=[p.grad for p in model.probe.coverage_head.parameters()]
    if model.auxiliary:
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients)>0
    else:assert all(g is None for g in gradients)

def coverage_checks():
    # Include a channel-zero-stride view to guard against the diagnosed pooling issue.
    x=torch.zeros(3,256,256)
    x[1]=1;x[2,:128]=1
    target=x[:,None]
    pooled=occupancy_target(target)
    assert pooled[0].sum()==0 and pooled[1].sum()==64
    assert pooled[2,:,:4].sum()==32 and pooled[2,:,4:].sum()==0
    z=torch.zeros(3,1,8,8,requires_grad=True)
    loss=coverage_loss(z,target)
    assert abs(float(loss)-.6931471805599453)<1e-6
    loss.backward()
    assert (z.grad[0]>0).all() and (z.grad[1]<0).all()
    return dict(passed=True,coverage_head_parameters=769,loss_weight=WEIGHT,
                inference_uses_auxiliary_head=False)
