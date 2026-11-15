"""Pinned Microsoft LoRA Linear; original MambaVision and native UPer64 preserved."""
from pathlib import Path
import hashlib,importlib.util,contextlib
import torch
from torch import nn
from stage45_upernet_decoder.author_modules import NativeUPerProbe
ROOT=Path(__file__).resolve().parent
AUTHOR=ROOT.parent.parent.parent/'research/peft_followup_20261112/author_sources/microsoft__LoRA/loralib/layers.py'
spec=importlib.util.spec_from_file_location('stage49_pinned_loralib',AUTHOR)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
LoRALinear=native.Linear

def inject_lora(encoder,rank=8,alpha=8,seed=142):
    targets=[]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for si in (2,3):
            for bi,block in enumerate(encoder.model.levels[si].blocks):
                kind=type(block.mixer).__name__
                names={'MambaVisionMixer':['in_proj','out_proj'],'Attention':['qkv','proj']}[kind]
                for parent,prefix,fields in [
                    (block.mixer,f'model.levels.{si}.blocks.{bi}.mixer',names),
                    (block.mlp,f'model.levels.{si}.blocks.{bi}.mlp',['fc1','fc2'])]:
                    for field in fields:
                        old=getattr(parent,field);assert type(old) is nn.Linear
                        new=LoRALinear(old.in_features,old.out_features,r=rank,lora_alpha=alpha,
                            lora_dropout=0.,merge_weights=False,fan_in_fan_out=False,
                            bias=old.bias is not None,device=old.weight.device,dtype=old.weight.dtype)
                        new.weight=old.weight;new.bias=old.bias
                        assert not new.weight.requires_grad and (new.bias is None or not new.bias.requires_grad)
                        setattr(parent,field,new)
                        targets.append(dict(path=prefix+'.'+field,kind=kind,rank=rank,alpha=alpha,
                            in_features=old.in_features,out_features=old.out_features,
                            trainable=rank*(old.in_features+old.out_features)))
    assert len(targets)==48
    return targets

def frozen_hash(model):
    # All original backbone parameters AND all backbone buffers, including late stages.
    h=hashlib.sha256()
    values=[(k,p) for k,p in model.encoder.named_parameters() if not p.requires_grad]
    values+=list(model.encoder.named_buffers())
    for k,v in values:
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
