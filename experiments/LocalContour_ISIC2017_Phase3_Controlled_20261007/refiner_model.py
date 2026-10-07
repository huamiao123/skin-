"""Same 2-layer node Transformer for circular local, medium and global attention."""
from __future__ import annotations

import torch
from torch import nn

def normalize_candidate_scores(local_scores,valid):
    """Shared per-node normalization; never mixes contour nodes."""
    valid=valid.bool()
    scores=local_scores.float().masked_fill(~valid,0)
    count=valid.sum(dim=-1,keepdim=True).clamp_min(1)
    mean=scores.sum(dim=-1,keepdim=True)/count
    centred=(scores-mean).masked_fill(~valid,0)
    scale=((centred.square().sum(dim=-1,keepdim=True)/count).sqrt()+1).clamp_min(1)
    return (centred/scale).clamp(-5,5)

class PointwiseBlock(nn.Module):
    def __init__(self,dim=128):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.fc1=nn.Linear(dim,dim*4)
        self.activation=nn.GELU()
        self.dropout1=nn.Dropout(0.1)
        self.fc2=nn.Linear(dim*4,dim)
        self.dropout2=nn.Dropout(0.1)

    def forward(self,x):
        return x+self.dropout2(self.fc2(self.dropout1(self.activation(self.fc1(self.norm(x))))))

class PointwiseRefiner(nn.Module):
    """Capacity-matched N0: every output node sees only its own 109 inputs."""
    def __init__(self,dim=128):
        super().__init__()
        self.input=nn.Linear(44+65,dim)
        self.blocks=nn.Sequential(PointwiseBlock(dim),PointwiseBlock(dim))
        self.head=nn.Linear(dim,65)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,context,local_scores,valid):
        normalized=normalize_candidate_scores(local_scores,valid)
        hidden=self.blocks(self.input(torch.cat((context.float(),normalized),dim=-1)))
        return normalized+self.head(hidden)

class NodeRefiner(nn.Module):
    def __init__(self,attention_range: int | None,dim=128,layers=2,heads=4):
        super().__init__()
        if attention_range not in (8,32,None):raise ValueError('Use +/-8, +/-32, or global')
        self.attention_range=attention_range
        self.input=nn.Linear(44+65,dim)
        layer=nn.TransformerEncoderLayer(d_model=dim,nhead=heads,dim_feedforward=dim*2,
                                         dropout=0.1,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=layers,enable_nested_tensor=False)
        self.head=nn.Linear(dim,65)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        if attention_range is None:self.register_buffer('attention_mask',None,persistent=False)
        else:
            pos=torch.arange(256)
            delta=(pos[:,None]-pos[None,:]).abs()
            circular=torch.minimum(delta,256-delta)
            self.register_buffer('attention_mask',circular>attention_range,persistent=False)

    def represent(self,context,local_scores,valid):
        normalized=normalize_candidate_scores(local_scores,valid)
        tokens=self.input(torch.cat((context.float(),normalized),dim=-1))
        hidden=self.encoder(tokens,mask=self.attention_mask)
        return normalized,hidden

    def forward(self,context,local_scores,valid):
        normalized,hidden=self.represent(context,local_scores,valid)
        return normalized+self.head(hidden)

class GatedNodeRefiner(NodeRefiner):
    def __init__(self,attention_range: int | None,dim=128,layers=2,heads=4):
        super().__init__(attention_range,dim,layers,heads)
        self.gate=nn.Linear(dim,1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self,context,local_scores,valid):
        normalized,hidden=self.represent(context,local_scores,valid)
        return normalized+self.head(hidden),self.gate(hidden).squeeze(-1)

def parameter_count(model):return sum(p.numel() for p in model.parameters())
