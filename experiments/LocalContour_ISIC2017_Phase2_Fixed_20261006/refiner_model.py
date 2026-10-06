"""Same 2-layer node Transformer for circular local, medium and global attention."""
from __future__ import annotations

import torch
from torch import nn

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
        valid=valid.bool()
        scores=local_scores.float().masked_fill(~valid,0)
        count=valid.sum(dim=-1,keepdim=True).clamp_min(1)
        mean=scores.sum(dim=-1,keepdim=True)/count
        centred=(scores-mean).masked_fill(~valid,0)
        scale=((centred.square().sum(dim=-1,keepdim=True)/count).sqrt()+1).clamp_min(1)
        normalized=(centred/scale).clamp(-5,5)
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
