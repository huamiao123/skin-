"""Frozen DINO cache, computed once per run; no spatial or intensity augmentation."""
from common import *
import numpy as np, torch

class MedicalData:
    def __init__(self):
        assert json.loads((ROOT/'data_manifest.json').read_text())['passed']
        self.rows={s:json.loads((ASSETS/f'{s}.json').read_text()) for s in SPLITS}
        self.allrows=json.loads((ASSETS/'all_rows.json').read_text())
        self.rgb=torch.from_numpy(np.load(ASSETS/'images256.npy')).cuda()
        self.masks=torch.from_numpy(np.load(ASSETS/'masks256.npy')).cuda()
        self.text=torch.from_numpy(np.load(ASSETS/'text512.npy')).cuda()
        self.fixed_index=len(self.allrows)
        self.indices={s:torch.tensor([r['index'] for r in rows],device='cuda') for s,rows in self.rows.items()}
        self.boxes=torch.tensor([[r[k] for k in ('top','left','new_height','new_width')] for r in self.allrows],device='cuda')
        self.mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        self.yy=torch.arange(256,device='cuda')[None,:,None]; self.xx=torch.arange(256,device='cuda')[None,None,:]

    def valid(self, indices):
        b=self.boxes[indices]
        return ((self.yy>=b[:,0,None,None])&(self.yy<(b[:,0]+b[:,2])[:,None,None])&
                (self.xx>=b[:,1,None,None])&(self.xx<(b[:,1]+b[:,3])[:,None,None])).float().unsqueeze(1)

    def inputs(self, indices):
        x=self.rgb[indices].permute(0,3,1,2).float()/255.; valid=self.valid(indices)
        return (x*valid+self.mean*(1-valid)-self.mean)/self.std

class FeatureCache:
    def __init__(self, encoder, data, splits=('train','val'), limits=None):
        self.encoder,self.data=encoder,data
        ids=sorted({r['index'] for s in splits for r in data.rows[s][:(limits or {}).get(s,None)]})
        self.lookup=torch.full((len(data.allrows),),-1,device='cuda',dtype=torch.long)
        self.lookup[torch.tensor(ids,device='cuda')]=torch.arange(len(ids),device='cuda')
        self.features=[torch.empty((len(ids),384,16,16),device='cuda',dtype=torch.bfloat16) for _ in range(4)]
        self.feature_bytes=sum(x.numel()*x.element_size() for x in self.features)

    @torch.no_grad()
    def encode(self, split, limit=None):
        ids=self.data.indices[split]
        if limit is not None: ids=ids[:limit]
        for start in range(0,len(ids),64):
            idx=ids[start:start+64]
            with torch.autocast('cuda',dtype=torch.bfloat16): fs=self.encoder(self.data.inputs(idx))
            dest=self.lookup[idx]; assert (dest>=0).all()
            for a,b in zip(self.features,fs): a[dest]=b.to(a.dtype)

    def batch(self, split, positions):
        idx=self.data.indices[split][positions]; dest=self.lookup[idx]; assert (dest>=0).all()
        return [f[dest] for f in self.features], self.data.masks[idx].float().unsqueeze(1), self.data.valid(idx), self.data.text[idx]
