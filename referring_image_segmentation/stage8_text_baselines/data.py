"""GPU-resident frozen pyramids with synchronized per-image color augmentation."""
from common import *
import numpy as np,torch
from torch.nn import functional as F

class RISData:
    def __init__(self):
        assert json.loads((ROOT/'data_manifest.json').read_text())['passed']
        self.images=json.loads((ASSETS/'images.json').read_text());self.mask_info=json.loads((ASSETS/'masks.json').read_text())
        self.rows={s:json.loads((ASSETS/f'{s}.json').read_text()) for s in ('train','val','testA','testB')}
        self.rgb=np.load(ASSETS/'images256.npy',mmap_mode='r');self.masks=np.load(ASSETS/'masks256.npy',mmap_mode='r')
        ti=json.loads((ASSETS/'text_index.json').read_text());text_lookup={sid:i for i,sid in enumerate(ti['sample_ids'])}
        self.text=torch.from_numpy(np.load(ASSETS/'text512.npy')).cuda();self.fixed_index=ti['fixed_index']
        self.info={}
        for s,rs in self.rows.items():
            self.info[s]={k:torch.tensor([r[k] for r in rs],device='cuda') for k in ('image_index','ann_index')}
            self.info[s]['text_index']=torch.tensor([text_lookup[r['sample_id']] for r in rs],device='cuda')
        self.mask_gpu=torch.from_numpy(np.asarray(self.masks).copy()).cuda()
        self.boxes=torch.tensor([[r['top'],r['left'],r['new_height'],r['new_width']] for r in self.images],device='cuda')
        yy=torch.arange(256,device='cuda')[None,:,None];xx=torch.arange(256,device='cuda')[None,None,:]
        self.yy,self.xx=yy,xx
        self.mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]

    def valid(self,indices):
        b=self.boxes[indices];return ((self.yy>=b[:,0,None,None])&(self.yy<(b[:,0]+b[:,2])[:,None,None])&
            (self.xx>=b[:,1,None,None])&(self.xx<(b[:,1]+b[:,3])[:,None,None])).float().unsqueeze(1)

class FeatureCache:
    SHAPES=((384,16,16),)*4+((96,64,64),(192,32,32),(384,16,16),(768,8,8))
    def __init__(self,encoder,data,splits=('train','val')):
        self.encoder,self.data=encoder,data
        ids=sorted({r['image_index'] for s in splits for r in data.rows[s]});self.ids=ids
        self.lookup=torch.full((len(data.images),),-1,device='cuda',dtype=torch.long)
        self.lookup[torch.tensor(ids,device='cuda')]=torch.arange(len(ids),device='cuda')
        self.features=[torch.empty((len(ids),*shape),device='cuda',dtype=torch.bfloat16) for shape in self.SHAPES]

    @torch.no_grad()
    def encode(self,split,epoch=0,rows=None):
        rows=self.data.rows[split] if rows is None else rows
        ids=sorted({r['image_index'] for r in rows})
        generator=torch.Generator().manual_seed(42+epoch*100003+17)
        colors=.9+.2*torch.rand((len(self.data.images),3),generator=generator)
        for start in range(0,len(ids),64):
            cpu_ids=ids[start:start+64];idx=torch.tensor(cpu_ids,device='cuda')
            x=torch.from_numpy(np.array(self.data.rgb[cpu_ids])).cuda().permute(0,3,1,2).float()/255.
            if split=='train':
                f=colors[cpu_ids].cuda();x=x*f[:,0,None,None,None]
                gray=x.mean(1,keepdim=True);x=gray+(x-gray)*f[:,2,None,None,None]
                mean=x.mean((1,2,3),keepdim=True);x=(mean+(x-mean)*f[:,1,None,None,None]).clamp(0,1)
                valid=self.data.valid(idx);x=x*valid+self.data.mean*(1-valid)
            with torch.autocast('cuda',dtype=torch.bfloat16):fs=self.encoder((x-self.data.mean)/self.data.std)
            dest=self.lookup[idx];assert (dest>=0).all()
            for a,b in zip(self.features,fs):a[dest]=b.to(a.dtype)

    def batch(self,split,positions):
        indices=self.data.info[split]['image_index'][positions]
        dest=self.lookup[indices];assert (dest>=0).all()
        features=[x[dest] for x in self.features]
        target=self.data.mask_gpu[self.data.info[split]['ann_index'][positions]].float().unsqueeze(1)
        text=self.data.text[self.data.info[split]['text_index'][positions]]
        return features,target,self.data.valid(indices),text
