"""Unidirectional DINO-query/VMamba-key-value window cross-attention."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT))
from stage4_fusion.models import EncoderPair, CHANNELS, FusionProbe
from stage3_confirmation.models import Probe as SingleProbe
from stage1_visual.models import tensor_hash, segmentation_loss

NAMES=('local_conv','window_cross_attention')
SPEC=dict(channels=64,resolutions=[64,32,16,8],decoder='unchanged stage2 SwinPixelDecoder',
    direction='Q from DINO; K and V from VMamba',heads=4,head_dim=16,window_size=8,
    cross_blocks_per_scale=1,shift_size=0,relative_position_bias='learned 225x4; trunc_normal std .02',
    qkv_bias=True,pre_norm='separate LayerNorm(64) for each branch',residual='D + 0.1 * message; scale fixed',
    dropout=0.,drop_path=0.,additional_ffn=False,
    local_control='concat normalized branches -> 1x1 128-to-72 -> GELU -> depthwise7x7 -> GELU -> 1x1 72-to-64',
    local_control_note='Approximate trainable-parameter matching; not identical receptive field or FLOPs',
    initialization='seed42 adapters/decoder match stage4; messages random, not function-matched; no task warm start')


def partition(x,window=8):
    b,h,w,c=x.shape
    assert h%window==0 and w%window==0
    return x.reshape(b,h//window,window,w//window,window,c).permute(0,1,3,2,4,5).reshape(-1,window*window,c)


def reverse(windows,b,h,w,window=8):
    c=windows.shape[-1]
    return windows.reshape(b,h//window,w//window,window,window,c).permute(0,1,3,2,4,5).reshape(b,h,w,c)


class WindowCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_d=nn.LayerNorm(64);self.norm_m=nn.LayerNorm(64)
        self.q=nn.Linear(64,64,bias=True);self.kv=nn.Linear(64,128,bias=True)
        self.proj=nn.Linear(64,64,bias=True)
        self.relative_position_bias_table=nn.Parameter(torch.zeros(225,4))
        nn.init.trunc_normal_(self.relative_position_bias_table,std=.02)
        y,x=torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
        coords=torch.stack((y,x)).flatten(1)
        relative=(coords[:,:,None]-coords[:,None,:]).permute(1,2,0).contiguous()+7
        index=relative[:,:,0]*15+relative[:,:,1]
        self.register_buffer('relative_position_index',index)
        self.mode='learned'

    def attention(self,d,m,manual=False):
        count,tokens,channels=d.shape
        q=self.q(d).reshape(count,tokens,4,16).permute(0,2,1,3)
        kv=self.kv(m).reshape(count,tokens,2,4,16).permute(2,0,3,1,4)
        k,v=kv.unbind(0)
        bias=self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(64,64,4).permute(2,0,1)
        bias=bias.to(q.dtype).unsqueeze(0)
        if self.mode=='uniform':
            out=v.mean(-2,keepdim=True).expand_as(v)
        elif self.mode=='same_position':
            out=v
        elif self.mode=='learned':
            if manual:
                weights=((q@k.transpose(-2,-1))*.25+bias).softmax(-1)
                out=weights@v
            else:
                out=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.,is_causal=False)
        else:raise ValueError(self.mode)
        return self.proj(out.transpose(1,2).reshape(count,tokens,channels))

    def forward(self,d,m):
        b,c,h,w=d.shape
        d=partition(self.norm_d(d.permute(0,2,3,1)))
        m=partition(self.norm_m(m.permute(0,2,3,1)))
        return reverse(self.attention(d,m),b,h,w).permute(0,3,1,2).contiguous()


class LocalConvInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_d=nn.LayerNorm(64);self.norm_m=nn.LayerNorm(64)
        self.message=nn.Sequential(nn.Conv2d(128,72,1),nn.GELU(),
            nn.Conv2d(72,72,7,padding=3,groups=72),nn.GELU(),nn.Conv2d(72,64,1))

    def forward(self,d,m):
        d=self.norm_d(d.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
        m=self.norm_m(m.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
        return self.message(torch.cat((d,m),dim=1))


class InteractionProbe(nn.Module):
    def __init__(self,seed,kind):
        super().__init__()
        assert kind in NAMES
        d=SingleProbe('swin',seed,CHANNELS['dinov3']);m=SingleProbe('swin',seed,CHANNELS['vmamba'])
        self.dino_adapter=d.adapter;self.mamba_adapter=m.adapter;self.decoder=d.decoder
        self.kind=kind
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            cls=WindowCrossAttention if kind=='window_cross_attention' else LocalConvInteraction
            self.interaction=nn.ModuleList([cls() for _ in range(4)])

    def forward(self,features):
        assert len(features)==8
        pyramid=[]
        for i,side in enumerate((64,32,16,8)):
            d=F.interpolate(self.dino_adapter[i](features[i]),(side,side),mode='bilinear',align_corners=False)
            m=F.interpolate(self.mamba_adapter[i](features[i+4]),(side,side),mode='bilinear',align_corners=False)
            pyramid.append(d+.1*self.interaction[i](d,m))
        return self.decoder(pyramid)


def build_probes(seed):return {n:InteractionProbe(seed,n) for n in NAMES}


def select_features(name,features):
    assert name in NAMES
    return features
