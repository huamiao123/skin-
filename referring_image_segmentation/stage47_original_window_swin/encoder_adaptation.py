"""Native Swin attention interventions with unchanged MambaVision window/padding paths."""
from pathlib import Path
import torch
from torch import nn
from stage46_encoder_windows_swin.encoder_adaptation import (
    native_scope,canonical_encoder_state,RESEARCH,AUTHOR,DropPath,to_2tuple,trunc_normal_)
class OriginalWindowSwinLevel(nn.Module):
    def __init__(self,original,hw,scope,arm):
        super().__init__()
        assert not original.conv
        self.conv=False;self.transformer_block=True;self.window_size=original.window_size;self.do_gt=False
        self.hw=hw;self.blocks=original.blocks;self.downsample=original.downsample
        self.padded_hw=((hw+self.window_size-1)//self.window_size)*self.window_size
        self.mamba_count=sum(type(b.mixer).__name__=='MambaVisionMixer' for b in self.blocks)
        assert all(type(b.mixer).__name__=='MambaVisionMixer' for b in self.blocks[:self.mamba_count])
        self._partition=scope['window_partition'];self._reverse=scope['window_reverse']
        self.transfer_records=[]
        for j,i in enumerate(range(self.mamba_count,len(self.blocks))):
            old=self.blocks[i];attn=old.mixer
            assert type(attn).__name__=='Attention'
            assert old.gamma_1==1 and old.gamma_2==1
            assert attn.attn_drop.p==attn.proj_drop.p==0
            assert isinstance(attn.q_norm,nn.Identity) and isinstance(attn.k_norm,nn.Identity)
            dim=old.norm1.normalized_shape[0]
            assert old.norm1.eps==old.norm2.eps==1e-5
            shift=self.window_size//2 if arm=='shifted_windows' and j%2 else 0
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(42)
                new=scope['SwinTransformerBlock'](dim=dim,input_resolution=(self.padded_hw,self.padded_hw),
                    num_heads=attn.num_heads,window_size=self.window_size,shift_size=shift,
                    mlp_ratio=old.mlp.fc1.out_features/dim,qkv_bias=attn.qkv.bias is not None,
                    drop=0.,attn_drop=0.,drop_path=getattr(old.drop_path,'drop_prob',0.),
                    norm_layer=nn.LayerNorm,fused_window_process=False)
            src={k.replace('mixer.','attn.',1) if k.startswith('mixer.') else k:v for k,v in old.state_dict().items()}
            dest=new.state_dict()
            assert set(src)<=set(dest)
            for k,v in src.items():assert v.shape==dest[k].shape,(k,v.shape,dest[k].shape)
            info=new.load_state_dict(src,strict=False)
            allowed={'attn.relative_position_bias_table','attn.relative_position_index','attn_mask'}
            assert not info.unexpected_keys and set(info.missing_keys)<=allowed,info
            nn.init.zeros_(new.attn.relative_position_bias_table)
            new.attn.relative_position_bias_table.requires_grad_(arm!='native_fixed')
            for k,v in src.items():assert torch.equal(v,new.state_dict()[k]),k
            self.blocks[i]=new
            self.transfer_records.append(dict(block=i,shift=shift,transferred_keys=sorted(src),new_keys=sorted(set(dest)-set(src))))
    def forward(self,x):
        b,c,h,w=x.shape
        assert h==w==self.hw
        ws=self.window_size
        pad_r=(ws-w%ws)%ws;pad_b=(ws-h%ws)%ws
        if pad_r or pad_b:x=torch.nn.functional.pad(x,(0,pad_r,0,pad_b))
        hp,wp=x.shape[-2:]
        tokens=self._partition(x.permute(0,2,3,1).contiguous(),ws).view(-1,ws*ws,c)
        for block in self.blocks[:self.mamba_count]:tokens=block(tokens)
        # Preserve padded features already evolved by Mamba; do not crop or zero them.
        x=self._reverse(tokens.view(-1,ws,ws,c),ws,hp,wp).view(b,hp*wp,c)
        for block in self.blocks[self.mamba_count:]:x=block(x)
        x=x.view(b,hp,wp,c).permute(0,3,1,2).contiguous()
        if pad_r or pad_b:x=x[:,:,:h,:w].contiguous()
        return (self.downsample(x),x) if self.downsample is not None else (x,x)

def adapt_encoder(encoder,arm):
    assert arm in ('original','native_fixed','position_bias','shifted_windows')
    if arm=='original':return {}
    scope,nodes=native_scope();records={}
    for i,hw,ws in [(2,16,14),(3,8,7)]:
        assert encoder.model.levels[i].window_size==ws
        encoder.model.levels[i]=OriginalWindowSwinLevel(encoder.model.levels[i],hw,scope,arm)
        records[str(i)]=encoder.model.levels[i].transfer_records
    records['author_ast']=nodes
    return records

def assert_initial_transfer(model,reference):
    a=canonical_encoder_state(model);b=reference.encoder.state_dict()
    assert a.keys()==b.keys(),(a.keys()-b.keys(),b.keys()-a.keys())
    for k,v in a.items():assert torch.equal(v.cpu(),b[k].cpu()),k
    extras={k:v for k,v in model.encoder.named_parameters() if k.endswith('relative_position_bias_table')}
    assert len(extras)==(0 if model.arm=='original' else 5)
    assert all(torch.count_nonzero(v)==0 for v in extras.values())
    assert all(v.requires_grad==(model.arm!='native_fixed') for v in extras.values())
