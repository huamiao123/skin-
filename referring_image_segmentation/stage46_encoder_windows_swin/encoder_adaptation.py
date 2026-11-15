"""Pinned Microsoft Swin blocks and explicit spatial orchestration inside MambaVision."""
from pathlib import Path
import ast,hashlib,inspect,sys
import torch
from torch import nn
from timm.models.layers import DropPath,to_2tuple,trunc_normal_
ROOT=Path(__file__).resolve().parent
RESEARCH=ROOT.parent.parent.parent/'research/swin_encoder_author_20261112'
AUTHOR=RESEARCH/'official_code/microsoft__Swin-Transformer/models/swin_transformer.py'
def native_scope():
    wanted={'Mlp','window_partition','window_reverse','WindowAttention','SwinTransformerBlock'}
    nodes=[n for n in ast.parse(AUTHOR.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
    assert {n.name for n in nodes}==wanted
    scope=dict(__name__='isolated_swin_source',torch=torch,nn=nn,DropPath=DropPath,to_2tuple=to_2tuple,
        trunc_normal_=trunc_normal_,WindowProcess=None,WindowProcessReverse=None)
    module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(module,str(AUTHOR),'exec'),scope)
    provenance={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in nodes}
    return scope,provenance

class HybridSwinLevel(nn.Module):
    def __init__(self,original,hw,scope):
        super().__init__()
        assert not original.conv
        self.conv=False;self.transformer_block=True;self.window_size=8;self.do_gt=False
        self.hw=hw;self.blocks=original.blocks;self.downsample=original.downsample
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
            shift=4 if hw>8 and j%2 else 0
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(42)
                new=scope['SwinTransformerBlock'](dim=dim,input_resolution=(hw,hw),
                    num_heads=attn.num_heads,window_size=8,shift_size=shift,
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
            for k,v in src.items():assert torch.equal(v,new.state_dict()[k]),k
            self.blocks[i]=new
            self.transfer_records.append(dict(block=i,shift=shift,transferred_keys=sorted(src),new_keys=sorted(set(dest)-set(src))))
    def forward(self,x):
        b,c,h,w=x.shape
        assert h==w==self.hw and h%8==0
        tokens=self._partition(x.permute(0,2,3,1).contiguous(),8).view(-1,64,c)
        for block in self.blocks[:self.mamba_count]:tokens=block(tokens)
        x=self._reverse(tokens.view(-1,8,8,c),8,h,w).view(b,h*w,c)
        for block in self.blocks[self.mamba_count:]:x=block(x)
        x=x.view(b,h,w,c).permute(0,3,1,2).contiguous()
        return (self.downsample(x),x) if self.downsample is not None else (x,x)

def adapt_encoder(encoder,arm):
    assert arm in ('original','window8','swin8')
    records={}
    if arm=='original':return records
    for level in encoder.model.levels[2:]:level.window_size=8
    if arm=='swin8':
        scope,nodes=native_scope()
        for i,hw in [(2,16),(3,8)]:
            encoder.model.levels[i]=HybridSwinLevel(encoder.model.levels[i],hw,scope)
            records[str(i)]=encoder.model.levels[i].transfer_records
        records['author_ast']=nodes
    return records

def canonical_encoder_state(model):
    mapped={}
    for key,value in model.encoder.state_dict().items():
        if key.endswith(('relative_position_bias_table','relative_position_index','attn_mask')):continue
        mapped[key.replace('.attn.','.mixer.')]=value
    return mapped

def assert_initial_transfer(model,reference):
    a=canonical_encoder_state(model);b=reference.encoder.state_dict()
    assert a.keys()==b.keys(),(a.keys()-b.keys(),b.keys()-a.keys())
    for k,v in a.items():assert torch.equal(v.cpu(),b[k].cpu()),k
    extras={k:v for k,v in model.encoder.named_parameters() if k.endswith('relative_position_bias_table')}
    assert len(extras)==(5 if model.arm=='swin8' else 0)
    assert all(torch.count_nonzero(v)==0 for v in extras.values())
