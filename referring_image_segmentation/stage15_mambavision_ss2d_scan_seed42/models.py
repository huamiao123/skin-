"""SS2D-style routing ablation: official VMamba routes, shared NVIDIA SSM weights."""
from common import *
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from stage14_mambavision_s_isic2018_seed42.models import FrozenEncoder as OfficialEncoder
from stage3_confirmation.models import Probe,segmentation_loss,tensor_hash

VENDOR=PROJECT/'stage1_visual/vendor/VMamba'
sys.path.insert(0,str(VENDOR))
# Keep the exact official Mamba extension used by stage14, not the VMamba runtime.
sys.path.insert(0,str(ROOT/'runtime'))
import vmamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as nvidia_scan

class FourRouteMixer(nn.Module):
    """No initialized tensors: retain every original weight, add only 2D routing."""
    def __init__(self,original):
        super().__init__()
        for name in ('d_inner','dt_rank','d_state'):setattr(self,name,getattr(original,name))
        for name in ('in_proj','x_proj','dt_proj','out_proj','conv1d_x','conv1d_z'):
            setattr(self,name,getattr(original,name))
        self.A_log=original.A_log;self.D=original.D
        self.routes=4
        assert set(self.state_dict())==set(original.state_dict())
        assert all(torch.equal(v,original.state_dict()[k]) for k,v in self.state_dict().items())

    def scan_sequence(self,x):
        length=x.shape[-1]
        x_dbl=self.x_proj(rearrange(x,'b d l -> (b l) d'))
        dt,b,c=torch.split(x_dbl,[self.dt_rank,self.d_state,self.d_state],dim=-1)
        dt=rearrange(self.dt_proj(dt),'(b l) d -> b d l',l=length)
        b=rearrange(b,'(b l) dstate -> b dstate l',l=length).contiguous()
        c=rearrange(c,'(b l) dstate -> b dstate l',l=length).contiguous()
        return nvidia_scan(x,dt,-torch.exp(self.A_log.float()),b,c,self.D.float(),
            z=None,delta_bias=self.dt_proj.bias.float(),delta_softplus=True,return_last_state=None)

    def forward(self,hidden_states):
        batch,length,_=hidden_states.shape
        side=__import__('math').isqrt(length);assert side*side==length
        xz=rearrange(self.in_proj(hidden_states),'b l d -> b d l')
        x,z=xz.chunk(2,dim=1)
        x=F.silu(F.conv1d(x,self.conv1d_x.weight,self.conv1d_x.bias,padding='same',groups=self.d_inner//2))
        z=F.silu(F.conv1d(z,self.conv1d_z.weight,self.conv1d_z.bias,padding='same',groups=self.d_inner//2))
        if self.routes==1:  # preflight identity only; formal config always four.
            y=self.scan_sequence(x)
        else:
            assert self.routes==4
            # VMamba's unmodified operators: row, column, reversed row/column.
            routed=vmamba.cross_scan_fn(x.reshape(batch,-1,side,side),
                in_channel_first=True,out_channel_first=True,scans=0,force_torch=True)
            ys=self.scan_sequence(routed.flatten(0,1)).reshape(batch,4,-1,side,side)
            # Shared pretraining has one stream: average the four aligned streams,
            # not a fourfold gain increase. FP32 merge avoids BF16 reduction loss.
            y=vmamba.cross_merge_fn(ys.float(),in_channel_first=True,out_channel_first=True,
                scans=0,force_torch=True).mul(.25).to(x.dtype)
        return self.out_proj(rearrange(torch.cat((y,z),dim=1),'b d l -> b l d'))

def patch_mixers(encoder):
    replacements=[]
    for level_index,level in enumerate(encoder.net.model.levels):
        for block_index,block in enumerate(level.blocks):
            if hasattr(block,'mixer') and block.mixer.__class__.__name__=='MambaVisionMixer':
                original=block.mixer;modified=FourRouteMixer(original);block.mixer=modified
                replacements.append((level_index,block_index,original,modified))
    assert [(i,j) for i,j,_,_ in replacements]==[(2,j) for j in range(4)]+[(3,j) for j in range(3)]
    encoder.eval()
    return replacements

class FrozenEncoder(OfficialEncoder):
    def __init__(self,apply_four_routes=True):
        super().__init__()
        original_hash=tensor_hash(self)
        if apply_four_routes:
            replaced=patch_mixers(self)
            assert tensor_hash(self)==original_hash,'Routing must not add or alter any pretrained state'
            self.audit=dict(self.audit,executed_implementation='MambaVision-S weights with VMamba official cross-scan/merge; shared four-route SSM ablation',
                original_official_forward=False,full_vmamba_ss2d_module=False,
                vmamba_commit=VMAMBA_COMMIT,routing='window-local row/column/reversed row/reversed column',
                scan_merge_backend='official VMamba PyTorch CUDA operators, force_torch=True',
                direction_parameters='all four routes share original NVIDIA SSM weights',
                merge='aligned FP32 sum /4, cast back to input dtype',
                added_parameters=0,replaced_blocks=[dict(level=i,block=j) for i,j,_,_ in replaced],
                pretrained_state_unchanged_by_replacement=True)

def build_probe():return Probe(decoder='swin',seed=42,channels=CHANNELS)
