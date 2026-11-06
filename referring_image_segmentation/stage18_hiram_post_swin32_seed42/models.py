"""MambaVision-S baseline and HiRAM immediately after the 32x32 Swin level.

HiRAM and the vectorized B=1-equivalent routine are reused from stage16, which
contains the upstream source and CC BY-NC 4.0 license. The two input branches
here are post-Swin fused and same-encoder skip features, not SAM and CNN.
"""
from common import *
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from safetensors.torch import load_file
from stage3_confirmation.models import Probe, segmentation_loss, tensor_hash

# stage1 imports prepend their runtime; restore the NVIDIA selective-scan ABI
# before importing either the pinned encoder or the HiRAM SSM implementation.
sys.path.insert(0,str(STAGE14/'runtime'))
from stage16_hiram_dual_seed42.models import HiRAM, batched_hiram, region_loss_numerators


class FrozenEncoder(nn.Module):
    """Unmodified stage14 pinned pretrained NVIDIA encoder."""
    def __init__(self):
        super().__init__()
        directory=STAGE14/'official/hf'
        assert sha(directory/'model.safetensors')==WEIGHT_SHA256
        downloaded=json.loads((STAGE14/'official/source_download.json').read_text())
        assert downloaded['hf_revision']=='e4f9ea93877fba6f1e2d231fd0f0a391368ad895'
        for row in downloaded['files']:assert sha(directory/row['path'])==row['sha256']
        self.net, info=AutoModel.from_pretrained(str(directory),trust_remote_code=True,
            local_files_only=True,output_loading_info=True)
        assert all(not info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),info
        state=load_file(str(directory/'model.safetensors'))
        loaded=self.net.load_state_dict(state,strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        assert set(state)==set(self.net.state_dict())
        assert all(torch.equal(v,self.net.state_dict()[k]) for k,v in state.items())
        assert self.net.config.dim==96 and self.net.config.depths==[3,3,7,5]
        import selective_scan_cuda
        assert sha(selective_scan_cuda.__file__)=='0829e6a7d729adf47a5fdf24925238ea859f6395985a0db9fee8a7a8954ba929'
        self.requires_grad_(False)
        self.eval()
        self.audit=dict(weight_sha256=WEIGHT_SHA256,strict_load=True,
            hf_revision=downloaded['hf_revision'],weights_equal=True,
            scan_path=selective_scan_cuda.__file__)

    def train(self,mode=True):return super().train(False)

    @torch.no_grad()
    def forward(self,x):
        assert not self.training and not any(m.training for m in self.modules())
        pooled,features=self.net(x)
        assert len(features)==4 and list(pooled.shape)==[len(x),768]
        for f,c,s in zip(features,CHANNELS,(64,32,16,8)):
            assert list(f.shape)==[len(x),c,s,s]
        return [f.contiguous() for f in features]


class DecoderProbe(Probe):
    def __init__(self,mode):
        assert mode in ('paired_baseline','region_only','hiram')
        super().__init__('swin',42,channels=CHANNELS)
        self.mode=mode
        if mode!='paired_baseline':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(1042)
                upstream=HiRAM(64)
            if mode=='hiram':
                self.hiram=upstream
                for mug in (self.hiram.mug_sam,self.hiram.mug_cnn):
                    mug.ssm_full.conv.requires_grad_(False)
                    mug.ssm_full_bwd.conv.requires_grad_(False)
            else:
                self.route_coarse=upstream.mug_sam.prior_generate.route_proj
                self.route_skip=upstream.mug_cnn.prior_generate.route_proj

    def forward(self,features,return_regions=False):
        assert len(features)==4
        pyramid=[]
        for f,adapter,side in zip(features,self.adapter,(64,32,16,8)):
            a=adapter(f)
            if a.shape[-2:]!=(side,side):
                a=F.interpolate(a,(side,side),mode='bilinear',align_corners=False)
            pyramid.append(a)
        # Historical coarse-to-fine decoder topology; only the 32x32 add is
        # observed after Swin32, with the pretrained skip participating as before.
        x=self.decoder.blocks[3](pyramid[3])
        regions=[]
        for i in (2,1,0):
            coarse=F.interpolate(x,size=pyramid[i].shape[-2:],mode='bilinear',align_corners=False)
            skip=pyramid[i]
            x=self.decoder.blocks[i](coarse+skip)
            if i==1 and self.mode=='hiram':
                updated_x,updated_skip,regions=batched_hiram(self.hiram,x,skip,self.training)
                # HiRAM has a residual in each output; preserve the original
                # Swin result and add only the second branch's update.
                # When both updates vanish, the model is the exact baseline.
                x=updated_x+(updated_skip-skip)
            elif i==1 and self.mode=='region_only':
                regions=[F.log_softmax(self.route_coarse(x.permute(0,2,3,1)),-1).permute(0,3,1,2),
                         F.log_softmax(self.route_skip(skip.permute(0,2,3,1)),-1).permute(0,3,1,2)]
        logits=F.interpolate(self.decoder.readout(x),size=(256,256),mode='bilinear',align_corners=False)
        return (logits,regions) if return_regions else logits


@torch.no_grad()
def region_targets(mask):
    fg=(mask>=.5).float()
    counts=F.conv2d(fg,torch.ones((1,1,5,5),device=fg.device),padding=2)
    boundary=((counts>0)^(counts==25)).float()
    interior=fg*(1-boundary)
    return [F.interpolate(boundary,(32,32),mode='bilinear',align_corners=False),
            F.interpolate(interior,(32,32),mode='bilinear',align_corners=False)]
