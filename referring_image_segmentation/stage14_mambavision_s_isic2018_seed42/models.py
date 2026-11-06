"""Official NVIDIA encoder and unmodified historical segmentation probe."""
from common import *
from torch import nn
from transformers import AutoModel
from safetensors.torch import load_file
from stage3_confirmation.models import Probe, segmentation_loss, tensor_hash
# Historical model imports prepend their own runtime. Restore priority before
# loading NVIDIA's remote module so its scan comes from the official Mamba wheel.
sys.path.insert(0,str(ROOT/'runtime'))

class FrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        directory=ROOT/'official/hf'
        assert sha(directory/'model.safetensors')==WEIGHT_SHA256
        downloaded=json.loads((ROOT/'official/source_download.json').read_text())
        assert downloaded['hf_revision']==HF_REVISION
        for row in downloaded['files']:assert sha(directory/row['path'])==row['sha256']
        self.net,info=AutoModel.from_pretrained(str(directory),trust_remote_code=True,
            local_files_only=True,output_loading_info=True)
        import selective_scan_cuda
        assert Path(selective_scan_cuda.__file__).parent==ROOT/'runtime','Unexpected historical scan extension shadowing the official wheel'
        assert sha(selective_scan_cuda.__file__)=='0829e6a7d729adf47a5fdf24925238ea859f6395985a0db9fee8a7a8954ba929'
        assert all(not info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),info
        # Verify every parameter AND buffer against the pinned safetensors object.
        state=load_file(str(directory/'model.safetensors'))
        result=self.net.load_state_dict(state,strict=True)
        assert set(state)==set(self.net.state_dict())
        assert all(torch.equal(v,self.net.state_dict()[k]) for k,v in state.items())
        config=self.net.config
        assert config.dim==96 and config.depths==[3,3,7,5]
        self.channels=CHANNELS
        self.requires_grad_(False);self.eval()
        self.audit=dict(model='nvidia/MambaVision-S-1K',hf_revision=HF_REVISION,
            official_repository='https://github.com/NVlabs/MambaVision',official_repository_revision=GITHUB_REVISION,
            executed_implementation='Unmodified pinned HF modeling_mambavision.py via AutoModel',
            weight_path=str(directory/'model.safetensors'),weight_sha256=WEIGHT_SHA256,
            strict_load=True,loading_info=info,missing_keys=list(result.missing_keys),unexpected_keys=list(result.unexpected_keys),
            all_parameters_and_buffers_equal_pretrained=True,channels=list(CHANNELS),
            params=sum(p.numel() for p in self.parameters()),
            classifier_params=sum(p.numel() for p in self.net.model.head.parameters()),
            classifier_note='Pretrained classification head retained, frozen and counted; not executed by feature extraction',
            batchnorm_modules=sum(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in self.modules()))

    def train(self,mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self,x):
        assert not self.training and not any(m.training for m in self.modules())
        pooled,features=self.net(x)
        assert len(features)==4 and list(pooled.shape)==[len(x),768]
        for f,c,s in zip(features,CHANNELS,(64,32,16,8)):
            assert list(f.shape)==[len(x),c,s,s],list(f.shape)
        return [f.contiguous() for f in features]

def build_probe():
    return Probe(decoder='swin',seed=SEED,channels=CHANNELS)
