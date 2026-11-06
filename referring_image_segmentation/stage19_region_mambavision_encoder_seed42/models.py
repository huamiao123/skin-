"""One window-local, pretrained-weight-sharing region-aware Mixer ablation."""
from common import *
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from stage3_confirmation.models import Probe, segmentation_loss, tensor_hash
from stage17_mambavision_decoder_hiram_seed42.models import FrozenEncoder as OfficialEncoder

# stage1/3 add their runtime to sys.path; the pinned NVIDIA encoder and scan
# must use the same official CUDA extension as the historical stage14 run.
sys.path.insert(0, str(STAGE14 / 'runtime'))
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class RegionScanMixer(nn.Module):
    """Preserve the pretrained projections and replace the scan sequence.

    The official level-2 input is a 16x16 map, padded to 28x28 and partitioned
    into four 14x14 windows per image. Each window is an independent sample.
    """
    def __init__(self, original, mode):
        super().__init__()
        assert mode in ('native', 'ordered_prior', 'region_scan')
        assert original.d_model == 384 and original.d_state == 8
        self.original = original
        self.mode = mode
        self.stochastic = False
        self.last_route = None
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(1042)
            self.router = nn.Sequential(nn.Linear(384,128), nn.GELU(), nn.Linear(128,3))
        self.region_prior = nn.Parameter(torch.zeros(3, original.d_state))
        mask = torch.zeros((1,1,28,28), dtype=torch.bool)
        mask[:,:,:16,:16] = True
        mask = mask.reshape(1,1,2,14,2,14).permute(0,2,4,3,5,1).reshape(4,196)
        self.register_buffer('window_valid', mask, persistent=False)

    @staticmethod
    def window_map(windows):
        assert windows.ndim == 3 and windows.shape[0] % 4 == 0 and windows.shape[1] == 196
        b = windows.shape[0] // 4
        c = windows.shape[2]
        return windows.reshape(b,2,2,14,14,c).permute(0,5,1,3,2,4).reshape(b,c,28,28)[:,:,:16,:16].contiguous()

    def _scanned(self, x, dt, b_coeff, c_coeff, mask, priority):
        """Scan each window's selected items; padded suffix cannot affect valid prefix."""
        count = mask.sum(dim=1)
        max_len = int(count.max().item())
        if max_len == 0:
            return None
        order = torch.argsort(priority.masked_fill(~mask, -torch.inf), dim=1,
                              descending=True, stable=True)[:, :max_len]
        selected = mask.gather(1, order)
        def take(v):
            return v.gather(2, order[:,None,:].expand(-1,v.shape[1],-1)).contiguous()
        mixer = self.original
        y = selective_scan_fn(take(x), take(dt), -torch.exp(mixer.A_log.float()),
              take(b_coeff), take(c_coeff), mixer.D.float(), z=None,
              delta_bias=mixer.dt_proj.bias.float(), delta_softplus=True,
              return_last_state=None)
        # torch.where prevents a NaN in an unused padded position propagating.
        y = torch.where(selected[:,None,:], y, torch.zeros_like(y))
        return order, y

    def forward(self, hidden_states):
        if self.mode == 'native':
            self.last_route = None
            return self.original(hidden_states)
        bw, length, channels = hidden_states.shape
        assert length == 196 and channels == 384 and bw % 4 == 0
        valid = self.window_valid.repeat(bw//4,1)
        logits = self.router(hidden_states)
        self.last_route = self.window_map(logits)
        with torch.autocast('cuda', enabled=False):
            probabilities = logits.float().softmax(dim=-1)
            prior = probabilities @ self.region_prior.float()
            prior = prior * valid[:,:,None]
            if self.mode == 'region_scan':
                if self.stochastic:
                    chosen = F.gumbel_softmax(logits.float(), hard=True, dim=-1).detach().argmax(-1)
                else:
                    chosen = logits.detach().argmax(-1)
        m = self.original
        xz = rearrange(m.in_proj(hidden_states), 'b l d -> b d l')
        x,z = xz.chunk(2,dim=1)
        x = F.silu(F.conv1d(x,m.conv1d_x.weight,m.conv1d_x.bias,padding='same',groups=m.d_inner//2))
        z = F.silu(F.conv1d(z,m.conv1d_z.weight,m.conv1d_z.bias,padding='same',groups=m.d_inner//2))
        projected = m.x_proj(rearrange(x,'b d l -> (b l) d'))
        dt,b_coeff,c_coeff = torch.split(projected,[m.dt_rank,m.d_state,m.d_state],dim=-1)
        dt = rearrange(m.dt_proj(dt),'(b l) d -> b d l',l=length)
        b_coeff = rearrange(b_coeff,'(b l) d -> b d l',l=length).contiguous()
        c_coeff = rearrange(c_coeff,'(b l) d -> b d l',l=length).contiguous()
        c_coeff = c_coeff + prior.transpose(1,2).to(c_coeff.dtype)
        row_order = torch.arange(length,0,-1,device=x.device,dtype=torch.float32)[None,:].expand(bw,-1)
        if self.mode == 'ordered_prior':
            groups=[(valid,row_order),(~valid,row_order)]
        else:
            groups=[(valid & (chosen==k), probabilities[:,:,k].detach()) for k in range(3)]
            groups.append((~valid,row_order))
        merged = torch.zeros_like(x)
        for mask,priority in groups:
            result = self._scanned(x,dt,b_coeff,c_coeff,mask,priority)
            if result is not None:
                idx,part = result
                merged = merged.scatter_add(2,idx[:,None,:].expand(-1,x.shape[1],-1),part)
        return m.out_proj(rearrange(torch.cat((merged,z),dim=1),'b d l -> b l d'))


class RegionEncoder(nn.Module):
    def __init__(self, mode):
        super().__init__()
        source = OfficialEncoder()
        expected = json.loads((STAGE14/'config.json').read_text())['encoder_initial_hash']
        assert tensor_hash(source) == expected
        self.audit = dict(source.audit, pretrained_encoder_initial_hash=expected)
        self.net = source.net
        self.net.requires_grad_(False)
        original = self.net.model.levels[2].blocks[0].mixer
        assert original.__class__.__name__ == 'MambaVisionMixer'
        self.net.model.levels[2].blocks[0].mixer = RegionScanMixer(original,mode)
        self.train(False)
        assert self.frozen_state_hash() == expected

    @property
    def mixer(self):
        return self.net.model.levels[2].blocks[0].mixer

    def train(self, mode=True):
        # Preserve BatchNorm/DropPath in pretrained eval mode; the new head has
        # no dropout/BN, and the hard routing mode is controlled separately.
        super().train(False)
        self.mixer.stochastic = bool(mode)
        return self

    def frozen_state_hash(self):
        h = hashlib.sha256()
        prefix = 'net.model.levels.2.blocks.0.mixer.'
        for key,value in self.state_dict().items():
            if key.startswith(prefix+'router.') or key == prefix+'region_prior':
                continue
            name = key.replace(prefix+'original.',prefix,1)
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def forward(self, images):
        assert not self.net.training and not any(m.training for m in self.net.modules())
        pooled, features = self.net(images)
        assert len(features) == 4 and list(pooled.shape) == [len(images),768]
        for f,c,s in zip(features,CHANNELS,(64,32,16,8)):
            assert list(f.shape) == [len(images),c,s,s]
        route = self.mixer.last_route
        self.mixer.last_route = None
        return [f.contiguous() for f in features],route


class Segmentor(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.encoder = RegionEncoder(mode)
        self.probe = Probe('swin',42,channels=CHANNELS)

    def forward(self, images):
        features, route = self.encoder(images)
        return self.probe(features), route


@torch.no_grad()
def region_targets(mask):
    fg = (mask >= .5).float()
    counts = F.conv2d(fg,torch.ones((1,1,5,5),device=fg.device),padding=2)
    boundary = ((counts>0) ^ (counts==25)).float()
    inside = fg*(1-boundary)
    background = 1-boundary-inside
    return F.avg_pool2d(torch.cat((background,boundary,inside),dim=1),16)


def region_numerators(logits, targets):
    assert logits.shape == targets.shape and logits.shape[1:] == (3,16,16)
    return -(targets.float() * F.log_softmax(logits.float(),dim=1)).sum((0,2,3))
