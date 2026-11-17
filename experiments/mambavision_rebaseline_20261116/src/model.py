"""Native MMSegmentation heads and the project's frozen-prefix MambaVision adapter.

No CUDA work occurs on import or construction. The caller chooses the device.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import torch
from torch import nn
from torch.nn import functional as F

PROJECT = Path('/home/waas/mambavision_stage22/source/referring_image_segmentation')
RESEARCH = PROJECT.parent.parent / 'research/upernet_author_20261112'
EXPERIMENT = Path(__file__).resolve().parents[1]
FCN_SHA256 = 'c6fc931ac546d5d1f19bd2bf84679aa42be3245412b4d4c02e89bf2b16ba473a'
HF = PROJECT / 'stage14_mambavision_s_isic2018_seed42/official/hf'
WEIGHT_SHA256 = '006af39903f2d88c5b455431d087d0fcbf5d5932114aef52542c0d2f0b1ec180'
CHANNELS = (96, 192, 384, 768)
ARMS = {'A-64-noaux': (64, False), 'B-512-noaux': (512, False), 'C-512-aux': (512, True)}

def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def native_classes():
    """Extract actual pinned author ASTs, validating every source against its manifest."""
    manifest = json.loads((RESEARCH / 'SOURCE_MANIFEST.json').read_text())
    record = next(r for r in manifest['repositories'] if r['repository'] == 'open-mmlab/mmsegmentation')
    if record['commit'] != 'c685fe6767c4cadf6b051983ca6208f1b9d1ccb8':
        raise RuntimeError('unexpected MMSegmentation revision')
    base = RESEARCH / 'official_code/open-mmlab__mmsegmentation'
    for entry in record['files']:
        path = base / entry['path']
        if not path.is_file() or sha256(path) != entry['sha256']:
            raise RuntimeError(f'upstream source hash mismatch: {path}')
    fcn = EXPERIMENT / 'vendor/fcn_head.py'
    if sha256(fcn) != FCN_SHA256:
        raise RuntimeError('FCNHead source hash mismatch')
    spec = importlib.util.spec_from_file_location('stage45_native_author', PROJECT / 'stage45_upernet_decoder/author_modules.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scope, provenance = mod.native_scope()
    provenance['decode_heads/fcn_head.py'] = {'sha256': FCN_SHA256, 'nodes': mod.extract(fcn, {'FCNHead'}, scope)}
    return scope['UPerHead'], scope['FCNHead'], provenance

class NativeHeads(nn.Module):
    def __init__(self, arm, seed=42):
        super().__init__()
        if arm not in ARMS: raise ValueError(f'Unknown arm: {arm}')
        width, aux = ARMS[arm]
        UPerHead, FCNHead, self.provenance = native_classes()
        common = dict(dropout_ratio=.1, num_classes=2, out_channels=1,
                      threshold=.5, norm_cfg=dict(type='SyncBN', requires_grad=True),
                      align_corners=False,
                      loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.decode_head = UPerHead(in_channels=list(CHANNELS), in_index=[0,1,2,3],
                                        channels=width, pool_scales=(1,2,3,6), **common)
            self.decode_head.init_weights()
            self.auxiliary_head = FCNHead(in_channels=384, in_index=2, channels=256,
                                          num_convs=1, concat_input=False, **common) if aux else None
            if self.auxiliary_head is not None: self.auxiliary_head.init_weights()
        self.arm = arm

    def forward(self, features, output_size):
        if len(features) != 4: raise ValueError('Expected four encoder stages')
        if len(output_size)!=2 or min(output_size)<=0: raise ValueError('invalid output size')
        for x, c, stride in zip(features, CHANNELS, (4, 8, 16, 32)):
            expected = tuple((int(n) + stride - 1)//stride for n in output_size)
            if x.ndim!=4 or x.shape[1]!=c or tuple(x.shape[-2:])!=expected:
                raise ValueError(f'encoder feature mismatch: expected channels={c} spatial={expected}, got {tuple(x.shape)}')
        if len({x.shape[0] for x in features})!=1: raise ValueError('batch mismatch')
        out = {'main': F.interpolate(self.decode_head(features), size=output_size, mode='bilinear', align_corners=False)}
        if self.training and self.auxiliary_head is not None:
            out['aux'] = F.interpolate(self.auxiliary_head(features), size=output_size, mode='bilinear', align_corners=False)
        return out

class SegmentationSystem(nn.Module):
    def __init__(self, encoder, arm, seed=42):
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        for level in self.encoder.model.levels[2:4]: level.requires_grad_(True)
        self.probe = NativeHeads(arm, seed)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def prefix(self, images):
        base = self.encoder.model
        x = base.patch_embed(images)
        x, f1 = base.levels[0](x)
        x, f2 = base.levels[1](x)
        return f1.contiguous(), f2.contiguous(), x.contiguous()

    def suffix(self, mid):
        x, f3 = self.encoder.model.levels[2](mid)
        _, f4 = self.encoder.model.levels[3](x)
        return f3.contiguous(), f4.contiguous()

    def from_cache(self, f1, f2, mid, output_size):
        f3, f4 = self.suffix(mid)
        return self.probe([f1, f2, f3, f4], output_size)

    def forward(self, images):
        f1, f2, mid = self.prefix(images)
        return self.from_cache(f1, f2, mid, images.shape[-2:])

def load_pretrained_encoder():
    """Load the exact Stage45 backbone and assert byte/hash/key equality on CPU."""
    from transformers import AutoModel
    from safetensors.torch import load_file
    if sha256(HF / 'model.safetensors') != WEIGHT_SHA256: raise RuntimeError('pretrained weights hash mismatch')
    for entry in json.loads((HF.parent / 'source_download.json').read_text())['files']:
        path = HF / entry['path']
        if not path.is_file() or sha256(path) != entry['sha256']: raise RuntimeError(f'backbone source hash mismatch: {path}')
    encoder, info = AutoModel.from_pretrained(str(HF), trust_remote_code=True, local_files_only=True, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')): raise RuntimeError(str(info))
    weights = load_file(str(HF / 'model.safetensors'), device='cpu')
    encoder.load_state_dict(weights, strict=True)
    if any(not torch.equal(v, encoder.state_dict()[k]) for k,v in weights.items()): raise RuntimeError('weight tensor mismatch')
    return encoder

def segmentation_loss(logits, targets):
    """Same FP32 BCE + per-image soft Dice as stage1_visual.models."""
    logits, targets = logits.float(), targets.float()
    if logits.shape != targets.shape: raise ValueError('logit/target shape mismatch')
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    pred, target = logits.sigmoid().flatten(1), targets.flatten(1)
    dice = (2*(pred*target).sum(1)+1)/(pred.sum(1)+target.sum(1)+1)
    return bce+1-dice.mean()

def objective(output, target, aux_weight=.4):
    main = segmentation_loss(output['main'], target)
    aux = segmentation_loss(output['aux'], target) if 'aux' in output else None
    return (main if aux is None else main + aux_weight*aux), {'main': main.detach(), 'aux': None if aux is None else aux.detach()}
