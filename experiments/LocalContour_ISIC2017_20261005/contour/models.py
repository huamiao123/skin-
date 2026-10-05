"""Strict public CNN adapter and one independently trained boundary score head."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_COMMIT = 'c4b16bf372067eb163a44f16104473383ac7d718'
PUBLIC_WEIGHT_SHA256 = '3a7fb064383cc68faf33b76561379b69a84141cf51e94679dc31e3a3e66a515d'


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_state_hash(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(str(tuple(array.shape)).encode())
        h.update(array.numpy().tobytes())
    return h.hexdigest()


def segmentation_mask(logits: torch.Tensor) -> torch.Tensor:
    # ReLU6 in the public final layer makes >=0 an invalid all-foreground rule.
    # Keep the author's sigmoid operation and strict > comparison exactly.
    return torch.sigmoid(logits) > 0.5


def _load_public_class(third_party_root: Path):
    manifest = json.loads((third_party_root / 'source_manifest.json').read_text())
    if manifest['commit'] != PUBLIC_COMMIT:
        raise ValueError('Public CNN source commit differs from the pinned commit')
    for entry in manifest['files']:
        path = third_party_root / entry['relative_path']
        if not path.is_file() or file_sha256(path) != entry['sha256']:
            raise ValueError(f'Public CNN source/weight hash mismatch: {entry["relative_path"]}')
    # The unchanged author's model imports a package literally named "modules".
    existing = sys.modules.get('modules')
    if existing is not None and Path(getattr(existing, '__file__', '')).resolve().parent != (third_party_root / 'modules').resolve():
        raise RuntimeError('A different global modules package conflicts with the pinned public CNN')
    if str(third_party_root) not in sys.path:
        sys.path.insert(0, str(third_party_root))
    spec = importlib.util.spec_from_file_location('_contour_public_msgunet', third_party_root / 'model/MSGUNet.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MSGUNet


class FrozenMSGUNet(nn.Module):
    def __init__(self, third_party_root: str | Path | None = None, checkpoint_path: str | Path | None = None):
        super().__init__()
        self.third_party_root = Path(third_party_root or PROJECT_ROOT / 'third_party/msgu_net')
        self.checkpoint_path = Path(checkpoint_path or self.third_party_root / 'weights/best_model_isic2017.pth')
        if file_sha256(self.checkpoint_path) != PUBLIC_WEIGHT_SHA256:
            raise ValueError('A different checkpoint cannot substitute for the verified public CNN')
        public_class = _load_public_class(self.third_party_root)
        # Avoid consuming the head's RNG sequence when constructing the CNN.
        with torch.random.fork_rng(devices=[]):
            self.cnn = public_class(in_channels=3, out_channels=1, base_channels=32)
        state = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
        self.cnn.load_state_dict(state, strict=True)
        for parameter in self.cnn.parameters():
            parameter.requires_grad_(False)
        self._feature = None
        self._hook = self.cnn.dec1.register_forward_hook(self._capture_feature)
        self.train(False)

    def _capture_feature(self, module, args, output):
        self._feature = output.detach()

    def train(self, mode: bool = True):
        # Every module and every BN buffer remains in evaluation mode.
        super().train(False)
        self.cnn.eval()
        return self

    @torch.no_grad()
    def forward(self, normalized_rgb: torch.Tensor) -> torch.Tensor:
        return self.cnn(normalized_rgb)

    @torch.no_grad()
    def forward_with_features(self, normalized_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        self._feature = None
        logits = self.cnn(normalized_rgb)
        feature = self._feature
        self._feature = None
        if feature is None or feature.shape[1:] != (32, 256, 256):
            raise RuntimeError('Expected the verified dec1 32x256x256 CNN feature')
        return {'logits': logits.detach(), 'probability': torch.sigmoid(logits).detach(), 'mask': segmentation_mask(logits), 'features': feature}

    def state_hash(self) -> str:
        return tensor_state_hash(self.cnn)

    def architecture_audit(self) -> dict:
        return {'model': 'MSGUNet(3,1,base_channels=32)', 'commit': PUBLIC_COMMIT, 'weight_sha256': PUBLIC_WEIGHT_SHA256, 'strict_state_dict': True, 'cnn_trainable_parameters': sum(p.numel() for p in self.cnn.parameters() if p.requires_grad), 'all_modules_eval': all(not m.training for m in self.cnn.modules()), 'final_output': 'author Conv2d + BatchNorm2d + ReLU6; treated as BCE logits exactly as public scripts', 'hard_mask_rule': 'torch.sigmoid(author_output) > 0.5', 'feature': 'dec1, full resolution 32x256x256', 'new_cnn_training': False, 'historical_official_val_exposure': 'author combines official train+val then 70/30 splits; this probe is development diagnosis'}


class BoundaryHead(nn.Module):
    """705 parameters; fixed FP16 cache precision, FP32 head operations."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(32, 16, 1), nn.GroupNorm(4, 16), nn.ReLU(), nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != 32:
            raise ValueError('Boundary head requires Bx32xHxW CNN features')
        # Use identical feature quantization for cache training and later RGB inference.
        return self.net(features.detach().to(torch.float16).to(torch.float32))

    def architecture_audit(self) -> dict:
        return {'parameters': sum(p.numel() for p in self.parameters()), 'input_feature': 'frozen CNN dec1 32x256x256', 'feature_storage': 'CPU float16 memmap; cast back to float32 for the head in every phase', 'head_operations': 'FP32 Conv1x1(32,16)+GroupNorm(4,16)+ReLU+Conv3x3(16,1)', 'additional_rgb_or_gradient_input': False, 'new_transformer': False}
