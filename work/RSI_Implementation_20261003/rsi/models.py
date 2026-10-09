"""CIR-Base: an ImageNet initialized CNN anchor with one message-only entrance.

RGB inputs are floats in [0, 1]. Pixel validity describes image letterboxing,
never an annotation. All segmentation outputs are logits at the input size.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import resnet34


PRETRAINED_DIR = Path(os.environ.get("RSI_PRETRAINED_DIR", "/home/featurize/rsi_data/pretrained"))
PROJECT_DIR = Path(__file__).resolve().parents[1]
WEIGHTS = {
    "cnn": ("https://download.pytorch.org/models/resnet34-b627a593.pth", "resnet34-b627a593.pth"),
    "transformer": ("https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth", "deit_small_patch16_224-cd65a155.pth"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(module: nn.Module) -> str:
    """Hash every named parameter and buffer, including BatchNorm counters."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _get_weights(kind: str, cache_dir: Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    url, filename = WEIGHTS[kind]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    prefix = filename.rsplit("-", 1)[1].split(".")[0]
    if not path.exists():
        temporary = path.with_suffix(".download")
        torch.hub.download_url_to_file(url, str(temporary), hash_prefix=prefix, progress=False)
        temporary.replace(path)
    sha256 = file_sha256(path)
    if not sha256.startswith(prefix):
        raise RuntimeError(f"Pretrained hash mismatch: {path}; expected prefix {prefix}, got {sha256}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    weights = checkpoint.get("model", checkpoint)
    if not isinstance(weights, dict) or not all(isinstance(v, Tensor) for v in weights.values()):
        raise RuntimeError(f"Unexpected pretrained state dictionary: {path}")
    return weights, {"source_url": url, "path": str(path.resolve()), "sha256": sha256,
                     "bytes": path.stat().st_size, "expected_hash_prefix": prefix,
                     "source_keys": len(weights)}


class ImageNetNormalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("Expected float RGB images with shape [B,3,H,W] in [0,1]")
        return (images - self.mean) / self.std


def resize(features: Tensor, size: tuple[int, int]) -> Tensor:
    return F.interpolate(features, size=size, mode="bilinear", align_corners=False)


class ConvGN(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2,
                                  bias=False), nn.GroupNorm(8, out_channels), nn.GELU())


class ResNet34Features(nn.Module):
    """Keep only stem/layer1/layer2/layer3; layer4 and the classifier are absent."""
    def __init__(self, pretrained: bool, cache_dir: Path) -> None:
        super().__init__()
        source = resnet34(weights=None)
        self.pretrained_audit: dict[str, Any] = {"loaded": False}
        if pretrained:
            weights, audit = _get_weights("cnn", cache_dir)
            result = source.load_state_dict(weights, strict=True)
            audit.update(loaded=True, missing_keys=list(result.missing_keys), unexpected_keys=list(result.unexpected_keys),
                         removed_modules=["layer4", "avgpool", "fc"], initialization="ImageNet1K classification")
            self.pretrained_audit = audit
        self.stem = nn.Sequential(source.conv1, source.bn1, source.relu, source.maxpool)
        self.layer1, self.layer2, self.layer3 = source.layer1, source.layer2, source.layer3

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        c4 = self.layer1(self.stem(images))
        c8 = self.layer2(c4)
        c16 = self.layer3(c8)
        return c4, c8, c16


class DecoderPrefix(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c16_proj = nn.Sequential(ConvGN(256, 128, 1), ConvGN(128, 128))
        self.c8_proj = ConvGN(128, 128, 1)
        self.fuse = nn.Sequential(ConvGN(256, 128), ConvGN(128, 128))

    def forward(self, c8: Tensor, c16: Tensor) -> Tensor:
        f16 = self.c16_proj(c16)
        return self.fuse(torch.cat([resize(f16, c8.shape[-2:]), self.c8_proj(c8)], dim=1))


class DecoderTail(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.c4_proj = ConvGN(64, 64, 1)
        self.fuse = nn.Sequential(ConvGN(192, 64), ConvGN(64, 64))
        self.up_half = ConvGN(64, 32)
        self.up_full = ConvGN(32, 32)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, f8: Tensor, c4: Tensor) -> Tensor:
        x = self.fuse(torch.cat([resize(f8, c4.shape[-2:]), self.c4_proj(c4)], dim=1))
        x = self.up_half(resize(x, (self.input_size // 2, self.input_size // 2)))
        return self.output(self.up_full(resize(x, (self.input_size, self.input_size))))


class CNNAnchor(nn.Module):
    def __init__(self, pretrained: bool, input_size: int, cache_dir: Path) -> None:
        super().__init__()
        self.input_size = input_size
        self.normalize = ImageNetNormalize()
        self.encoder = ResNet34Features(pretrained, cache_dir)
        self.prefix = DecoderPrefix()
        self.tail = DecoderTail(input_size)

    def train(self, mode: bool = True) -> "CNNAnchor":
        super().train(False if getattr(self, "_permanently_frozen", False) else mode)
        return self

    def features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        if images.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(f"Expected {self.input_size} x {self.input_size} images")
        c4, c8, c16 = self.encoder(self.normalize(images))
        return self.prefix(c8, c16), c4

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        f8, c4 = self.features(images)
        return f8, c4, self.tail(f8, c4)


def interpolate_deit_position(position: Tensor, grid_size: tuple[int, int]) -> Tensor:
    """Preserve the sole CLS position; bicubic interpolate only patch positions."""
    cls, patches = position[:, :1], position[:, 1:]
    source_side = math.isqrt(patches.shape[1])
    if source_side * source_side != patches.shape[1]:
        raise ValueError("DeiT non-distilled checkpoint must have one CLS and a square patch grid")
    patches = patches.reshape(1, source_side, source_side, position.shape[-1]).permute(0, 3, 1, 2)
    patches = F.interpolate(patches, size=grid_size, mode="bicubic", align_corners=False)
    return torch.cat([cls, patches.flatten(2).transpose(1, 2)], dim=1)


class DeiTContext(nn.Module):
    def __init__(self, pretrained: bool, input_size: int, cache_dir: Path) -> None:
        super().__init__()
        self.normalize = ImageNetNormalize()
        self.encoder = timm.create_model("deit_small_patch16_224", pretrained=False,
                                         img_size=input_size, num_classes=0)
        self.grid_size = self.encoder.patch_embed.grid_size
        if self.encoder.num_prefix_tokens != 1 or self.encoder.num_features != 384 or len(self.encoder.blocks) != 12:
            raise RuntimeError("Expected non-distilled DeiT-Small, one CLS, width384, depth12")
        # Explicit matmul attention makes FP32 deterministic validation transparent.
        for block in self.encoder.blocks:
            block.attn.fused_attn = False
        self.pretrained_audit: dict[str, Any] = {"loaded": False}
        if pretrained:
            weights, audit = _get_weights("transformer", cache_dir)
            source_pos = weights["pos_embed"].clone()
            removed = sorted(key for key in weights if key.startswith("head."))
            weights = {key: value for key, value in weights.items() if key not in removed}
            weights["pos_embed"] = interpolate_deit_position(source_pos, self.grid_size)
            result = self.encoder.load_state_dict(weights, strict=True)
            if not torch.equal(self.encoder.pos_embed[:, :1].detach(), source_pos[:, :1]):
                raise RuntimeError("CLS position changed during interpolation")
            audit.update(loaded=True, missing_keys=list(result.missing_keys), unexpected_keys=list(result.unexpected_keys),
                         loaded_keys=len(weights), removed_classifier_keys=removed,
                         source_position_shape=list(source_pos.shape), target_position_shape=list(self.encoder.pos_embed.shape),
                         cls_shape=list(self.encoder.cls_token.shape), cls_position_preserved=True,
                         prefix_tokens=1, distilled=False, patch_grid=list(self.grid_size),
                         position_interpolation="bicubic; align_corners=False; CLS excluded",
                         initialization="ImageNet1K classification")
            self.pretrained_audit = audit

    def forward(self, images: Tensor) -> Tensor:
        tokens = self.encoder.forward_features(self.normalize(images))
        patches = tokens[:, 1:]
        return patches.transpose(1, 2).reshape(images.shape[0], 384, *self.grid_size)


class AuxiliaryHead(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.proj = ConvGN(384, 128, 1)
        self.up_quarter = ConvGN(128, 64)
        self.up_full = ConvGN(64, 32)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, t16: Tensor) -> Tensor:
        x = self.up_quarter(resize(self.proj(t16), (self.input_size // 4, self.input_size // 4)))
        return self.output(self.up_full(resize(x, (self.input_size, self.input_size))))


class CrossAttentionMessage(nn.Module):
    """One Q/K/V projection each, followed by the only zero initialized layer Wo.

    There is no nn.MultiheadAttention with a second implicit projection. CNN
    query width128; DeiT key/value width384; internal width128, four heads.
    """
    def __init__(self) -> None:
        super().__init__()
        self.heads, self.head_dim = 4, 32
        self.q_proj = nn.Linear(128, 128)
        self.k_proj = nn.Linear(384, 128)
        self.v_proj = nn.Linear(384, 128)
        self.Wo = nn.Linear(128, 128)
        nn.init.zeros_(self.Wo.weight)
        nn.init.zeros_(self.Wo.bias)

    def forward(self, f8: Tensor, t16: Tensor, context_valid: Tensor | None = None) -> Tensor:
        batch, channels, height, width = f8.shape
        query, context = f8.flatten(2).transpose(1, 2), t16.flatten(2).transpose(1, 2)
        def split_heads(tokens: Tensor) -> Tensor:
            return tokens.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(self.q_proj(query)), split_heads(self.k_proj(context)), split_heads(self.v_proj(context))
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if context_valid is not None:
            if context_valid.shape != (batch, context.shape[1]):
                raise ValueError("context_valid must have shape [B,context_tokens]")
            if not bool(context_valid.any(dim=1).all()):
                raise ValueError("Each image must have at least one valid context token")
            scores = scores.masked_fill(~context_valid[:, None, None, :].bool(), float("-inf"))
        attended = torch.matmul(scores.softmax(dim=-1), v).transpose(1, 2).reshape(batch, height * width, channels)
        return self.Wo(attended).transpose(1, 2).reshape(batch, channels, height, width)


class ControlledModel(nn.Module):
    def __init__(self, pretrained: bool = True, input_size: int = 256,
                 cache_dir: str | Path | None = None, audit_path: str | Path | None = None) -> None:
        super().__init__()
        if input_size % 32 != 0:
            raise ValueError("input_size must be divisible by32")
        self.input_size = input_size
        cache_dir = Path(cache_dir) if cache_dir is not None else PRETRAINED_DIR
        self.cnn = CNNAnchor(pretrained, input_size, cache_dir)
        self.transformer = DeiTContext(pretrained, input_size, cache_dir)
        self.aux_head = AuxiliaryHead(input_size)
        self.message = CrossAttentionMessage()
        self._stage = "A_CNN"
        self.set_stage(self._stage)
        self.pretrained_audit = {"cnn": self.cnn.encoder.pretrained_audit,
                                "transformer": self.transformer.pretrained_audit,
                                "model": self.architecture_audit()}
        if pretrained:
            destination = Path(audit_path) if audit_path is not None else PROJECT_DIR / "outputs/pretrained_audit.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Repeated training construction must not erase the recorded GPU
            # feasibility/acceptance measurements for the same architecture.
            previous = json.loads(destination.read_text()) if destination.exists() else {}
            same_weights = all(previous.get(name, {}).get("sha256") == self.pretrained_audit[name].get("sha256")
                               for name in ("cnn", "transformer"))
            same_architecture = previous.get("model") == self.pretrained_audit["model"]
            if same_weights and same_architecture:
                previous.update(self.pretrained_audit)
                self.pretrained_audit = previous
            destination.write_text(json.dumps(self.pretrained_audit, ensure_ascii=False, indent=2) + "\n")

    @property
    def tail(self) -> DecoderTail:
        return self.cnn.tail

    @property
    def prefix(self) -> DecoderPrefix:
        return self.cnn.prefix

    @property
    def stage(self) -> str:
        return self._stage

    def set_stage(self, stage: str) -> "ControlledModel":
        stage = stage.replace("-", "_").upper()
        if stage not in {"A_CNN", "A_T", "B", "D"}:
            raise ValueError(f"Unknown training stage: {stage}")
        self._stage = stage
        for name, module in self._components().items():
            trainable = self._component_trainable(name)
            module.requires_grad_(trainable)
            for parameter in module.parameters():
                parameter.grad = None
            module.train(self.training and trainable)
        return self

    def _components(self) -> dict[str, nn.Module]:
        return {"cnn": self.cnn, "transformer": self.transformer, "aux_head": self.aux_head, "message": self.message}

    def _component_trainable(self, name: str) -> bool:
        return ((self._stage == "A_CNN" and name == "cnn")
                or (self._stage == "A_T" and name in {"transformer", "aux_head"})
                or (self._stage in {"B", "D"} and name == "message"))

    def train(self, mode: bool = True) -> "ControlledModel":
        super().train(mode)
        if hasattr(self, "_stage"):
            for name, module in self._components().items():
                module.train(mode and self._component_trainable(name))
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def cnn_features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        return self.cnn.features(images)

    def forward_cnn(self, images: Tensor) -> Tensor:
        return self.cnn(images)[2]

    def forward_transformer(self, images: Tensor) -> Tensor:
        return self.aux_head(self.transformer(images))

    def context_token_valid(self, valid: Tensor | None, batch_size: int) -> Tensor | None:
        if valid is None:
            return None
        if valid.ndim == 3:
            valid = valid[:, None]
        if valid.shape != (batch_size, 1, self.input_size, self.input_size):
            raise ValueError("valid must describe the input image pixels, shape [B,1,H,W]")
        # Include partially valid tokens; mask only fully padded 16x16 patches.
        return F.max_pool2d(valid.float(), kernel_size=16, stride=16).flatten(1) > 0

    def _features(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self._stage in {"B", "D"}:
            with torch.no_grad():
                f8, c4 = self.cnn.features(images)
                t16 = self.transformer(images)
            return f8, c4, t16
        f8, c4 = self.cnn.features(images)
        return f8, c4, self.transformer(images)

    def forward(self, images: Tensor, valid: Tensor | None = None, alpha: float | Tensor = 1.0,
                return_aux: bool = False):
        """Deployment: one final tail decode; no anchor/annotation is an input.

        return_aux=True returns (final_logits, auxiliary_logits), otherwise logits.
        """
        f8, c4, t16 = self._features(images)
        message = self.message(f8, t16, self.context_token_valid(valid, images.shape[0]))
        logits = self.tail(f8 + alpha * message, c4)
        return (logits, self.aux_head(t16)) if return_aux else logits

    def forward_pair(self, images: Tensor, valid: Tensor | None = None, alpha: float | Tensor = 1.0) -> dict[str, Tensor]:
        """Training/diagnostics explicitly pay for both anchor and updated tails."""
        f8, c4, t16 = self._features(images)
        with torch.no_grad():
            z0 = self.tail(f8, c4)
        message = self.message(f8, t16, self.context_token_valid(valid, images.shape[0]))
        z1 = self.tail(f8 + alpha * message, c4)
        return {"z0": z0, "z1": z1, "message": message, "f8": f8, "c4": c4, "t16": t16}

    def anchor_state_hash(self) -> str:
        return state_hash(self.cnn)

    def frozen_state_hash(self) -> str:
        hashes = {name: state_hash(module) for name, module in self._components().items()
                  if not self._component_trainable(name)}
        return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()

    def permanent_anchor(self) -> CNNAnchor:
        anchor = copy.deepcopy(self.cnn).eval().requires_grad_(False)
        anchor._permanently_frozen = True
        return anchor

    def architecture_audit(self) -> dict[str, Any]:
        counts = {name: sum(p.numel() for p in module.parameters()) for name, module in self._components().items()}
        return {"input_size": self.input_size, "parameters": counts, "total_parameters": sum(counts.values()),
                "cnn_feature_channels": [64, 128, 256], "cnn_last_layer": "layer3", "deit_width": 384,
                "deit_depth": 12, "message_width": 128, "message_heads": 4,
                "message_qkv_projection_count": 1, "zero_initialization": "Wo.weight and Wo.bias only",
                "decoder_norm": "GroupNorm8", "decoder_activation": "GELU", "align_corners": False,
                "normalization": {"location": "inside each encoder path", "input_rgb_range": [0.0, 1.0],
                                  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "transformer_padding_propagation": "context message mask does not remove encoder padding interactions",
                "deployment_tail_decodes": 1, "auxiliary_head_deployment_bypass": False}
