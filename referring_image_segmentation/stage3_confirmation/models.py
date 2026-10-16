"""Seed-controlled initialization of the already established decoder structures."""
from pathlib import Path
import sys
import torch
from torch import nn

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from stage1_visual.models import FrozenEncoder, LightDecoder, Probe as BaseProbe, segmentation_loss, tensor_hash
from stage2_decoder.models import SwinPixelDecoder


class Probe(BaseProbe):
    def __init__(self, decoder, seed, channels=(384,384,384,384)):
        nn.Module.__init__(self)
        # CPU-only generators avoid modifying the process CUDA RNG during construction.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.adapter=nn.ModuleList([nn.Sequential(nn.Conv2d(c,64,1,bias=False),
                 nn.GroupNorm(8,64),nn.GELU()) for c in channels])
            torch.random.default_generator.manual_seed(seed)
            light=LightDecoder()
            if decoder=='light':
                self.decoder=light
            elif decoder=='swin':
                torch.random.default_generator.manual_seed(seed)
                self.decoder=SwinPixelDecoder()
                self.decoder.readout.load_state_dict(light.readout.state_dict())
            else:
                raise ValueError(decoder)
