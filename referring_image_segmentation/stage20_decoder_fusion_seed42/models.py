"""Frozen MambaVision and controlled HEDS AXIS skip ablations.

The bridge is the upstream HEDS-Net class at a pinned revision, adapted only
by explicit NCHW -> NHWC -> NCHW conversion at the 32 x 32 feature level.
The upstream shape heuristic works in NHWC here because 32 != 64. Calling it
directly on NCHW would incorrectly permute the channels. We do not generalize
this wrapper to 64 x 64, where those dimensions become ambiguous.
"""
from common import *
import importlib.util
from torch.nn import functional as F
from stage3_confirmation.models import Probe, segmentation_loss, tensor_hash
from stage17_mambavision_decoder_hiram_seed42.models import FrozenEncoder

UPSTREAM = ROOT / 'references' / 'axis_bridge_source.py'
UPSTREAM_SHA256 = '5a054761661647b0ed8fe8d2bf67f4878c0f9aafdaf75d2e6f68e482a64bddf1'

def load_axis_bridge():
    assert sha(UPSTREAM) == UPSTREAM_SHA256, 'Unexpected upstream AXIS source'
    spec = importlib.util.spec_from_file_location('heds_axis_bridge_pinned', UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AxisBridge


class DecoderProbe(Probe):
    def __init__(self, mode):
        assert mode in ('paired_baseline', 'gate_only', 'axis_bridge')
        super().__init__('swin', 42, channels=CHANNELS)
        self.mode = mode
        if mode != 'paired_baseline':
            AxisBridge = load_axis_bridge()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(1042)
                self.bridge = AxisBridge(64, use_coordinate_attention=(mode == 'axis_bridge'))

    def forward(self, features):
        assert len(features) == 4
        pyramid = []
        for feature, adapter, side in zip(features, self.adapter, (64, 32, 16, 8)):
            x = adapter(feature)
            if x.shape[-2:] != (side, side):
                x = F.interpolate(x, (side, side), mode='bilinear', align_corners=False)
            pyramid.append(x)
        x = self.decoder.blocks[3](pyramid[3])
        for i in (2, 1, 0):
            coarse = F.interpolate(x, size=pyramid[i].shape[-2:], mode='bilinear', align_corners=False)
            skip = pyramid[i]
            if i == 1 and self.mode != 'paired_baseline':
                assert list(coarse.shape[1:]) == [64, 32, 32]
                x = self.bridge(coarse.permute(0, 2, 3, 1), skip.permute(0, 2, 3, 1))
                x = x.permute(0, 3, 1, 2).contiguous()
            else:
                x = coarse + skip
            x = self.decoder.blocks[i](x)
        return F.interpolate(self.decoder.readout(x), size=(256, 256), mode='bilinear', align_corners=False)
