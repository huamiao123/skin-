"""Seed replication of the existing concatenation and capacity controls."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from stage4_fusion.models import EncoderPair, FusionProbe, CHANNELS
from stage5_selective_fusion.models import CapacityProbe
from stage3_confirmation.models import Probe as SingleProbe
from stage1_visual.models import tensor_hash, segmentation_loss

SEEDS = (43, 44, 45)
NAMES = ('capacity_concat', 'fusion_concat')
ALL_NAMES = ('dinov3',) + NAMES
SPEC = dict(decoder='unchanged stage2 SwinPixelDecoder', channels=64,
    resolutions=[64,32,16,8], fusion='Concat followed by Conv2d(128,64,1,bias=True)',
    capacity='one frozen DINO forward; two independently trainable DINO adapters',
    initialization='per-run seed; decoder/DINO adapter/fusion initial tensors paired',
    baseline='read-only stage3 DINO+Swin checkpoints after reuse audit')


def build_probes(seed):
    return dict(capacity_concat=CapacityProbe(seed), fusion_concat=FusionProbe('swin',seed))


def select_features(name, features):
    if name == 'dinov3': return features[:4]
    if name == 'capacity_concat': return features[:4] + features[:4]
    assert name == 'fusion_concat'
    return features
