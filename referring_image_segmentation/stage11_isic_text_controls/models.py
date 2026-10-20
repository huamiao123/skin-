"""Reuse the established DINO/Swin visual and text heads; initial hashes are verified."""
from common import *
from stage10_mosmed_text.models import Probe,masked_loss,FrozenEncoder,tensor_hash

def build_probes():return {n:Probe(n,SEED) for n in NAMES}
