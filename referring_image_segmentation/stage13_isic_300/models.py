"""Exact existing architectures, combined without loading trained weights."""
from common import *
from stage10_mosmed_text.models import Probe as LateProbe
from stage12_predecoder_text.models import Probe as PreProbe,masked_loss,FrozenEncoder,tensor_hash

def build_probes():
    return {n:(PreProbe(n,SEED) if n in PRE_NAMES else LateProbe(n,SEED)) for n in NAMES}
