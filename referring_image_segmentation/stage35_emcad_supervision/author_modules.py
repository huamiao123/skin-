"""Original EMCAD decoder; author-style multi-output supervision is a training-only option."""
from pathlib import Path
import copy,importlib.util,sys
import torch
from torch import nn
from torch.nn import functional as F
R=Path('/home/waas/mambavision_stage22/research/topconf_open_source_20261111')
PROJECT=R.parent.parent/'source/referring_image_segmentation'
OLD=R.parent/'module_design_20261111/official_code'
NEW=R/'official_code'
def _import(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m
class AuthorModuleDecoder(nn.Module):
    def __init__(self,variant,baseline_decoder,seed=42):
        super().__init__()
        assert variant in ('emcad_single','emcad_deep')
        self.variant=variant
        official=_import(OLD/'SLDGroup__EMCAD/lib/decoders.py','stage35_native_emcad')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.author_decoder=official.EMCAD(channels=[64]*4,expansion_factor=2,activation='relu6')
            self.readout=copy.deepcopy(baseline_decoder.readout)
        if variant=='emcad_deep':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.aux_heads=nn.ModuleList([nn.Conv2d(64,1,1) for _ in range(3)])
    def forward(self,pyramid):
        assert len(pyramid)==4
        for p,h in zip(pyramid,[64,32,16,8]):
            assert p.shape[1:]==(64,h,h)
        d=self.author_decoder(pyramid[3],[pyramid[2],pyramid[1],pyramid[0]])
        fine=F.interpolate(self.readout(d[-1]),size=(256,256),mode='bilinear',align_corners=False)
        if self.variant=='emcad_deep' and self.training:
            return [F.interpolate(head(feat),size=(256,256),mode='bilinear',align_corners=False)
                    for head,feat in zip(self.aux_heads,d[:3])]+[fine]
        return fine
