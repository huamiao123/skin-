"""No new preprocessing: reuse the exact historical data class and seed."""
from common import *
from stage3_confirmation.data import ImageData as HistoricalData

class ImageData(HistoricalData):
    def __init__(self):
        super().__init__(SEED)
        original=json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())
        assert self.audit==original['members'],'Historical sample/content mismatch'
        assert len(self.ids['train'])==1886 and len(self.ids['val'])==808

