"""Identical historical augmentation with sentence/word inputs for each model."""
from common import *
from stage12_predecoder_text.data import MedicalData as WordData
from stage9_isic_text.data import FeatureCache as HistoricalCache

class MedicalData(WordData):
    def text_for_model(self,name,indices):
        if name==NAMES[0]:return None
        if name in PRE_NAMES:return self.word_batch(indices,fixed=name in FIXED_NAMES)
        if name in FIXED_NAMES:return self.text[self.fixed_index].expand(len(indices),-1)
        return self.text[indices]

class FeatureCache(HistoricalCache):
    SHAPES=((384,16,16),)*4
