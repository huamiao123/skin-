"""Exact historical ISIC augmentation, input ordering and cache, for DINO only."""
from common import *
from stage9_isic_text.data import MedicalData,FeatureCache as HistoricalCache

class FeatureCache(HistoricalCache):
    SHAPES=((384,16,16),)*4
