"""Exact historical image augmentation plus a new immutable word-token bank."""
from common import *
import numpy as np,torch
from stage9_isic_text.data import MedicalData as HistoricalData,FeatureCache as HistoricalCache

class MedicalData(HistoricalData):
    def __init__(self):
        super().__init__()
        assert json.loads((ROOT/'token_provenance.json').read_text())['passed']
        self.words=torch.from_numpy(np.load(TOKEN_ASSETS/'word_features512.npy')).cuda()
        self.word_valid=torch.from_numpy(np.load(TOKEN_ASSETS/'word_valid.npy')).cuda()
        self.row_word_index=torch.from_numpy(np.load(TOKEN_ASSETS/'row_word_index.npy')).cuda()
        self.fixed_word_index=json.loads((TOKEN_ASSETS/'text_vocabulary.json').read_text())['fixed_index']

    def word_batch(self,indices,fixed=False):
        dest=torch.full_like(indices,self.fixed_word_index) if fixed else self.row_word_index[indices]
        return dict(tokens=self.words[dest],valid=self.word_valid[dest])

class FeatureCache(HistoricalCache):
    SHAPES=((384,16,16),)*4
    def batch(self,split,positions,fixed=False):
        fs,y,valid,_=super().batch(split,positions)
        idx=self.data.indices[split][positions]
        return fs,y,valid,self.data.word_batch(idx,fixed=fixed)
