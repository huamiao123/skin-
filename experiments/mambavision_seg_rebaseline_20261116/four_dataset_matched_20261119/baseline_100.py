"""Launch the 100-epoch UPer64 baseline on the last two target datasets."""
from pathlib import Path
import sys
import fit_matched
from matched_core import recipe, sha

def fit_baseline(arm):
    assert arm == 'uper64'
    reference = recipe.config(arm)
    assert (reference['max_epochs'], reference['patience']) == (200, 25)
    recipe.MAX_EPOCHS = 100
    config = dict(reference, max_epochs=100, scheduler='cosine to 1% over 100 epochs',
                  baseline_protocol_sha256=sha(Path(__file__)))
    recipe.config = lambda name: config.copy() if name == arm else None
    original_fit(arm)

if __name__ == '__main__':
    dataset, = sys.argv[1:]
    if dataset not in ('Waterloo', 'ISIC2018'):
        raise ValueError(dataset)
    original_fit = recipe.fit
    recipe.fit = fit_baseline
    fit_matched.main(dataset, 'uper64')
