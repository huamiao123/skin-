"""Documented actual-tail action export; performance scoring is val-only."""
import argparse
from pathlib import Path
from .export import export_checkpoint
from .runtime import load_config

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--run',required=True)
    p.add_argument('--split',choices=['val'],default='val');p.add_argument('--actions',type=float,nargs='+',default=[0.,1.])
    p.add_argument('--seed',type=int,default=17);p.add_argument('--d0-checkpoint')
    a=p.parse_args();cfg=load_config(a.config);path=Path(a.run)
    path=path/'best.pth' if path.is_dir() else path
    if any(x not in cfg['fixed_shrink']['alphas'] for x in a.actions):
        p.error('v2 restricts alpha to 0, 1/3, 2/3, 1')
    for alpha in a.actions:
        export_checkpoint(path,cfg,run_id=f'{path.parent.name}-alpha-{alpha:.6f}',
                          alpha=alpha,method='fixed_shrink',d0_checkpoint=a.d0_checkpoint,seed=a.seed)

if __name__=='__main__':main()
