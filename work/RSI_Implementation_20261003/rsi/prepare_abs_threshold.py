"""Freeze the B canonical M-train, image/rater weighted 75th percentile."""
import argparse
from pathlib import Path
from .runtime import load_config
from .train import prepare_q

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--run',required=True)
    p.add_argument('--split',choices=['train'],default='train');p.add_argument('--quantile',type=float,default=.75)
    a=p.parse_args()
    if a.quantile!=.75: p.error('v2 fixes the quantile at 0.75')
    path=Path(a.run);path=path/'best.pth' if path.is_dir() else path
    print(prepare_q(path,load_config(a.config)))

if __name__=='__main__':main()
