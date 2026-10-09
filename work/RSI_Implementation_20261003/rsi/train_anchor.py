"""Documented A-stage command, with the completed previous stage resolved."""
import argparse
from pathlib import Path
from .runtime import load_config
from .train import train_stage

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--branch',choices=['cnn','transformer'],required=True);p.add_argument('--init')
    a=p.parse_args();cfg=load_config(a.config)
    stage='A_CNN' if a.branch=='cnn' else 'A_T'
    init=a.init or (str(Path(cfg['run_root'])/f'seed{a.seed}'/'A_CNN/best.pth') if stage=='A_T' else None)
    train_stage(cfg,a.seed,stage,init=init)

if __name__=='__main__':main()
