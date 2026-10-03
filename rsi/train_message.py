"""Documented ordinary-message command."""
import argparse
from pathlib import Path
from .runtime import load_config
from .train import train_stage

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--stage',choices=['B'],default='B');p.add_argument('--init')
    a=p.parse_args();cfg=load_config(a.config)
    init=a.init or str(Path(cfg['run_root'])/f'seed{a.seed}'/'A_T/best.pth')
    train_stage(cfg,a.seed,'B',init=init)

if __name__=='__main__':main()
