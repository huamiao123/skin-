"""Documented D-stage command; all methods resolve q from the same B."""
import argparse
from pathlib import Path
from .runtime import load_config
from .train import train_stage, prepare_q

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--init',required=True);p.add_argument('--method',required=True,choices=['d0','rsi','mean_hinge','abs_hard'])
    p.add_argument('--weight',type=float,default=1.);p.add_argument('--run-name')
    a=p.parse_args();cfg=load_config(a.config);path=Path(a.init)
    path=path/'best.pth' if path.is_dir() else path
    if a.method=='d0' and a.weight!=0: p.error('D0 requires weight 0')
    if a.method!='d0' and a.weight not in cfg['conditional_weight_grid']:
        p.error('v2 restricts positive weights to the preregistered grid')
    q=prepare_q(path,cfg)
    train_stage(cfg,a.seed,'D',init=path,method=a.method,weight=a.weight,q=q,run_name=a.run_name)

if __name__=='__main__':main()
