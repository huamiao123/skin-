"""Documented validation summary entry point."""
import argparse
import csv
from pathlib import Path
from .runtime import PROJECT,load_config
from .summarize import summarize_pilot

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--protocol',default='P2-IMA-M-v2')
    p.add_argument('--split',choices=['val'],default='val');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--input',nargs='+',required=True);p.add_argument('--output')
    a=p.parse_args();cfg=load_config(a.config)
    if a.protocol!=cfg['protocol_id']:p.error('protocol must match the frozen configuration')
    rows=[]
    for name in a.input:
        with Path(name).open(newline='') as f:rows.extend(csv.DictReader(f))
    summarize_pilot(rows,a.output or PROJECT/'outputs'/f'seed{a.seed}',
                    bootstrap_repeats=cfg['bootstrap_repeats'],bootstrap_seed=a.seed)

if __name__=='__main__':main()
