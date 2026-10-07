"""Resolve the sibling Phase-1 project and licensed local dataset at runtime."""
from __future__ import annotations

import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def phase1_root():
    override=os.environ.get('LOCALCONTOUR_PHASE1_ROOT')
    if override:return Path(override).expanduser().resolve()
    for candidate in (ROOT.parent/'contour_probe_20261005',
                      ROOT.parent/'LocalContour_ISIC2017_20261005'):
        if (candidate/'contour/geometry.py').is_file():return candidate.resolve()
    raise RuntimeError('Set LOCALCONTOUR_PHASE1_ROOT to the Phase-1 LocalContour directory')

def isic2017_root():
    return Path(os.environ.get('ISIC2017_ROOT','/home/featurize/rsi_data/benchmarks/isic2017')).expanduser().resolve()

PHASE1=phase1_root()
ISIC2017=isic2017_root()
TASKBOOK=ROOT/'protocols/LocalContour_下一阶段实验计划_2026-10-05.txt'
