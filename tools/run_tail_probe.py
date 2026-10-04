"""Run only the preregistered isolated seed17 decoder-tail experiment."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path: sys.path.insert(0,str(PROJECT))
from rsi.runtime import load_config,atomic_json,sha256_file


def freeze_sources(cfg):
    from tools.tail_probe_training import provenance
    root=Path(cfg['output_root'])
    code=provenance(cfg)
    path=root/'code_provenance.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if prior['source_bundle_hash']!=code['source_bundle_hash']:
            raise ValueError('Complete frozen experiment code changed')
        return prior
    code.update(frozen_utc=datetime.now(timezone.utc).isoformat(),
                coverage='All rsi, tools, tests Python plus all configs YAML; taskbook SHA included',
                parent_diff='new_source_diff.patch')
    patch=subprocess.check_output(['git','diff',cfg['review_commit'],'--','rsi','tools','tests','configs','protocols'],cwd=PROJECT)
    (root/'new_source_diff.patch').write_bytes(patch)
    atomic_json(path,code)
    return code


def deliver_reuse(cfg):
    root=Path(cfg['output_root'])
    report=json.loads((root/'reuse_decision.json').read_text())
    if not report.get('reused',report.get('reuse_approved',report.get('old_F_reused',False))):
        # Analysis provides detailed same-condition evidence; its actual schema
        # is checked explicitly by preflight before this helper is called.
        if report.get('status') not in {'PASS','REUSED','reuse_passed'}:
            raise ValueError('Legacy F reuse was not approved')
    for group in ['F-Mean','F-RSI']:
        src=Path(cfg['selected_checkpoints'][group]).parent
        dst=Path(cfg['run_root'])/'seed17'/group;dst.mkdir(parents=True,exist_ok=True)
        old=json.loads((src/'DONE.json').read_text())
        for name in ['best.pth','latest.pth','epoch_diagnostics.csv','per_image_canonical_diagnostics.csv','config_resolved.json','environment.json']:
            target=dst/name
            if target.exists():
                if sha256_file(target)!=sha256_file(src/name): raise ValueError('Reused evidence mirror differs')
            else: shutil.copy2(src/name,target)
        atomic_json(dst/'legacy_DONE.json',old)
        atomic_json(dst/'DONE.json',dict(group=group,run=group,seed=17,stage='TailProbe',epochs=40,reused=True,
            best_epoch=old['best_epoch'],best_val_dice=old['best_val_dice'],
            best_sha256=old['best_sha256'],latest_sha256=old['latest_sha256'],
            legacy_checkpoint=str(src/'best.pth'),legacy_DONE=str(src/'DONE.json'),
            legacy_DONE_sha256=sha256_file(src/'DONE.json'),
            reuse_evidence=str(root/'reuse_decision.json'),reuse_evidence_sha256=sha256_file(root/'reuse_decision.json'),
            historical_wall_seconds=old['wall_seconds'],new_training_seconds=0.,
            test_scoring_locked=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ['preflight','train','export','summarize','all']:
        sub=commands.add_parser(name)
        sub.add_argument('--config',default=str(PROJECT/'configs/tail_probe_seed17.yaml'))
        sub.add_argument('--dry-run',action='store_true')
        if name=='train': sub.add_argument('--group',required=True,choices=['F-Mean','F-RSI','U-NoMessage','U-Mean','U-RSI'])
        if name=='export': sub.add_argument('--scope',default='val',choices=['train','val','test'])
    args=parser.parse_args();cfg=load_config(args.config)
    if args.command=='export' and args.scope!='val': raise PermissionError('Test and train scoring export are locked')
    from tools.tail_probe_training import check_inputs
    root,manifest_hash,init_hash=check_inputs(cfg)
    if args.dry_run:
        print(json.dumps(dict(status='DRY_RUN',command=args.command,group=getattr(args,'group',None),
              manifest_hash=manifest_hash,init_hash=init_hash,epochs=40,seed=17,
              train_order=cfg['train_order'],output_root=str(root),test_scoring_locked=True),ensure_ascii=False))
        return
    try:
        if args.command in ['preflight','all']:
            from tools.tail_probe_preflight import run_preflight
            result=run_preflight(cfg)
            if result.get('status')!='PASS': raise ValueError('Preflight not passed: '+str(result.get('status')))
            atomic_json(root/'reuse_decision.json',dict(status='PASS',reused=True,same_condition_factors=True,
                groups=['F-Mean','F-RSI'],reason='Full asset/budget, actual FP32/AMP train/val forward/objective/gradient and all 624-reference readouts passed.',
                engineering_acceptance_sha256=sha256_file(root/'engineering_acceptance.json'),
                readout=result['full_f_reuse_readout'],core_environment='Same 4090/torch2.2.2+cu121/torchvision0.17.2/timm1.0.22',
                sampling='Unchanged original loader seed+epoch*100003 and sha256(seed:epoch:image_id) augmentation; micro4 effective8',
                discarded_smoke_weights=True,seed=17,test_scoring_locked=True))
            deliver_reuse(cfg)
            print('PREFLIGHT_PASS',flush=True)
        if args.command in ['train','all']:
            from tools.tail_probe_training import train_group
            freeze_sources(cfg)
            groups=cfg['train_order'] if args.command=='all' else [args.group]
            for group in groups:
                print('TRAIN_START',group,flush=True);train_group(cfg,group);print('TRAIN_DONE',group,flush=True)
        if args.command in ['export','all']:
            from tools.tail_probe_analysis import run_export
            run_export(cfg,scope='val')
        if args.command in ['summarize','all']:
            from tools.tail_probe_analysis import run_summarize
            run_summarize(cfg)
    except BaseException as exc:
        atomic_json(root/'last_failure.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
            command=args.command,group=getattr(args,'group',None),exception_type=type(exc).__name__,
            message=str(exc),traceback=traceback.format_exc(),test_scoring_locked=True))
        with (root/'failure_and_resume_log.md').open('a') as f:
            f.write('\n'+datetime.now(timezone.utc).isoformat()+' '+args.command+' '+type(exc).__name__+': '+str(exc)+'\n')
        raise


if __name__=='__main__': main()
