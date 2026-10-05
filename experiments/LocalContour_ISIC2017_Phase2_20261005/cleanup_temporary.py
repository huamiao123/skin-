"""Remove only rebuildable Phase-2 arrays after the complete final audit."""
from __future__ import annotations

import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'cache'

def main():
    audit=json.loads((ROOT/'results/phase2_summary/audit.json').read_text())
    if audit['status']!='PASS' or not audit['temporary_cache_ready_to_delete']:
        raise RuntimeError('Final audit must pass before derived cache cleanup')
    models=ROOT/'models'
    done=sorted(models.glob('*/DONE.json'))
    if len(done)!=21 or any(json.loads(p.read_text())['status']!='PASS' for p in done):
        raise RuntimeError('All 21 model DONE manifests must exist before cleanup')
    if not CACHE.is_dir():raise RuntimeError('Temporary cache already missing')
    provenance=ROOT/'results/cache_provenance';provenance.mkdir(parents=True,exist_ok=True)
    for source,name in [(CACHE/'COMPLETE.json','feature_cache_COMPLETE.json'),
                        (CACHE/'candidates_R32/COMPLETE.json','candidate_cache_COMPLETE.json'),
                        (CACHE/'node_context_COMPLETE.json','node_context_COMPLETE.json')]:
        if not source.is_file():raise RuntimeError('Missing cache metadata '+str(source))
        shutil.copyfile(source,provenance/name)
    files=[p for p in CACHE.rglob('*') if p.is_file()]
    cache_bytes=sum(p.stat().st_size for p in files)
    score_files=sorted([*models.glob('*/candidate_scores.npy'),*models.glob('*/gate_logits.npy')])
    if len(score_files)!=21+6:raise RuntimeError('Unexpected count of temporary model scores')
    score_bytes=sum(p.stat().st_size for p in score_files)
    before=shutil.disk_usage(ROOT).free
    shutil.rmtree(CACHE)
    for file in score_files:file.unlink()
    after=shutil.disk_usage(ROOT).free
    record={'status':'PASS','completed_utc':datetime.now(timezone.utc).isoformat(),
            'derived_cache_files_deleted':len(files),'derived_cache_logical_bytes_deleted':cache_bytes,
            'derived_model_score_files_deleted':len(score_files),
            'derived_model_score_logical_bytes_deleted':score_bytes,
            'total_logical_bytes_deleted':cache_bytes+score_bytes,
            'filesystem_free_bytes_before':before,'filesystem_free_bytes_after':after,
            'filesystem_free_bytes_delta':after-before,
            'retained':'all source, protocols, numeric results, diagnostic panels, model DONE and training logs, local best.pth checkpoint files',
            'weights_to_GitHub':False,'official_test_scored':False,
            'rebuild':'run build_feature_cache.py, build_candidate_cache.py --radius 32 and build_node_context.py; re-run trained models to regenerate candidate scores'}
    (provenance/'cleanup.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
