import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('/home/featurize/medseg_transformer_seed42_20261009')
status = {'started_utc': datetime.now(timezone.utc).isoformat(), 'jobs': {}}
for name in ('swin_pretrained', 'ptu_2262'):
    status['active'] = name
    status['jobs'][name] = {'state': 'running'}
    (ROOT / 'queue_status.json').write_text(json.dumps(status, indent=2))
    with (ROOT / f'{name}.log').open('a') as log:
        code = subprocess.call([sys.executable, '-u', str(ROOT / 'train.py'), name],
                               stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    complete = (ROOT / 'results' / name / 'DONE.json').exists()
    status['jobs'][name] = {'state': 'complete' if code == 0 and complete else 'failed',
                            'returncode': code, 'ended_utc': datetime.now(timezone.utc).isoformat()}
status['active'] = None
status['finished_utc'] = datetime.now(timezone.utc).isoformat()
(ROOT / 'queue_status.json').write_text(json.dumps(status, indent=2))
sys.exit(0 if all(j['state'] == 'complete' for j in status['jobs'].values()) else 1)
