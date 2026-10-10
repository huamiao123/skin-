"""Background supervisor for the fixed-budget ordinary CNN+Transformer experiment."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/featurize/medseg_cnn_transformer_seed42_20261010')
status = {'model': 'cnn_transformer', 'seed': 42, 'epochs': 300,
          'started_utc': datetime.now(timezone.utc).isoformat(), 'state': 'running'}
(ROOT / 'run_status.json').write_text(json.dumps(status, indent=2) + '\n')
with (ROOT / 'train.log').open('a') as log:
    code = subprocess.call([sys.executable, '-u', str(ROOT / 'train.py'), 'cnn_transformer'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
status.update(state='complete' if code == 0 and (ROOT / 'results/cnn_transformer/DONE.json').exists()
              else 'failed', returncode=code, ended_utc=datetime.now(timezone.utc).isoformat())
(ROOT / 'run_status.json').write_text(json.dumps(status, indent=2) + '\n')
sys.exit(code)
