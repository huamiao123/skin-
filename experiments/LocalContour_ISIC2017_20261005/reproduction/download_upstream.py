"""Fetch the frozen upstream implementation; weights remain untracked local assets."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

project=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--include-weights',action='store_true')
args=parser.parse_args()
manifest=json.loads((project/'third_party/msgu_net/source_manifest.json').read_text())
for item in manifest['files']:
    relative=Path(item['relative_path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid upstream relative path')
    if relative.suffix in {'.pth','.pt','.ckpt'} and not args.include_weights:
        continue
    with urlopen(item['url'],timeout=60) as response:
        content=response.read()
    if hashlib.sha256(content).hexdigest()!=item['sha256']:
        raise ValueError('Upstream bytes changed: '+str(relative))
    destination=project/'third_party/msgu_net'/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(content)
    print(str(relative),len(content))
print('Fetched pinned assets. Do not commit weights or data caches.')
