"""Verify published snapshot bytes without loading models or reading datasets."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    inventory = json.loads((ROOT / 'publication/source_inventory.json').read_text())
    checked = {}
    for item in inventory['files']:
        if item['storage'] != 'git':
            continue
        path = (ROOT / item['published_path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Inventory path escapes repository')
        identity = (item['bytes'], item['sha256'])
        if path in checked:
            if checked[path] != identity:
                raise ValueError('Conflicting duplicate snapshot declaration')
            continue
        if not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('Published bytes differ: ' + item['published_path'])
        checked[path] = identity
    weights = json.loads((ROOT / 'publication/weights_manifest.json').read_text())
    if weights['storage'] != 'local_only' or weights['weights_uploaded'] or weights['release_created']:
        raise ValueError('Weights must remain local, as requested')
    objects = {o['sha256']: o for o in weights['objects']}
    if len(objects) != 32 or len(weights['objects']) != 32:
        raise ValueError('Expected 30 training/debug plus 2 classification objects')
    for sha, obj in objects.items():
        if not re.fullmatch(r'[0-9a-f]{64}', sha) or not isinstance(obj['bytes'], int) or obj['bytes'] <= 0:
            raise ValueError('Invalid weight object metadata')
    aliases = {p['source_path']: p for p in weights['paths']}
    if len(aliases) != 59 or len(weights['paths']) != 59:
        raise ValueError('Expected 59 distinct original weight paths')
    for alias in aliases.values():
        original = Path(alias['source_path'])
        if not original.is_absolute() or '..' in original.parts or not original.is_relative_to('/home/featurize'):
            raise ValueError('Invalid original weight path')
        obj = objects[alias['sha256']]
        if alias['bytes'] != obj['bytes'] or alias['category'] not in obj['categories'] or alias['source_path'] not in obj['source_paths']:
            raise ValueError('Weight path/object mapping differs')
    for obj in objects.values():
        actual = sorted(a['source_path'] for a in aliases.values() if a['sha256'] == obj['sha256'])
        if not actual or actual != sorted(obj['source_paths']):
            raise ValueError('Missing or inconsistent object aliases')
    local_weights = {}
    for item in inventory['files']:
        if item['storage'] == 'local_weight_not_uploaded':
            alias = aliases[item['source_path']]
            if (item['bytes'], item['sha256'], item['weight_category']) != (alias['bytes'], alias['sha256'], alias['category']):
                raise ValueError('Local weight inventory differs')
            if item['published_path'] is not None or item['source_path'] in local_weights:
                raise ValueError('Weight marked for publication or duplicated')
            local_weights[item['source_path']] = item
        elif item['storage'] not in {'git', 'withheld_third_party_data'}:
            raise ValueError('Unknown storage category')
    if set(local_weights) != set(aliases):
        raise ValueError('Weight inventories do not cover the same paths')
    print(json.dumps({'status': 'passed', 'distinct_snapshot_files_verified': len(checked),
                      'local_weight_objects_metadata': len(objects),
                      'local_weight_path_metadata': len(aliases), 'weight_binaries_uploaded': 0,
                      'models_or_datasets_loaded': False}))


if __name__ == '__main__':
    main()
