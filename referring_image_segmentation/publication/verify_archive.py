"""Verify the published snapshot without local data, weights or training."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / 'publication/source_inventory.json').read_text())
    checked = 0
    failures = []
    for row in inventory['files'] + inventory['publication_files']:
        if row['disposition'] != 'git':
            continue
        path = root / row['path']
        if not path.is_file() or path.is_symlink():
            failures.append(row['path'] + ': missing or not a regular file')
        elif path.stat().st_size != row['published_bytes'] or sha256(path) != row['published_sha256']:
            failures.append(row['path'] + ': content mismatch')
        checked += 1
    if failures:
        raise SystemExit('\n'.join(failures))
    print(json.dumps({'passed': True, 'published_files_checked': checked,
                      'local_only_files_checked': False}, indent=2))


if __name__ == '__main__':
    main()
