"""Verify this published snapshot without loading checkpoints or using a GPU."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'publication/stage14_19_archive_manifest.json').read_text())
    failures = []
    for item in manifest['files']:
        path = root / item['path']
        if not path.is_file() or path.is_symlink():
            failures.append(item['path'] + ': missing or not a regular file')
        elif path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            failures.append(item['path'] + ': content mismatch')
    actual = sorted(str(p.relative_to(root)) for d in root.glob('stage1[4-9]*')
                    for p in d.rglob('*.pth') if p.is_file())
    expected = sorted(item['path'] for item in manifest['checkpoints'])
    if actual != expected or any(not name.endswith('/best_dice.pth') for name in actual):
        failures.append('Unexpected or missing checkpoint file')
    # Raw pixel arrays, runtime binaries and pretrained checkpoints are not in
    # this public archive. Scalar result JSON/CSV and numeric charts are allowed.
    roots = list(root.glob('stage1[4-9]*')) + [root / 'diagnostics/feature_analysis_20261101']
    forbidden = {'.npy', '.npz', '.pt', '.ckpt', '.safetensors', '.whl', '.so'}
    for folder in roots:
        for path in folder.rglob('*'):
            if path.is_file() and (path.suffix in forbidden or 'examples' in path.parts):
                failures.append(str(path.relative_to(root)) + ': unexpected local-only artifact')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(json.dumps(dict(passed=True, files_checked=len(manifest['files']),
                          best_checkpoints_checked=len(expected)), indent=2))


if __name__ == '__main__':
    main()
