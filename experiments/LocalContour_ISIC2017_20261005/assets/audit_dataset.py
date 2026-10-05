"""Read-only official ISIC 2017 train/val asset audit; never open test data."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
SOURCE = Path('/home/featurize/work/RSI_Implementation_20261003/data_manifests')
DATA = Path('/home/featurize/rsi_data/benchmarks/isic2017')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def audit_row(task):
    row, member_images, member_masks = task
    out = dict(row)
    errors = []
    for key, expected in [('image_path', member_images), ('mask_path', member_masks)]:
        path = Path(row[key])
        raw = path.read_bytes()
        meta = expected[path.name]
        current_sha = hashlib.sha256(raw).hexdigest()
        current_crc = f'{zlib.crc32(raw) & 0xffffffff:08x}'
        ref_key = 'rgb_sha256' if key == 'image_path' else 'mask_sha256'
        if current_sha != row[ref_key]:
            errors.append(f'{key}: historical SHA-256 mismatch')
        if len(raw) != meta['uncompressed_bytes'] or current_crc != meta['zip_crc32']:
            errors.append(f'{key}: official archive member byte-count/CRC mismatch')
        out['current_' + ref_key] = current_sha
        out['current_' + key.replace('_path', '_crc32')] = current_crc
        with Image.open(io.BytesIO(raw)) as im:
            if list(im.size) != [int(row['width']), int(row['height'])]:
                errors.append(f'{key}: dimensions mismatch')
            if row['split'] == 'val':
                array = np.array(im.convert('RGB' if key == 'image_path' else 'L'))
                psha = hashlib.sha256(array.tobytes()).hexdigest()
                pkey = 'rgb_pixel_sha256' if key == 'image_path' else 'mask_pixel_sha256'
                if psha != row[pkey]:
                    errors.append(f'{key}: decoded pixel SHA-256 mismatch')
                out['current_' + pkey] = psha
                if key == 'mask_path':
                    values = np.unique(array).tolist()
                    out['current_mask_values'] = json.dumps(values)
                    if not set(values).issubset({0, 255}):
                        errors.append('mask nonbinary')
    out['current_decode_validation'] = 'full_rgb_and_mask_pixels' if row['split'] == 'val' else 'headers_and_whole_file_integrity'
    out['errors'] = json.dumps(errors, ensure_ascii=False)
    return out

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = SOURCE / 'external_isic2017_manifest.csv'
    with manifest.open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] in ('train', 'val')]
    rows.sort(key=lambda r: (r['split'], r['image_id']))
    errors = []
    members = {}
    source_hashes = {str(manifest): sha(manifest)}
    for split, expected_n in [('train', 2000), ('val', 150)]:
        for kind in ('images', 'masks'):
            p = SOURCE / f'external_isic2017_{split}_{kind}_members.json'
            source_hashes[str(p)] = sha(p)
            obj = json.loads(p.read_text())
            (OUT / p.name).write_text(json.dumps(obj, indent=2) + '\n')
            members[split, kind] = obj
            actual = {p.name for p in (DATA / split / kind).iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
            if actual != set(obj) or len(obj) != expected_n:
                errors.append({'split': split, 'kind': kind, 'missing': sorted(set(obj) - actual), 'extra': sorted(actual - set(obj)), 'actual': len(actual), 'official_members': len(obj)})
        split_rows = [r for r in rows if r['split'] == split]
        if len(split_rows) != expected_n or len({r['image_id'] for r in split_rows}) != expected_n:
            errors.append({'split': split, 'manifest_count_error': len(split_rows)})
    tasks = [(r, members[r['split'], 'images'], members[r['split'], 'masks']) for r in rows]
    checked = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, r in enumerate(pool.map(audit_row, tasks), 1):
            checked.append(r)
            if json.loads(r['errors']):
                errors.append({'image_id': r['image_id'], 'split': r['split'], 'errors': json.loads(r['errors'])})
            if i % 250 == 0 or i == len(rows):
                print(f'asset audit {i}/{len(rows)} image/mask pairs; errors={len(errors)}', flush=True)
    fieldnames = list(dict.fromkeys(k for r in checked for k in r))
    for split in ('train', 'val'):
        selected = [r for r in checked if r['split'] == split]
        path = OUT / f'isic2017_{split}_asset_manifest.csv'
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(selected)
    val = sorted((r for r in checked if r['split'] == 'val'), key=lambda r: r['image_id'])
    order = np.random.default_rng(17).permutation(len(val))
    calibration = {val[int(i)]['image_id'] for i in order[:50]}
    with (OUT / 'val_split_seed17.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'role', 'permutation_rank', 'image_path', 'mask_path', 'width', 'height'])
        w.writeheader()
        for rank, index in enumerate(order):
            r = val[int(index)]
            w.writerow({k: r[k] for k in ['image_id', 'image_path', 'mask_path', 'width', 'height']} | {'role': 'calibration50' if r['image_id'] in calibration else 'locked_verification100', 'permutation_rank': rank})
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS' if not errors else 'FAIL', 'dataset': 'ISIC2017 Task1 official split', 'data_root': str(DATA), 'counts': {'train': 2000, 'val': 150}, 'test_image_or_mask_files_opened': 0, 'test_model_scoring_performed': False, 'full_current_file_SHA_and_official_CRC_checked_pairs': len(checked), 'full_decoded_RGB_and_mask_pixels_checked_val_pairs': len(val), 'train_decode_scope': 'PIL headers plus whole-file SHA/official-member CRC; historical complete decoding is preserved by identical SHA', 'seed': 17, 'split_algorithm': 'lexicographically sorted official val IDs; numpy.random.default_rng(17).permutation(150); first 50 calibration, final 100 locked verification', 'calibration_count': 50, 'locked_verification_count': 100, 'source_manifest_and_member_sha256': source_hashes, 'errors': errors, 'checkpoint_selection_history': 'No compliant 2017 CNN checkpoint has been identified; therefore prior use of official val for such checkpoint selection is unknown and cannot be certified independent.', 'independent_test_claim_allowed': False, 'prepared_split_is_unused': True, 'data_mixing': 'No ISIC2018, IMA, PH2 or other years added to probe manifests', 'output_sha256': {str(p): sha(p) for p in OUT.iterdir() if p.is_file() and (p.suffix == '.csv' or 'members.json' in p.name)}}
    (OUT / 'data_audit.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: summary[k] for k in ['status', 'counts', 'calibration_count', 'locked_verification_count', 'errors']}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
