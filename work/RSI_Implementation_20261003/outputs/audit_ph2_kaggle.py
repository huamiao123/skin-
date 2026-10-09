"""Validate and extract untouched BMP bytes from the paper-cited PH2 mirror."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


ROOT = Path('/home/featurize/rsi_data/ph2_kaggle_mirror')
PROJECT = Path('/home/featurize/work/RSI_Implementation_20261003')
ZIP = ROOT / 'ph2dataset_kaggle.zip'
PREFIX = 'ph2_dataset/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    manifest = PROJECT / 'data_manifests/ph2_kaggle_mirror_manifest.csv'
    errors, rows, members, repeated, dimensions = [], [], [], [], Counter()
    provenance = json.loads((ROOT / 'download_provenance.json').read_text())
    with zipfile.ZipFile(ZIP) as archive:
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise RuntimeError('ZIP CRC failure: ' + bad_crc)
        names = set(archive.namelist())
        images = sorted(name for name in names if name.startswith(PREFIX + 'trainx/') and name.endswith('.bmp'))
        masks = sorted(name for name in names if name.startswith(PREFIX + 'trainy/') and name.endswith('_lesion.bmp'))
        expected_mask_names = {PREFIX + 'trainy/' + Path(name).stem + '_lesion.bmp' for name in images}
        if len(images) != 200 or len(masks) != 200 or expected_mask_names != set(masks):
            raise RuntimeError('Expected exactly 200 distinct image/lesion BMP pairs')
        for member in archive.infolist():
            data = archive.read(member)
            members.append({'name': member.filename, 'bytes': len(data), 'zip_crc32': f'{member.CRC:08x}', 'sha256': sha(data)})
        member_hashes = {row['name']: row['sha256'] for row in members}
        for name in sorted(names):
            if name.startswith(PREFIX):
                other = name[len(PREFIX):]
                if other not in names or member_hashes[other] != member_hashes[name]:
                    errors.append({'kind': 'mirror_directory_duplicates_differ', 'member': name})
                else:
                    repeated.append([name, other])
        for name in images:
            image_id = Path(name).stem
            mask_name = PREFIX + 'trainy/' + image_id + '_lesion.bmp'
            image_data, mask_data = archive.read(name), archive.read(mask_name)
            image_path, mask_path = ROOT / 'raw' / name, ROOT / 'raw' / mask_name
            for output, data in [(image_path, image_data), (mask_path, mask_data)]:
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists() and sha(output.read_bytes()) != sha(data):
                    raise RuntimeError('Existing extraction differs: ' + str(output))
                if not output.exists():
                    output.write_bytes(data)
            with Image.open(io.BytesIO(image_data)) as image, Image.open(io.BytesIO(mask_data)) as mask:
                image.load()
                mask.load()
                width, height = image.size
                image_pixels = np.asarray(image.convert('RGB'))
                mask_pixels = np.asarray(mask.convert('L'))
                values = sorted(int(value) for value in np.unique(mask_pixels))
                dimensions_match = image.size == mask.size
                binary = set(values).issubset({0, 255})
                is_bmp = image.format == 'BMP' and mask.format == 'BMP'
                if not dimensions_match or not binary or not is_bmp:
                    errors.append({'kind': 'invalid_pair', 'image_id': image_id, 'dimensions_match': dimensions_match,
                                   'mask_values': values, 'image_format': image.format, 'mask_format': mask.format})
                dimensions[f'{width}x{height}'] += 1
                row = {'dataset': 'PH2_public_Kaggle_mirror', 'split': 'external_all', 'image_id': image_id,
                       'image_path': str(image_path), 'mask_path': str(mask_path), 'width': width, 'height': height,
                       'rgb_sha256': sha(image_data), 'rgb_pixel_sha256': sha(image_pixels.tobytes()),
                       'mask_sha256': sha(mask_data), 'mask_pixel_sha256': sha(mask_pixels.tobytes()),
                       'mask_mode': mask.mode, 'mask_values': json.dumps(values), 'mask_empty': not bool(mask_pixels.any()),
                       'decode_passed': True, 'dimensions_match': dimensions_match, 'mask_binary': binary,
                       'image_original_mode': image.mode, 'image_format': image.format, 'mask_format': mask.format,
                       'image_zip_member': name, 'mask_zip_member': mask_name,
                       'patient_id': '', 'lesion_id': '', 'group_id': 'PH2:' + image_id,
                       'source_type': 'public_Kaggle_mirror', 'official_distribution_verified': False,
                       'source_url': provenance['dataset_page'], 'dataset_version': 1,
                       'zip_sha256': provenance['zip_sha256'], 'audit_complete': dimensions_match and binary and is_bmp}
                rows.append(row)
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(ROOT / 'zip_member_integrity.json', members)
    primary_manifest = PROJECT / 'data_manifests/final_references.csv'
    with primary_manifest.open() as handle:
        primary_rows = list(csv.DictReader(handle))
    primary_file, primary_pixel = {}, {}
    for row in primary_rows:
        primary_file.setdefault(row['image_sha256'], set()).add(row['image_id'])
        key = (row['image_pixel_sha256'], int(row['original_w']), int(row['original_h']))
        primary_pixel.setdefault(key, set()).add(row['image_id'])
    overlap = []
    for row in rows:
        for method, matches in [('file_sha256', primary_file.get(row['rgb_sha256'], set())),
                                ('decoded_rgb_pixel_sha256_and_dimensions', primary_pixel.get((row['rgb_pixel_sha256'], row['width'], row['height']), set()))]:
            for image_id in sorted(matches):
                overlap.append({'ph2_image_id': row['image_id'], 'ima_image_id': image_id, 'method': method})
    audit = {'source_type': 'public_Kaggle_mirror', 'official_distribution_verified': False,
             'official_package_hash_comparison': 'not_available', 'zip_crc_complete': True,
             'zip_members': len(members), 'zip_sha256': provenance['zip_sha256'],
             'zip_bytes': ZIP.stat().st_size, 'zip_member_bytes': sum(row['bytes'] for row in members),
             'duplicate_directory_copy_pairs': len(repeated), 'all_duplicate_directory_copies_byte_identical': len(repeated) == 400 and not errors,
             'selected_pairs': len(rows), 'images': len(rows), 'lesion_masks': len(rows),
             'dimensions': dict(dimensions), 'all_images_and_masks_BMP': all(row['image_format'] == row['mask_format'] == 'BMP' for row in rows),
             'all_masks_binary': all(row['mask_binary'] for row in rows), 'all_dimensions_match': all(row['dimensions_match'] for row in rows),
             'all_decoded': all(row['decode_passed'] for row in rows), 'empty_masks': sum(row['mask_empty'] for row in rows),
             'uniform_256_preprocessing_detected': set(dimensions) == {'256x256'},
             'extraction_preserves_exact_archive_member_bytes': True, 'mirror_folder_names_are_not_experiment_splits': True,
             'patient_identity_metadata_available': False, 'clinical_classification_metadata_available': False,
             'originality_limitation': 'BMP bytes retained exactly at their stored, nonuniform dimensions. No official package checksum or authors original-file comparison is available, so original acquisition dimensions and official-source equivalence remain unverified.',
             'mirror_license_metadata': 'Unknown', 'errors': errors, 'file_audit_complete': len(rows) == 200 and not errors,
             'ima_final_manifest_sha256': sha(primary_manifest.read_bytes()), 'ima_exact_overlap': overlap,
             'manifest': str(manifest), 'manifest_sha256': sha(manifest.read_bytes()),
             'member_integrity': str(ROOT / 'zip_member_integrity.json')}
    dump(PROJECT / 'data_manifests/ph2_kaggle_mirror_audit.json', audit)
    dump(ROOT / 'ph2_kaggle_mirror_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
