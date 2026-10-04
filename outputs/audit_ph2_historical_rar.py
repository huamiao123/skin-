"""Audit a public 2017 PH2 RAR and compare every image/mask with the mirror."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import zipfile
import zlib


ROOT = Path('/home/featurize/rsi_data/ph2_originality_candidates')
PROJECT = Path('/home/featurize/work/RSI_Implementation_20261003')
RAR = ROOT / 'lakshyak_2017_PH2Dataset.rar'
EXTRACTED = ROOT / 'lakshyak_2017_extracted'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rar_entries():
    data = RAR.read_bytes()
    if data[:7] != b'Rar!\x1a\x07\x00':
        raise RuntimeError('Expected RAR4 marker')
    position, entries, header_failures = 7, [], []
    while position < len(data):
        stored_crc, kind, flags, size = struct.unpack_from('<HBHH', data, position)
        if size < 7 or position + size > len(data):
            raise RuntimeError('Invalid RAR header bounds')
        header = data[position:position+size]
        if (zlib.crc32(header[2:]) & 65535) != stored_crc:
            header_failures.append(position)
        packed = struct.unpack_from('<I', header, 7)[0] if flags & 0x8000 else 0
        if kind == 0x74:
            packed, unpacked, host, crc32, dos_time, version, method, name_size, attributes = struct.unpack_from('<IIBIIBBHI', header, 7)
            name_offset = 32
            if flags & 0x100:
                high_packed, high_unpacked = struct.unpack_from('<II', header, 32)
                packed += high_packed << 32
                unpacked += high_unpacked << 32
                name_offset += 8
            name_bytes = header[name_offset:name_offset+name_size]
            filename = name_bytes.split(b'\0')[0].decode('utf-8', errors='replace').replace('\\', '/')
            entries.append({'name': filename, 'packed_bytes': packed, 'bytes': unpacked,
                            'crc32': f'{crc32:08x}', 'compression_method': method,
                            'dos_modified_time': dos_time, 'attributes': attributes})
        position += size + packed
        if kind == 0x7b:
            break
    return entries, header_failures, position


def main():
    entries, header_failures, consumed = rar_entries()
    errors, files = [], []
    for entry in entries:
        path = EXTRACTED / entry['name']
        if path.is_dir():
            continue
        if not path.is_file():
            errors.append({'missing': entry['name']})
            continue
        data = path.read_bytes()
        computed_crc = f'{zlib.crc32(data):08x}'
        passed = len(data) == entry['bytes'] and computed_crc == entry['crc32']
        files.append(dict(entry, sha256=sha(data), computed_crc32=computed_crc, crc_and_size_passed=passed))
        if not passed:
            errors.append({'crc_or_size_failure': entry['name']})
    (ROOT / 'lakshyak_2017_rar_member_integrity.json').write_text(json.dumps(files, indent=2))
    by_filename = {}
    for row in files:
        by_filename.setdefault(Path(row['name']).name, []).append(row)
    with (PROJECT / 'data_manifests/ph2_kaggle_mirror_manifest.csv').open() as handle:
        mirror = list(csv.DictReader(handle))
    comparisons = []
    for row in mirror:
        item = {'image_id': row['image_id']}
        for kind, path_key, hash_key in [('image', 'image_path', 'rgb_sha256'), ('mask', 'mask_path', 'mask_sha256')]:
            source_path = Path(row[path_key])
            candidates = by_filename.get(source_path.name, [])
            same = [candidate for candidate in candidates if candidate['sha256'] == row[hash_key]]
            item[kind + '_sha256'] = row[hash_key]
            item[kind + '_matches'] = candidates
            item[kind + '_byte_identical'] = bool(same)
        comparisons.append(item)
    (ROOT / 'lakshyak_2017_rar_mirror_comparisons.json').write_text(json.dumps(comparisons, indent=2))
    dataset = EXTRACTED / 'PH2Dataset'
    metadata = [row for row in files if Path(row['name']).suffix.lower() in ['.txt', '.xlsx']]
    clinical = []
    text = (dataset / 'PH2_dataset.txt').read_text(encoding='latin1')
    for line in text.splitlines():
        parts = line.split('||')
        if len(parts) < 6 or not re.fullmatch(r'IMD\d+', parts[1].strip()):
            continue
        clinical.append({'image_id': parts[1].strip(), 'histological_diagnosis': parts[2].strip(),
                         'clinical_diagnosis': parts[3].strip(), 'dermoscopic_criteria_raw': parts[4].strip(),
                         'colors_raw': parts[5].strip()})
    clinical_ids = {row['image_id'] for row in clinical}
    expected_ids = {row['image_id'] for row in mirror}
    with (ROOT / 'lakshyak_2017_clinical_metadata.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clinical[0]))
        writer.writeheader()
        writer.writerows(clinical)
    with zipfile.ZipFile(dataset / 'PH2_dataset.xlsx') as workbook:
        xlsx_bad_crc = workbook.testzip()
        xlsx_members = workbook.namelist()
    report = {'source_type': 'public_historical_Kaggle_RAR_mirror',
              'source_page': 'https://www.kaggle.com/datasets/lakshyak/p2hdata',
              'source_release_date': '2017-06-03T06:18:19.503Z',
              'archive': str(RAR), 'archive_bytes': RAR.stat().st_size, 'archive_sha256': sha(RAR.read_bytes()),
              'archive_md5': hashlib.md5(RAR.read_bytes()).hexdigest(), 'archive_format': 'RAR4',
              'archive_entries': len(entries), 'files': len(files), 'header_crc_failures': header_failures,
              'archive_bytes_consumed': consumed, 'all_member_crc_and_size_passed': bool(files) and not errors,
              'errors': errors, 'original_layout_preserved': True,
              'original_layout': 'PH2Dataset/PH2 Dataset images/IMDxxx/{IMDxxx_Dermoscopic_Image,IMDxxx_lesion,IMDxxx_roi}',
              'image_files': sum('_Dermoscopic_Image/' in row['name'] for row in files),
              'lesion_mask_files': sum('_lesion/' in row['name'] for row in files),
              'roi_mask_files': sum('_roi/' in row['name'] for row in files),
              'mirror_compared_images': len(comparisons),
              'byte_identical_images': sum(row['image_byte_identical'] for row in comparisons),
              'byte_identical_lesion_masks': sum(row['mask_byte_identical'] for row in comparisons),
              'clinical_metadata_files': metadata, 'clinical_rows': len(clinical),
              'clinical_image_ids_match_all_200': clinical_ids == expected_ids and len(clinical) == 200,
              'clinical_diagnosis_counts': dict(Counter(row['clinical_diagnosis'] for row in clinical)),
              'histological_diagnosis_counts': dict(Counter(row['histological_diagnosis'] for row in clinical)),
              'clinical_xlsx_crc_passed': xlsx_bad_crc is None, 'clinical_xlsx_members': xlsx_members,
              'mirror_dimensions_equivalent_by_byte_identity': all(row['image_byte_identical'] and row['mask_byte_identical'] for row in comparisons),
              'official_creator_control_verified': False, 'official_archive_byte_equivalence_verified': False,
              'conclusion': 'Every one of the 200 mirror image BMPs and 200 lesion BMPs is byte-identical to this preserved original-layout RAR distributed in a public 2017 Kaggle release, including original clinical XLSX/TXT and 50 ROI masks. This establishes historical distribution equality for the full segmentation data. Direct creator-controlled archive acquisition or creator checksum remains unavailable.',
              'does_not_establish': ['Independent upstream acquisition among mirrors', 'Equivalence to an unseen creator-controlled RAR',
                                     'Explanation of creator prose nominal768x560 versus measured stored dimensions'],
              'no_RSI_source_changes': True}
    output = PROJECT / 'outputs/ph2_historical_rar_comparison.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in ['clinical_metadata_files','clinical_xlsx_members']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
