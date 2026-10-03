"""Rebuild the preregistered P2-IMA-M-v2 selections from original CSVs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/featurize/rsi_data')
FIELDS = ['protocol_id', 'subset', 'image_id', 'reference_id', 'annotator', 'tool', 'skill_level',
          'mskObjectID', 'mask_md5', 'seg_filename', 'split', 'original_split', 'group_id',
          'patient_id', 'lesion_id', 'image_path', 'mask_path', 'metadata_width', 'metadata_height',
          'copyright_license', 'attribution']


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_references(rows, subset):
    # Source filtering MUST precede annotator deduplication for each subset.
    selected = defaultdict(dict)
    for row in rows:
        if row['annotator'] not in {f'A{i:02d}' for i in range(16)}:
            continue
        if subset in {'H', 'T1'} and row['skill_level'] != 'S1':
            continue
        if subset == 'H' and row['tool'] not in {'T1', 'T2'}:
            continue
        if subset == 'T1' and row['tool'] != 'T1':
            continue
        image_id, annotator = row['ISIC_id'], row['annotator']
        priority = ({'T1': 0, 'T2': 1, 'T3': 2}.get(row['tool'], 99), row['mskObjectID'])
        old = selected[image_id].get(annotator)
        if old is None or priority < old[0]: selected[image_id][annotator] = (priority, row)
    return {image_id: [entry[1] for _, entry in sorted(annotators.items())]
            for image_id, annotators in selected.items() if len(annotators) >= 2}


def known_groups(images, duplicate_groups=None):
    parent = {row['isic_id']: row['isic_id'] for row in images}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a,b)] = min(a,b)
    seen = {}
    missing = {'', 'unknown', 'nan', 'none', 'null', 'na'}
    for row in images:
        for field in ['patient_id', 'lesion_id']:
            value = row.get(field, '').strip()
            if value.lower() in missing: continue
            key = (field, value)
            if key in seen: union(row['isic_id'], seen[key])
            else: seen[key] = row['isic_id']
    for group in duplicate_groups or []:
        for image_id in group[1:]:
            if group[0] not in parent or image_id not in parent:raise ValueError('duplicate group ID absent from frozen metadata')
            union(group[0],image_id)
    return {k: find(k) for k in parent}


def reconstruct(root: Path, output: Path):
    meta = root / 'ima/metadata'
    images = read_csv(meta / 'img_metadata.csv'); image_metadata = {r['isic_id']: r for r in images}
    raw_refs = read_csv(meta / 'seg_metadata.csv')
    duplicate_path=output/'confirmed_duplicate_groups.json'
    duplicate_audit=json.loads(duplicate_path.read_text()) if duplicate_path.is_file() else {}
    groups = known_groups(images,duplicate_audit.get('groups',[]))
    split_map = {}
    for split in ['train', 'val', 'test']:
        for row in read_csv(meta / f'{split}.csv'):
            image_id = Path(row['image']).stem
            if image_id in split_map: raise ValueError(f'duplicate original split ID: {image_id}')
            split_map[image_id] = split
    selections = {s: select_references(raw_refs, s) for s in ['M', 'H', 'T1']}
    before = {s: dict(Counter(split_map[i] for i in selection if i in split_map)) for s, selection in selections.items()}
    rank = {'train': 0, 'val': 1, 'test': 2}
    protection = {}
    for image_id in selections['M']:
        if image_id in split_map:
            group = groups[image_id]; split = split_map[image_id]
            if rank[split] > rank.get(protection.get(group), -1): protection[group] = split
    retained = {i: split_map[i] for i in selections['M'] if i in split_map and split_map[i] == protection[groups[i]]}
    removed = [{'image_id': i, 'original_split': split_map[i], 'group_id': groups[i],
                'protected_split': protection[groups[i]], 'reason': 'known_case_group_split_conflict'}
               for i in sorted(selections['M']) if i in split_map and i not in retained]
    mask_index = {p.name: p for p in (root / 'ima/masks').rglob('*.png')}
    rows = []
    counts = {}; ref_counts = {}
    for subset, selected in selections.items():
        counts[subset] = dict(Counter(retained[i] for i in selected if i in retained))
        ref_counts[subset] = dict(Counter(retained[i] for i in selected if i in retained for _ in selected[i]))
        for image_id in sorted(selected):
            if image_id not in retained: continue
            im = image_metadata[image_id]
            for ref in selected[image_id]:
                rows.append(dict(protocol_id='P2-IMA-M-v2', subset=subset, image_id=image_id,
                    reference_id=ref['mskObjectID'], annotator=ref['annotator'], tool=ref['tool'],
                    skill_level=ref['skill_level'], mskObjectID=ref['mskObjectID'], mask_md5=ref['mask_md5'],
                    seg_filename=ref['seg_filename'], split=retained[image_id], original_split=split_map[image_id],
                    group_id=groups[image_id], patient_id=im.get('patient_id', ''), lesion_id=im.get('lesion_id', ''),
                    image_path=str(root / 'ima/images' / f'{image_id}.jpg'),
                    mask_path=str(mask_index.get(ref['seg_filename'], root / 'ima/masks' / ref['seg_filename'])),
                    metadata_width=im.get('pixels_x', ''), metadata_height=im.get('pixels_y', ''),
                    copyright_license=im.get('copyright_license',''), attribution=im.get('attribution','')))
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'selected_references.csv'; write_csv(target, rows, FIELDS)
    write_csv(output / 'removed_images.csv', removed, ['image_id','original_split','group_id','protected_split','reason'])
    group_rows = [dict(image_id=i, group_id=groups[i], patient_id=image_metadata[i].get('patient_id',''),
                       lesion_id=image_metadata[i].get('lesion_id',''), original_split=split_map.get(i,''),
                       retained_split=retained.get(i,'')) for i in sorted(groups)]
    write_csv(output / 'split_groups.csv', group_rows,
              ['image_id','group_id','patient_id','lesion_id','original_split','retained_split'])
    overlap = {}
    split_sets = {s: {groups[i] for i,v in retained.items() if v==s} for s in rank}
    for a,b in [('train','val'),('train','test'),('val','test')]: overlap[f'{a}:{b}'] = sorted(split_sets[a]&split_sets[b])
    report = {'protocol_id':'P2-IMA-M-v2','image_metadata_rows':len(images),'reference_metadata_rows':len(raw_refs),
              'source_sha256':{p.name:file_sha(p) for p in sorted(meta.glob('*.csv'))},
              'rules':{'original_annotators':'A00-A15','subset_filter_before_dedup':True,
                       'dedup_priority':['T1','T2','T3','mskObjectID_lexicographic'],
                       'grouping':'patient_id OR lesion_id transitive components across all image metadata, plus documented verified RGB duplicates or conservatively suspected same-case links',
                       'protection_order':['test','val','train'],'missing_identity':'not merged'},
              'before_group_cleanup':before,'image_counts':counts,'reference_counts':ref_counts,
              'identity_coverage_all_images':{field:sum(bool(r.get(field,'')) for r in images) for field in ['patient_id','lesion_id']},
              'identity_coverage_retained_M':{field:sum(bool(image_metadata[i].get(field,'')) for i in retained) for field in ['patient_id','lesion_id']},
              'known_group_count_all_images':len(set(groups.values())),
              'verified_rgb_duplicate_group_source_sha256':file_sha(duplicate_path) if duplicate_path.is_file() else None,
              'verified_rgb_duplicate_groups':duplicate_audit.get('groups',[]),
              'identity_limit':'image deduplication and known patient/lesion isolation; missing identities prevent a complete patient-independent guarantee',
              'removed_counts':dict(Counter(r['original_split'] for r in removed)),
              'known_group_split_intersection':overlap,'selected_references_sha256':file_sha(target),
              'status':'metadata_candidates_only; requires RGB/mask file audit'}
    write_json(output / 'metadata_protocol_audit.json',report)
    print(json.dumps(report, indent=2))
    return rows, report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=Path('data_manifests'));a=p.parse_args()
    reconstruct(a.root,a.output)


if __name__ == '__main__': main()
