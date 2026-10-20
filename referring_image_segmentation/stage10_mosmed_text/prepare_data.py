"""Audit the original split, repair cross-volume overlap, freeze exact image/text pairs."""
from common import *
import re, zipfile, collections, xml.etree.ElementTree as ET
from datetime import datetime, timezone
import numpy as np
from PIL import Image

NS = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
PRIORITY = {'train': 0, 'val': 1, 'test': 2}

def read_xlsx(path):
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('x:si', NS)]
        rows = []
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//x:row', NS):
            cells = {}
            for cell in row.findall('x:c', NS):
                value = cell.find('x:v', NS); value = value.text if value is not None else ''
                if cell.get('t') == 's': value = shared[int(value)]
                elif cell.get('t') == 'inlineStr': value = ''.join(cell.find('x:is', NS).itertext())
                cells[re.sub('[0-9]', '', cell.get('r'))] = value
            rows.append(cells)
    assert rows[0].get('A') == 'Image' and rows[0].get('B') == 'text'
    return [dict(id=r['A'], text=r['B']) for r in rows[1:] if r.get('A')]

def source_group(name):
    if name.startswith('bjorke_'):
        return 'ambiguous_bjorke_source_bucket', 'patient identifiers unavailable; entire source held together'
    m = re.match(r'Jun_radiopaedia_\d+_(\d+)_', name)
    if m: return 'radiopaedia_patient_' + m.group(1), 'radiopaedia case identifier; series grouped together'
    group = re.sub(r'_\d+\.png$', '', name)
    assert group != name and (group.startswith('Jun_coronacases_case') or group.startswith('Morozov_study_')), name
    return group, 'source CT volume identifier'

def main():
    ASSETS.mkdir(exist_ok=True)
    provenance = json.loads((ROOT / 'sources/download_provenance.json').read_text())
    originals = {s: read_xlsx(provenance['texts'][s]['path']) for s in SPLITS}
    assert {s: len(v) for s, v in originals.items()} == dict(train=2183, val=273, test=273)
    images = {p.name: p for p in (RAW / 'frames').glob('*.png')}
    masks = {p.name: p for p in (RAW / 'masks').glob('*.png')}
    assert len(images) == len(masks) == 2729, (len(images), len(masks))
    parent = {}; group_splits = collections.defaultdict(set); records = []
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)
    seen_native, seen_input = {}, {}
    for split in SPLITS:
        for item in originals[split]:
            name = item['id']; assert name in images and name in masks, name
            image = Image.open(images[name]).convert('RGB'); mask = Image.open(masks[name]).convert('L')
            assert image.size == mask.size, name
            pixels = np.asarray(image); gt = np.asarray(mask)
            values = np.unique(gt)
            assert values.min() >= 0 and values.max() <= 255, name
            group, reason = source_group(name); find(group); group_splits[group].add(split)
            w, h = image.size; ratio = min(256 / w, 256 / h); nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
            decoded = hashlib.sha256(str(image.size).encode() + pixels.tobytes()).hexdigest()
            input_hash = hashlib.sha256(np.asarray(image.resize((nw, nh), Image.Resampling.BILINEAR)).tobytes()).hexdigest()
            for lookup, digest in ((seen_native, decoded), (seen_input, input_hash)):
                if digest in lookup: union(group, lookup[digest])
                else: lookup[digest] = group
            records.append(dict(**item, original_split=split, original_group=group, group_basis=reason,
                image_path=str(images[name]), mask_path=str(masks[name]), image_sha256=sha(images[name]), mask_sha256=sha(masks[name]),
                decoded_image_sha256=decoded, input_image_sha256=input_hash,
                width=w, height=h, new_width=nw, new_height=nh, top=(256-nh)//2, left=(256-nw)//2,
                gt_pixels=int((gt > 0).sum()), native_mask_values=values.tolist(),
                image_channel_equal=bool(np.array_equal(pixels[:,:,0], pixels[:,:,1]) and np.array_equal(pixels[:,:,1], pixels[:,:,2]))))
    assert len({r['id'] for r in records}) == 2729
    component_splits = collections.defaultdict(set)
    for r in records: component_splits[find(r['original_group'])].add(r['original_split'])
    group_assignment = {g: max(ss, key=PRIORITY.get) for g, ss in component_splits.items()}
    removed, kept, seen_ids, seen_native, seen_input = [], [], set(), {}, {}
    for r in sorted(records, key=lambda r: (-PRIORITY[r['original_split']], r['id'])):
        r['group_id'] = find(r['original_group']); r['split'] = group_assignment[r['group_id']]
        duplicate = seen_native.get(r['decoded_image_sha256']) or seen_input.get(r['input_image_sha256'])
        if duplicate:
            removed.append(dict(id=r['id'], retained=duplicate, reason='exact decoded native/input image duplicate; group merged before splitting'))
            continue
        assert r['id'] not in seen_ids; seen_ids.add(r['id'])
        seen_native[r['decoded_image_sha256']] = seen_input[r['input_image_sha256']] = r['id']; kept.append(r)
    allrows, inputs, targets = [], [], []
    for split in SPLITS:
        rows = sorted([r for r in kept if r['split'] == split], key=lambda r: r['id'])
        for r in rows:
            r['index'] = len(allrows); allrows.append(r)
            x = np.empty((256,256,3), np.uint8); x[:] = [124,116,104]; y = np.zeros((256,256), np.uint8)
            t,l,h,w = [r[k] for k in ('top','left','new_height','new_width')]
            x[t:t+h,l:l+w] = np.asarray(Image.open(r['image_path']).convert('RGB').resize((w,h), Image.Resampling.BILINEAR))
            y[t:t+h,l:l+w] = np.asarray(Image.open(r['mask_path']).convert('L').resize((w,h), Image.Resampling.NEAREST)) > 0
            inputs.append(x); targets.append(y)
        dump(ASSETS / f'{split}.json', rows)
    np.save(ASSETS / 'images256.npy', np.stack(inputs)); np.save(ASSETS / 'masks256.npy', np.stack(targets))
    dump(ASSETS / 'all_rows.json', allrows)
    groups = {s: {r['group_id'] for r in allrows if r['split'] == s} for s in SPLITS}
    assert all(not (groups[a] & groups[b]) for a,b in (('train','val'),('train','test'),('val','test')))
    moved = [dict(id=r['id'], source=r['original_split'], destination=r['split'], group_id=r['group_id']) for r in allrows if r['original_split'] != r['split']]
    audit = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), dataset='MosMedData+ (LViT-style text annotation)',
        original_counts={s: len(v) for s,v in originals.items()}, counts={s: sum(r['split']==s for r in allrows) for s in SPLITS},
        group_counts={s: len(v) for s,v in groups.items()}, original_cross_split_groups={g:sorted(ss) for g,ss in group_splits.items() if len(ss)>1},
        group_assignment=group_assignment, reassigned=moved, removed_duplicates=removed,
        split_policy='Whole source volume/patient groups; exact image duplicates join groups; assign each group test > val > train based on original memberships. All ambiguous bjorke slices are held together.',
        group_overlap_after_repair=0, patient_disjointness='Traceable source groups disjoint; complete cross-source patient identity and bjorke patient identity not established',
        unique_texts=len({r['text'] for r in allrows}), empty_masks={s:sum(r['gt_pixels']==0 and r['split']==s for r in allrows) for s in SPLITS},
        grayscale_images=sum(r['image_channel_equal'] for r in allrows),
        mask_binarization='native grayscale > 0, matching author LViT Load_Dataset.py; nearest-neighbor geometric resizing',
        masks_with_non_0_255_values=sum(not set(r['native_mask_values']).issubset({0,255}) for r in allrows),
        source_provenance_sha256=sha(ROOT/'sources/download_provenance.json'),
        text_caveat='LViT paper states professional annotations include lesion number/location and mask-based quality checks. Public mirror not bitwise authenticated against author original. Extra annotation supplied at inference; not ordinary report-only clinical validation.',
        members={str(p.relative_to(ROOT)):sha(p) for p in ASSETS.glob('*') if p.is_file()})
    dump(ROOT/'data_manifest.json', audit)
    print('DATA_READY', json.dumps({k:audit[k] for k in ('counts','group_counts','unique_texts','empty_masks')}), 'moved',len(moved),'removed',len(removed),flush=True)

if __name__ == '__main__': main()
