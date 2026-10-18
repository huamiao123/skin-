"""Official ISIC2016 images/masks plus pinned MedVLSM P3 annotations."""
from common import *
import requests, zipfile, time, re
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np
from PIL import Image
from datetime import datetime, timezone

RAW = Path('/home/featurize/rsi_data/benchmarks/isic2016')
ANN = PROJECT / 'text_stage_preparation/isic_text_audit/medvlsm'
BASE = 'https://isic-archive.s3.amazonaws.com/challenges/2016/'
FILES = ['ISBI2016_ISIC_Part1_' + x + '.zip' for x in
         ('Training_Data', 'Training_GroundTruth', 'Test_Data', 'Test_GroundTruth')]

def download_extract(name):
    RAW.mkdir(parents=True, exist_ok=True)
    meta = RAW / (name + '.json')
    if meta.exists():
        old = json.loads(meta.read_text())
        if all((RAW / f['member']).is_file() for f in old['members']): return old
    path = RAW / name; tmp = RAW / (name + '.part'); url = BASE + name
    if not path.exists():
        for attempt in range(4):
            try:
                with requests.get(url, stream=True, timeout=(20, 90)) as r:
                    r.raise_for_status(); total = int(r.headers.get('Content-Length', 0)); size = 0
                    with tmp.open('wb') as f:
                        for chunk in r.iter_content(4*1024**2):
                            f.write(chunk); size += len(chunk)
                    assert not total or size == total, (name, size, total)
                tmp.replace(path); break
            except Exception:
                if attempt == 3: raise
                time.sleep(2 * (attempt + 1))
    result = dict(url=url, archive_bytes=path.stat().st_size, archive_sha256=sha(path), members=[])
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for info in z.infolist():
            if info.is_dir() or not info.filename.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            target = RAW / info.filename
            assert target.resolve().is_relative_to(RAW.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            blob = z.read(info)
            if target.exists(): assert target.read_bytes() == blob
            else: target.write_bytes(blob)
            result['members'].append(dict(member=info.filename, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
    dump(meta, result)
    path.unlink()  # Verified extracted data and hashes retained; avoid duplicate archive storage.
    print('OFFICIAL_DATA_READY', name, len(result['members']), flush=True)
    return result

def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as pool: sources = list(pool.map(download_extract, FILES))
    index = defaultdict(list)
    for source in sources:
        for m in source['members']:
            p = RAW / m['member']; match = re.search(r'ISIC_\d+', p.name)
            if match: index[match.group()].append(p)
    originals = {s: json.loads((ANN / f'{s}.json').read_text()) for s in SPLITS}
    prepared, removed = {}, []
    # Preserve test first, then validation; delete overlapping/duplicate training records.
    seen_id, seen_hash, seen_resized = {}, {}, {}
    for split in ('test', 'val', 'train'):
        prepared[split] = []
        for source_row, row in enumerate(originals[split]):
            sid = row['segment_id']; paths = index[sid]
            images = [p for p in paths if p.suffix.lower() in ('.jpg', '.jpeg')]
            masks = [p for p in paths if p.suffix.lower() == '.png' and 'segmentation' in p.name.lower()]
            assert images and masks, (split, sid, paths)
            # Prefer the member from the official split when duplicate archive members exist.
            key = 'Test' if split == 'test' else 'Training'
            image_path = next((p for p in images if key in str(p)), images[0])
            mask_path = next((p for p in masks if key in str(p)), masks[0])
            image = Image.open(image_path).convert('RGB'); mask = Image.open(mask_path).convert('L')
            assert image.size == mask.size, sid
            decoded = hashlib.sha256(str(image.size).encode() + np.asarray(image).tobytes()).hexdigest()
            resized = hashlib.sha256(np.asarray(image.resize((256,256), Image.Resampling.BILINEAR)).tobytes()).hexdigest()
            duplicate = seen_id.get(sid) or seen_hash.get(decoded) or seen_resized.get(resized)
            if duplicate:
                removed.append(dict(split=split, source_row=source_row, id=sid, retained=duplicate,
                    reason='same ISIC ID or exact decoded image content (native/256 square)'))
                continue
            kept = dict(split=split, id=sid)
            seen_id[sid] = seen_hash[decoded] = seen_resized[resized] = kept
            prompt = row['prompts']['p3']; assert isinstance(prompt, str) and 'skin melanoma' in prompt
            text = prompt.replace('skin melanoma', 'skin lesion')
            assert text and 'located' not in text and not any(word in text.split() for word in ['small','medium','large'])
            w,h=image.size; ratio=min(256/w,256/h); nw,nh=max(1,round(w*ratio)),max(1,round(h*ratio))
            prepared[split].append(dict(id=sid, sample_id=sid, image_path=str(image_path), mask_path=str(mask_path),
                image_sha256=sha(image_path), mask_sha256=sha(mask_path), decoded_image_sha256=decoded,
                resized_image_sha256=resized, source_annotation_row=source_row, prompt_original=prompt, text=text,
                width=w,height=h,new_width=nw,new_height=nh,top=(256-nh)//2,left=(256-nw)//2))
    # Cache only 256 inputs; original image/GT paths remain the evaluation authority.
    allrows = []; images256=[]; masks256=[]
    for split in SPLITS:
        for row in prepared[split]:
            row['index']=len(allrows); allrows.append(row)
            x=np.empty((256,256,3),np.uint8);x[:]=np.array([124,116,104],np.uint8)
            y=np.zeros((256,256),np.uint8)
            t,l,h,w=[row[k] for k in ('top','left','new_height','new_width')]
            x[t:t+h,l:l+w]=np.asarray(Image.open(row['image_path']).convert('RGB').resize((w,h),Image.Resampling.BILINEAR))
            m=np.asarray(Image.open(row['mask_path']).convert('L'));assert set(np.unique(m)).issubset({0,255}),row['id']
            assert m.any(),row['id']
            y[t:t+h,l:l+w]=np.asarray(Image.fromarray(m).resize((w,h),Image.Resampling.NEAREST))>=128
            images256.append(x);masks256.append(y)
        dump(ASSETS/f'{split}.json',prepared[split])
    np.save(ASSETS/'images256.npy',np.stack(images256));np.save(ASSETS/'masks256.npy',np.stack(masks256))
    dump(ASSETS/'all_rows.json',allrows)
    audit=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),dataset='ISIC2016 + MedVLSM P3',
        source_commit='dc36a3af83035efa344c8ce26397528ad173569e',raw_counts={s:len(originals[s]) for s in SPLITS},
        counts={s:len(prepared[s]) for s in SPLITS},removed=removed,official_archives=sources,
        annotations={s:dict(path=str(ANN/f'{s}.json'),sha256=sha(ANN/f'{s}.json')) for s in SPLITS},
        members={str(p.relative_to(ROOT)):sha(p) for p in ASSETS.glob('*') if p.is_file()},
        prompt='P3 (VQA color + shape); replace uniform class string skin melanoma with skin lesion in all splits',
        excluded_prompt_fields='bbox, GT-derived size/count/location (P4-P6/P8-P9) not model inputs',
        exact_image_split_overlap=0,patient_disjointness='not established',
        initialized_from='generic pretrained encoders only, no previous ISIC/RefCOCO-trained decoder checkpoint')
    dump(ROOT/'data_manifest.json',audit)
    print('DATA_READY',json.dumps(audit['counts']), 'removed',len(removed),flush=True)

if __name__=='__main__':main()
