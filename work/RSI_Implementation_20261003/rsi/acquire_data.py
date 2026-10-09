"""Resumable acquisition from the documented official RSI data sources."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
import time
import zipfile
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path('/home/featurize/rsi_data')
LOG_LOCK = threading.Lock()


def digest(path: Path, algorithm='sha256') -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def event(root: Path, **record):
    record['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with LOG_LOCK:
        with (root / 'downloads.jsonl').open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def download(url: str, target: Path, *, root=ROOT, checksum=None, expected_size=None, attempts=8):
    target.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    if target.is_file() and (expected_size is None or target.stat().st_size == expected_size):
        if checksum is None or digest(target, checksum.split(':')[0]) == checksum.split(':')[1]:
            return target
    part = target.with_suffix(target.suffix + '.part')
    for attempt in range(attempts):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                range_total = response.headers.get('Content-Range', '').split('/')[-1]
                if response.status_code == 416 and not range_total.isdigit():
                    head = requests.head(url, timeout=60); head.raise_for_status()
                    range_total = head.headers.get('Content-Length', '')
                if response.status_code == 416 and offset and range_total.isdigit() and offset == int(range_total):
                    pass
                else:
                    response.raise_for_status()
                    resume = offset and response.status_code == 206
                    response_length = response.headers.get('Content-Length')
                    before = offset if resume else 0
                    with part.open('ab' if resume else 'wb') as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                    if response_length and part.stat().st_size - before != int(response_length):
                        raise ValueError('HTTP content length mismatch')
            if expected_size is not None and part.stat().st_size != expected_size:
                raise ValueError(f'size mismatch: {part.stat().st_size} != {expected_size}')
            if checksum and digest(part, checksum.split(':')[0]) != checksum.split(':')[1]:
                raise ValueError('official checksum mismatch')
            part.replace(target)
            event(root, status='complete', url=url, path=str(target), bytes=target.stat().st_size,
                  sha256=digest(target), official_checksum=checksum)
            return target
        except Exception as exc:
            event(root, status='retry', url=url, path=str(target), attempt=attempt + 1, error=str(exc))
            if attempt == attempts - 1:
                raise
            time.sleep(min(2 ** attempt, 30))


def acquire_zenodo(root: Path, workers=4):
    response = requests.get('https://zenodo.org/api/records/14201693', timeout=60)
    response.raise_for_status()
    record = response.json()
    meta = root / 'ima/metadata'; meta.mkdir(parents=True, exist_ok=True)
    (meta / 'zenodo_record.json').write_text(json.dumps(record, indent=2))
    # CSVs are first so the protocol can be reconstructed while the ZIP downloads.
    files = sorted(record['files'], key=lambda f: f['key'].endswith('.zip'))
    def get(f):
        target = root / 'ima' / ('downloads' if f['key'].endswith('.zip') else 'metadata') / f['key']
        if f['key'].endswith('.zip'):
            result = multipart_download(f['links']['self'], target, f['size'], f['checksum'], root)
        else:
            result = download(f['links']['self'], target, root=root, checksum=f['checksum'], expected_size=f['size'])
        print('downloaded', result.name, result.stat().st_size, flush=True)
    with ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(get, f) for f in files]):
            future.result()
    archive = root / 'ima/downloads/segs.zip'
    masks = root / 'ima/masks'; masks.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            name = Path(info.filename)
            if name.is_absolute() or '..' in name.parts: raise ValueError('unsafe ZIP path')
            destination = masks / name
            if not destination.is_file() or destination.stat().st_size != info.file_size:
                z.extract(info, masks)
    print('IMA masks extracted', sum(p.is_file() for p in masks.rglob('*')), flush=True)


def multipart_download(url, target, size, checksum, root, workers=16, chunk_size=2*1024*1024):
    """Parallel verified HTTP ranges; preserves old partial bytes and resumes pieces."""
    if target.is_file() and target.stat().st_size==size and digest(target,'md5')==checksum.split(':')[1]:
        return target
    parts=target.parent/(target.name+'.parts');parts.mkdir(parents=True,exist_ok=True)
    old=target.with_suffix(target.suffix+'.part')
    ranges=[(n,start,min(start+chunk_size,size)-1) for n,start in enumerate(range(0,size,chunk_size))]
    if old.is_file():
        with old.open('rb') as handle:
            for n,start,end in ranges:
                if end >= old.stat().st_size:break
                destination=parts/f'{n:04d}.bin'
                if not destination.is_file():
                    handle.seek(start);destination.write_bytes(handle.read(end-start+1))
    def get(triple):
        n,start,end=triple;destination=parts/f'{n:04d}.bin';length=end-start+1
        if destination.is_file() and destination.stat().st_size==length:return n
        for attempt in range(8):
            try:
                with requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,120)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                        raise ValueError('server did not return requested exact range')
                    temp=destination.with_suffix('.part')
                    with temp.open('wb') as handle:
                        for chunk in response.iter_content(262144):handle.write(chunk)
                if temp.stat().st_size!=length:raise ValueError('range length mismatch')
                temp.replace(destination);return n
            except Exception as exc:
                event(root,status='range_retry',url=url,range_start=start,range_end=end,attempt=attempt+1,error=str(exc))
                if attempt==7:raise
                time.sleep(min(2**attempt,30))
    with ThreadPoolExecutor(workers) as pool:
        for count,future in enumerate(as_completed([pool.submit(get,r) for r in ranges]),1):
            future.result()
            if count%5==0 or count==len(ranges):print('ZIP ranges',count,'/',len(ranges),flush=True)
    assembled=target.with_suffix('.zip.assembled')
    with assembled.open('wb') as out:
        for n,start,end in ranges:
            with (parts/f'{n:04d}.bin').open('rb') as handle:
                for block in iter(lambda:handle.read(1048576),b''):out.write(block)
    if assembled.stat().st_size!=size or digest(assembled,'md5')!=checksum.split(':')[1]:
        raise ValueError('assembled archive failed official MD5')
    assembled.replace(target)
    event(root,status='complete',url=url,path=str(target),bytes=size,sha256=digest(target),official_checksum=checksum)
    return target


def acquire_rgb(root: Path, manifest: Path, workers=6, limit=None, splits=None):
    rows = list(csv.DictReader(manifest.open()))
    if splits: rows=[r for r in rows if r['split'] in splits]
    tag='_'.join(splits) if splits else 'all'
    split_order = {'train': 0, 'val': 1, 'test': 2}
    ids = sorted({r['image_id']: r['split'] for r in rows}.items(), key=lambda x: (split_order[x[1]], x[0]))
    if limit: ids = ids[:limit]
    api_dir = root / 'ima/api_metadata'; api_dir.mkdir(parents=True, exist_ok=True)
    def get(pair):
        image_id, split = pair
        info_path = api_dir / f'{image_id}.json'
        if info_path.is_file(): info = json.loads(info_path.read_text())
        else:
            # Frozen Zenodo clinical/identity metadata remains authoritative for this
            # protocol. The official S3 full-image URL pattern is verified by v2 API
            # responses cached for the initial 107 images; avoid thousands of API
            # calls (rate-limited) when downloading the same public image objects.
            info = {'isic_id': image_id, 'files': {'full': {
                'url': f'https://isic-archive.s3.amazonaws.com/images/{image_id}.jpg'}},
                'download_url_provenance': 'official v2 API full.url pattern; clinical metadata from frozen Zenodo CSV'}
            tmp = info_path.with_suffix('.json.part'); tmp.write_text(json.dumps(info)); tmp.replace(info_path)
        if info.get('isic_id') != image_id: raise ValueError('ISIC identity mismatch')
        f = info['files']['full']
        # ISIC API file size can be stale relative to the current official S3 object.
        # Keep the API metadata and validate downloaded image decoding in file audit.
        download(f['url'], root / 'ima/images' / f'{image_id}.jpg', root=root)
        return split
    failures = []
    counts = {}
    with ThreadPoolExecutor(workers) as pool:
        fs = {pool.submit(get, pair): pair for pair in ids}
        for n, future in enumerate(as_completed(fs), 1):
            try:
                split = future.result(); counts[split] = counts.get(split, 0) + 1
            except Exception as exc:
                failures.append({'image_id': fs[future][0], 'error': str(exc)})
                event(root, status='rgb_failed',image_id=fs[future][0],error=str(exc))
            if n % 25 == 0 or n == len(fs):
                status = {'completed': counts, 'failed': len(failures), 'processed': n, 'total': len(fs)}
                (root / f'ima/rgb_download_status_{tag}.json').write_text(json.dumps(status, indent=2))
                print(status, flush=True)
    (root / f'ima/rgb_download_failures_{tag}.json').write_text(json.dumps(failures, indent=2))
    if failures: raise RuntimeError(f'{len(failures)} RGB downloads failed; rerun to resume')


def recover_complete_prefix_masks(root: Path, manifest: Path):
    """Recover complete local ZIP entries for T0; validate each official mask MD5.

    This does not mark the full archive or the full data protocol accepted.
    """
    wanted={r['seg_filename']:r['mask_md5'] for r in csv.DictReader(manifest.open())}
    archive_part=root/'ima/downloads/segs.zip.part'
    data=archive_part.read_bytes();pos=0;completed=0;scanned=0
    masks=root/'ima/masks';masks.mkdir(exist_ok=True)
    while pos+30<=len(data):
        values=struct.unpack_from('<I5H3I2H',data,pos)
        signature,version,flag,method,mtime,mdate,crc,csize,usize,nlength,elength=values
        if signature!=0x04034b50:break
        name=data[pos+30:pos+30+nlength].decode('utf-8');start=pos+30+nlength+elength
        if method!=8:break
        decoder=zlib.decompressobj(-15)
        try: payload=decoder.decompress(data[start:])
        except zlib.error:break
        if not decoder.eof:break
        used=len(data)-start-len(decoder.unused_data);end=start+used
        if flag&8:
            if end+16>len(data):break
            if data[end:end+4]==b'PK\x07\x08':end+=4
            expected_crc,expected_csize,expected_usize=struct.unpack_from('<III',data,end);end+=12
            if expected_csize!=used or expected_usize!=len(payload) or expected_crc!=zlib.crc32(payload):
                raise ValueError('ZIP local entry descriptor checksum mismatch')
        else:
            if csize!=used or usize!=len(payload) or crc!=zlib.crc32(payload):raise ValueError('ZIP local entry checksum mismatch')
        scanned+=1
        if name in wanted:
            if hashlib.md5(payload).hexdigest()!=wanted[name]:raise ValueError('Recovered mask failed official CSV MD5')
            destination=masks/Path(name).name
            if not destination.exists():destination.write_bytes(payload)
            completed+=1
        pos=end
    event(root,status='prefix_masks_recovered',archive_prefix=str(archive_part),entries_scanned=scanned,
          selected_masks_validated=completed,full_archive_checksum_complete=False)
    print({'prefix_entries_scanned':scanned,'selected_masks_official_md5_verified':completed},flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['zenodo', 'rgb', 'prefix-masks'])
    p.add_argument('--root', type=Path, default=ROOT); p.add_argument('--manifest', type=Path)
    p.add_argument('--workers', type=int, default=6); p.add_argument('--limit', type=int)
    p.add_argument('--splits',nargs='+',choices=['train','val','test'])
    a = p.parse_args()
    if a.action == 'zenodo': acquire_zenodo(a.root, a.workers)
    elif a.action=='rgb': acquire_rgb(a.root, a.manifest, a.workers, a.limit,a.splits)
    else: recover_complete_prefix_masks(a.root,a.manifest)


if __name__ == '__main__': main()
