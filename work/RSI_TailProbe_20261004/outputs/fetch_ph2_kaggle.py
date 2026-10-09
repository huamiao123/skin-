"""Acquire a public, paper-cited PH2 mirror without claiming official provenance.

No study source modules or historical datasets are changed by this helper.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import requests


ROOT = Path('/home/featurize/rsi_data/ph2_kaggle_mirror')
API = 'https://www.kaggle.com/api/v1/datasets/download/athina123/ph2dataset'
META_API = 'https://www.kaggle.com/api/v1/datasets/view/athina123/ph2dataset'
ZIP = ROOT / 'ph2dataset_kaggle.zip'
PART = Path(str(ZIP) + '.part')


def dump(name, obj):
    path = ROOT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(temp, path)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = 'RSI-data-provenance-audit/1.0'
    try:
        response = session.get(META_API, timeout=(20, 30))
        response.raise_for_status()
        metadata = response.json()
        dump('kaggle_metadata.json', metadata)
        print(json.dumps({'metadata_saved': True, 'keys': list(metadata)}, ensure_ascii=False), flush=True)
    except Exception as exc:
        print('metadata_request: ' + str(exc), flush=True)
    started = time.time()
    while not ZIP.exists():
        offset = PART.stat().st_size if PART.exists() else 0
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        try:
            with session.get(API, headers=headers, stream=True, timeout=(20, 45)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError('Resume request was not honored; keep existing prefix for investigation')
                if offset:
                    content_range = response.headers.get('Content-Range', '')
                    if not content_range.startswith(f'bytes {offset}-'):
                        raise RuntimeError('Unexpected Content-Range: ' + content_range)
                    expected = int(content_range.rsplit('/', 1)[1])
                else:
                    expected = int(response.headers['Content-Length'])
                destination = urlsplit(response.url)
                provenance = {
                    'source_type': 'public_Kaggle_mirror', 'official_distribution_verified': False,
                    'dataset_page': 'https://www.kaggle.com/datasets/athina123/ph2dataset',
                    'download_endpoint': API, 'metadata_endpoint': META_API,
                    'download_host': destination.netloc, 'download_path': destination.path,
                    'expected_http_bytes': expected, 'http_status': response.status_code,
                    'etag': response.headers.get('ETag'),
                    'last_modified': response.headers.get('Last-Modified'),
                    'primary_paper_availability': [
                        'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0344622',
                        'https://www.nature.com/articles/s41598-026-43127-1'],
                    'official_package_hash_comparison': 'not_available',
                    'acquired_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                }
                dump('download_provenance.json', provenance)
                print(json.dumps({'http': response.status_code, 'expected_bytes': expected, 'resume_bytes': offset}), flush=True)
                last_report = time.time()
                with PART.open('ab' if offset else 'wb') as handle:
                    for chunk in response.iter_content(65536):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        offset += len(chunk)
                        if time.time() - last_report >= 15:
                            print(json.dumps({'downloaded_bytes': offset, 'percent': round(100*offset/expected, 2),
                                              'elapsed_seconds': round(time.time()-started, 1)}), flush=True)
                            last_report = time.time()
                if PART.stat().st_size != expected:
                    raise RuntimeError(f'Wrong size {PART.stat().st_size} != {expected}')
                os.replace(PART, ZIP)
                sha = hashlib.sha256()
                with ZIP.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024*1024), b''):
                        sha.update(chunk)
                provenance.update({'complete': True, 'zip_sha256': sha.hexdigest(),
                                   'actual_bytes': ZIP.stat().st_size, 'elapsed_seconds': time.time()-started})
                dump('download_provenance.json', provenance)
                print(json.dumps({'complete': True, 'zip_sha256': sha.hexdigest(), 'bytes': ZIP.stat().st_size}), flush=True)
        except Exception as exc:
            print('download_retry: ' + str(exc), flush=True)
            time.sleep(3)


if __name__ == '__main__':
    main()
