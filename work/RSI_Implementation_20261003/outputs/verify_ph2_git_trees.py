"""Compare local PH2 member bytes with pinned public Git tree blob hashes."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor

import requests


ROOT = Path('/home/featurize/rsi_data/ph2_originality_candidates')
PROJECT = Path('/home/featurize/work/RSI_Implementation_20261003')
REPOSITORIES = [
    ('pranavm0610/MENAS', '89e4e7e9d4e986c0d89120e7d1396ce38cb3ab82'),
    ('wahyuhwwkwk/116UAS_KecerdasanBuatan', '77db7eddd935e1e17718aae4a4ba18f0fea90ff4'),
]


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest(), len(data)


def fetch_and_compare(spec, expected):
    repo, commit = spec
    session = requests.Session()
    session.headers.update({'User-Agent': 'RSI-public-data-provenance-audit', 'Accept': 'application/vnd.github+json'})
    url = f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'
    response = session.get(url, timeout=(20, 45))
    result = {'repository': repo, 'source_commit': commit, 'tree_url': url,
              'status': response.status_code, 'official_creator_control_verified': False}
    if response.status_code != 200:
        result['error'] = response.text[:500]
        return result
    tree = response.json()
    snapshot = ROOT / (repo.replace('/', '__') + '_' + commit + '_tree.json')
    snapshot.write_text(json.dumps(tree, indent=2))
    result.update({'tree_sha': tree['sha'], 'truncated': tree.get('truncated', False),
                   'tree_entries': len(tree['tree']), 'tree_snapshot': str(snapshot),
                   'tree_snapshot_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                   'github_remaining_public_api_requests': response.headers.get('X-RateLimit-Remaining')})
    blobs = [item for item in tree['tree'] if item['type'] == 'blob']
    relevant = [item for item in blobs if re.search(r'/IMD\d+_Dermoscopic_Image/IMD\d+\.bmp$', item['path'])
                or re.search(r'/IMD\d+_lesion/IMD\d+_lesion\.bmp$', item['path'])]
    lookup = {}
    for item in relevant:
        lookup.setdefault(Path(item['path']).name, []).append(item)
    comparisons = []
    for filename, source in sorted(expected.items()):
        matches = lookup.get(filename, [])
        exact = [item for item in matches if item['sha'] == source['git_blob_sha1'] and item.get('size') == source['bytes']]
        comparisons.append({'filename': filename, 'kind': source['kind'],
                            'local_git_blob_sha1': source['git_blob_sha1'], 'local_bytes': source['bytes'],
                            'repository_matches': matches, 'same_blob_and_size': bool(exact)})
    comparison_path = ROOT / (repo.replace('/', '__') + '_' + commit + '_comparisons.json')
    comparison_path.write_text(json.dumps(comparisons, indent=2))
    result.update({'expected_images': sum(item['kind'] == 'image' for item in comparisons),
                   'expected_lesion_masks': sum(item['kind'] == 'mask' for item in comparisons),
                   'same_image_blobs': sum(item['kind'] == 'image' and item['same_blob_and_size'] for item in comparisons),
                   'same_mask_blobs': sum(item['kind'] == 'mask' and item['same_blob_and_size'] for item in comparisons),
                   'missing_or_different': [item for item in comparisons if not item['same_blob_and_size']],
                   'comparison_details': str(comparison_path),
                   'clinical_files': [item for item in blobs if Path(item['path']).suffix.lower() in ['.xls', '.xlsx', '.csv'] and 'ph2' in item['path'].lower()],
                   'archive_files': [item for item in blobs if Path(item['path']).suffix.lower() in ['.rar', '.zip'] and 'ph2' in item['path'].lower()]})
    commit_response = session.get(f'https://api.github.com/repos/{repo}/commits/{commit}', timeout=(20, 30))
    if commit_response.status_code == 200:
        data = commit_response.json()
        snapshot = ROOT / (repo.replace('/', '__') + '_' + commit + '_commit.json')
        snapshot.write_text(json.dumps(data, indent=2))
        result['commit_author_date'] = data['commit']['author']['date']
        result['commit_committer_date'] = data['commit']['committer']['date']
    return result


def main():
    ROOT.mkdir(exist_ok=True)
    manifest = PROJECT / 'data_manifests/ph2_kaggle_mirror_manifest.csv'
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    expected = {}
    for row in rows:
        for kind, column in [('image', 'image_path'), ('mask', 'mask_path')]:
            path = Path(row[column])
            blob, size = git_blob(path)
            expected[path.name] = {'git_blob_sha1': blob, 'bytes': size, 'kind': kind}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda spec: fetch_and_compare(spec, expected), REPOSITORIES))
    result = {'local_manifest': str(manifest), 'local_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'comparison': "Git blob SHA1 of bytes: SHA1(b'blob ' + decimal_length + b'\\0' + member_bytes), plus file byte length",
              'repositories': results, 'official_archive_equivalence_verified': False,
              'limitation': 'Pinned third-party repository trees prove equality to those historical distributed BMP blobs. They do not prove repository independence or equivalence to an unseen creator-controlled RAR.'}
    output = PROJECT / 'outputs/ph2_full_git_blob_comparison.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
