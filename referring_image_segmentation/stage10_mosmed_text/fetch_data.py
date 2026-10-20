"""Pinned public image release and small public mirrors of the LViT text tables."""
from common import *
import requests, zipfile, io, time
from urllib.parse import quote

IMAGE_REF = 'maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset'
TEXT_REF = 'xiaopan9802/textdiff-mosmeddata'

def fetch(url):
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=(20, 90)); r.raise_for_status(); return r.content
        except requests.RequestException:
            if attempt == 3: raise
            time.sleep(2 * (attempt + 1))

def main():
    sources = ROOT / 'sources'; sources.mkdir(exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    api = 'https://www.kaggle.com/api/v1/datasets/'
    for ref, target in ((IMAGE_REF, 'kaggle_metadata.json'), (TEXT_REF, 'text_mirror_metadata.json')):
        if not (sources / target).exists(): (sources / target).write_bytes(fetch(api + 'view/' + ref))
    audit_path = RAW / 'download_audit.json'
    if not audit_path.exists():
        url = api + 'download/' + IMAGE_REF + '?datasetVersionNumber=2'
        path = RAW / 'source_v2.zip'; tmp = path.with_suffix('.part')
        if not path.exists():
            with requests.get(url, stream=True, timeout=(20, 90)) as r:
                r.raise_for_status()
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(4 * 1024**2): f.write(chunk)
            tmp.replace(path)
        record = dict(url=url, archive_bytes=path.stat().st_size, archive_sha256=sha(path), members=[])
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for info in z.infolist():
                if info.is_dir(): continue
                dest = RAW / info.filename; assert dest.resolve().is_relative_to(RAW.resolve())
                dest.parent.mkdir(parents=True, exist_ok=True); blob = z.read(info)
                dest.write_bytes(blob)
                record['members'].append(dict(path=info.filename, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
        dump(audit_path, record); path.unlink()
    audit = json.loads(audit_path.read_text())
    assert all((RAW / r['path']).is_file() for r in audit['members'])
    texts = {}
    for split in SPLITS:
        name = split.title() + '_text_MosMedData.xlsx'
        member = 'MosMedData/TextDiff_code/datasets/Covid19/' + split + '/' + name
        url = api + 'download/' + TEXT_REF + '/' + quote(member, safe='') + '?datasetVersionNumber=1'
        path = sources / name
        if not path.exists():
            blob = fetch(url)
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                if '[Content_Types].xml' not in z.namelist():
                    assert len(z.namelist()) == 1; blob = z.read(z.namelist()[0])
            path.write_bytes(blob)
        with zipfile.ZipFile(path) as z: assert 'xl/workbook.xml' in z.namelist()
        texts[split] = dict(url=url, path=str(path), sha256=sha(path), bytes=path.stat().st_size)
    dump(ROOT / 'sources/download_provenance.json', dict(images=audit, texts=texts,
        original_author_repository='https://github.com/HUANGLIZI/LViT',
        text_delivery='Public third-party mirror; author OneDrive and Google Drive unavailable from this environment',
        original_author_binary_identity_verified=False,
        note='LViT-style tables; expected split counts and image IDs checked separately. No text is generated from local masks.'))
    print('FETCH_READY', len(audit['members']), flush=True)

if __name__ == '__main__': main()
