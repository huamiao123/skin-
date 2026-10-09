"""Operational downloader; experiment rsi sources remain frozen.

Use plain HTTP 200 GET when the network path serves ranged requests slowly.
Existing retained bytes are verified against the prefix streamed from source.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests


def sha(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['isic2018','ph2']);a=p.parse_args()
    base=Path('/home/featurize/rsi_data/benchmarks')
    if a.kind=='isic2018':
        url='https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip'
        target=base/'isic2018/downloads/ISIC2018_Task1-2_Training_Input.zip';size=11165358566;md5=None
    else:
        url='https://zenodo.org/api/records/17498821/files/PH2.zip/content'
        target=base/'ph2_mirror/downloads/PH2.zip';size=424765398;md5='a37f6a83c777ca861f055fdf49757991'
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_suffix('.zip.part')
    state=Path(f'outputs/{a.kind}_full_get_status.json')
    status={'pid':os.getpid(),'source_url':url,'expected_bytes':size,'method':'plain HTTP GET; retained prefix hash verified before appending','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'state':'starting'}
    def save():
        status['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        tmp=state.with_suffix('.json.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(state)
    save()
    if not target.exists():
        for attempt in range(1,6):
            offset=partial.stat().st_size if partial.exists() else 0
            retained_sha=sha(partial) if offset else None
            status.update(state='requesting',attempt=attempt,retained_bytes=offset,retained_prefix_sha256=retained_sha);save()
            try:
                started=time.monotonic();last=started;received=0;prefix=hashlib.sha256();prefix_checked=offset==0
                with requests.get(url,headers={'Accept-Encoding':'identity'},stream=True,timeout=(20,90)) as r:
                    r.raise_for_status()
                    if r.status_code!=200:raise ValueError(f'Expected plain GET200, got {r.status_code}')
                    if int(r.headers.get('Content-Length',size))!=size:raise ValueError('source length changed')
                    status.update(state='streaming',http_status=r.status_code,http_etag=r.headers.get('ETag'),http_content_length=r.headers.get('Content-Length'),final_url=r.url);save()
                    print('HTTP',r.status_code,'retain',offset,'total',size,flush=True)
                    with partial.open('ab') as out:
                        for chunk in r.iter_content(1024*1024):
                            if not chunk:continue
                            if received<offset:
                                prefix_len=min(len(chunk),offset-received);prefix.update(chunk[:prefix_len]);suffix=chunk[prefix_len:]
                                if received+prefix_len==offset:
                                    if prefix.hexdigest()!=retained_sha:raise ValueError('source stream does not match retained prefix')
                                    prefix_checked=True;status['retained_prefix_verified']=True;print('PREFIX VERIFIED',offset,flush=True)
                            else:suffix=chunk
                            if suffix:
                                if not prefix_checked:raise ValueError('prefix must be verified first')
                                out.write(suffix)
                            received+=len(chunk);now=time.monotonic()
                            if now-last>=20:
                                out.flush();status.update(streamed_bytes=received,retained_bytes=partial.stat().st_size,mib_per_second=round(received/(now-started)/1048576,3));save();print('PROGRESS',received,'/',size,'retained',partial.stat().st_size,'MiB/s',status['mib_per_second'],flush=True);last=now
                if received!=size or partial.stat().st_size!=size:raise ValueError('download length mismatch')
                if md5 and sha(partial,'md5')!=md5:raise ValueError('repository MD5 mismatch')
                partial.replace(target);break
            except Exception as e:
                status.update(state='retry',error=repr(e));save();print('RETRY',repr(e),flush=True)
                if attempt==5:raise
        else:raise RuntimeError('download failed')
    status.update(state='downloaded',path=str(target),bytes=target.stat().st_size,local_sha256=sha(target),repository_md5=md5,repository_md5_verified=(sha(target,'md5')==md5) if md5 else None);save()
    if a.kind=='isic2018':
        from rsi.acquire_external import extract_zip,save_status
        outer=json.loads(Path('outputs/external_data_status.json').read_text());job=next(j for j in outer['sources'] if j['dataset']=='ISIC2018' and j['split']=='train' and j['kind']=='images')
        job.update(state='downloaded',bytes=size,local_sha256=status['local_sha256'],download_method=status['method'],get_http_status=200,retained_prefix_verified=status.get('retained_prefix_verified',False));save_status(outer)
        extract_zip(job,target,base,outer);status['state']='extracted';save()
    print('DONE',json.dumps(status),flush=True)


if __name__=='__main__':main()
