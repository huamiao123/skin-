"""Operational HTTP range acquisition outside frozen research source."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests


def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['isic2018','ph2']);a=p.parse_args()
    base=Path('/home/featurize/rsi_data/benchmarks')
    if a.kind=='isic2018':
        url='https://s3.amazonaws.com/isic-archive/challenges/2018/ISIC2018_Task1-2_Training_Input.zip';size=11165358566
        target=base/'isic2018/downloads/ISIC2018_Task1-2_Training_Input.zip';etag='"1808495b9cdce381f83f9a3ed1419c4f-213"';md5=None;max_workers=64
    else:
        url='https://zenodo.org/records/17498821/files/PH2.zip?download=1';size=424765398
        target=base/'ph2_mirror/downloads/PH2.zip';etag=None;md5='a37f6a83c777ca861f055fdf49757991';max_workers=16
    partial=target.with_suffix('.zip.part');offset=partial.stat().st_size if partial.exists() else 0
    parts=target.parent/(target.name+'.range1m');parts.mkdir(parents=True,exist_ok=True)
    statepath=Path(f'outputs/{a.kind}_parallel_range_status.json')
    lock=threading.RLock();cooldown=0
    state={'pid':os.getpid(),'state':'downloading','source_url':url,'expected_bytes':size,'retained_prefix_bytes':offset,'retained_prefix_sha256':digest(partial) if offset else None,'http_etag':etag,'repository_md5':md5,'range_chunk_bytes':1048576,'completed_range_bytes':0,'range_bytes_in_flight':0,'failures':[],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'completed_ranges':[],'phases':[]}
    def save():
        with lock:
            state['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());tmp=statepath.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(statepath)
    save()
    triples=[(start,min(start+1048576,size)-1) for start in range(offset,size,1048576)]
    def get(pair):
        nonlocal cooldown
        start,end=pair;path=parts/f'{start}-{end}.bin';temp=path.with_suffix('.part');length=end-start+1
        if path.exists() and path.stat().st_size==length:return {'start':start,'end':end,'bytes':length,'sha256':digest(path),'cached':True}
        for attempt in range(8):
            try:
                while time.monotonic()<cooldown:time.sleep(min(cooldown-time.monotonic(),1))
                have=temp.stat().st_size if temp.exists() else 0
                if have>length:raise ValueError('oversized partial range')
                if have==length:temp.replace(path);return {'start':start,'end':end,'bytes':length,'sha256':digest(path),'cached':False}
                first=start+have;headers={'Range':f'bytes={first}-{end}','Accept-Encoding':'identity'}
                if etag:headers['If-Match']=etag
                with requests.get(url,headers=headers,stream=True,timeout=(20,120)) as r:
                    if r.status_code==429:
                        delay=int(r.headers.get('Retry-After','60'))
                        with lock:cooldown=max(cooldown,time.monotonic()+delay)
                        raise requests.HTTPError(f'HTTP429; global cooldown {delay}s')
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {first}-{end}/{size}':raise ValueError('incorrect exact Content-Range')
                    if etag and r.headers.get('ETag')!=etag:raise ValueError('source ETag changed')
                    with temp.open('ab') as out:
                        for b in r.iter_content(65536):
                            if b:out.write(b)
                if temp.stat().st_size!=length:raise ValueError('range length mismatch')
                temp.replace(path)
                return {'start':start,'end':end,'bytes':length,'sha256':digest(path),'cached':False}
            except Exception as e:
                with lock:state['failures'].append({'start':start,'end':end,'attempt':attempt+1,'error':repr(e)});save()
                print('RANGE RETRY',start,end,repr(e),flush=True)
                if attempt==7:raise
                time.sleep(min(2**attempt,8))
    todo=list(triples);workers=8;global_start=time.monotonic()
    while todo:
        count=len(todo) if workers==max_workers else min(len(todo),workers)
        batch=todo[:count];todo=todo[count:];phase_start=time.monotonic();last=phase_start;completed=0
        print('PHASE',workers,'ranges',count,flush=True)
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            fs={pool.submit(get,pair):pair for pair in batch}
            while fs:
                done,_=concurrent.futures.wait(fs,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    pair=fs.pop(f);record=f.result();state['completed_ranges'].append(record);state['completed_range_bytes']+=record['bytes'];completed+=record['bytes']
                now=time.monotonic()
                if done or now-last>=20:
                    active_bytes=sum((parts/f'{s}-{e}.part').stat().st_size for s,e in fs.values() if (parts/f'{s}-{e}.part').exists())
                    state.update(active_workers=workers,completed_range_count=len(state['completed_ranges']),total_range_count=len(triples),range_bytes_in_flight=active_bytes,average_mib_per_second=round((state['completed_range_bytes']+active_bytes)/(now-global_start)/1048576,3))
                    save()
                    if now-last>=20:print('PROGRESS',state['completed_range_count'],'/',len(triples),'bytes',state['completed_range_bytes']+active_bytes,'MiB/s',state['average_mib_per_second'],flush=True);last=now
        elapsed=time.monotonic()-phase_start;rate=completed/elapsed/1048576
        state['phases'].append({'workers':workers,'bytes':completed,'seconds':round(elapsed,3),'mib_per_second':round(rate,3)});save();print('PHASE DONE',workers,'MiB/s',round(rate,3),flush=True)
        workers=min(workers*2,max_workers)
    state['state']='assembling';save()
    # Preserve the original prefix, append verified ordered ranges, and check full length.
    with partial.open('ab') as out:
        for start,end in triples:
            path=parts/f'{start}-{end}.bin'
            with path.open('rb') as f:
                for b in iter(lambda:f.read(1048576),b''):out.write(b)
    if partial.stat().st_size!=size:raise ValueError('assembled length mismatch')
    if md5 and digest(partial,'md5')!=md5:raise ValueError('mirror-published MD5 mismatch')
    partial.replace(target);state.update(state='downloaded',bytes=size,local_sha256=digest(target),repository_md5_verified=(digest(target,'md5')==md5) if md5 else None);save()
    if a.kind=='isic2018':
        from rsi.acquire_external import extract_zip,save_status
        outer=json.loads(Path('outputs/external_data_status.json').read_text());job=next(j for j in outer['sources'] if j['dataset']=='ISIC2018' and j['split']=='train' and j['kind']=='images')
        job.update(state='downloaded',bytes=size,local_sha256=state['local_sha256'],download_method='Exact ranges; preserved prefix; ETag matched; no published independent source hash',get_http_status=206,resolved_url=url);save_status(outer)
        extract_zip(job,target,base,outer);state['state']='extracted';save()
    print('DONE',json.dumps({k:v for k,v in state.items() if k not in ['completed_ranges','failures']}),flush=True)


if __name__=='__main__':main()
