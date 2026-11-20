import hashlib,subprocess,tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
d=Path('/home/waas/mv_branch_sm120_20261120');d.mkdir(exist_ok=True)
origin='https://developer.download.nvidia.cn/compute/cuda/redist/'
catalog=requests.get(origin+'redistrib_12.8.1.json',timeout=20);catalog.raise_for_status()
items=catalog.json()
pypi=requests.get('https://pypi.org/pypi/mamba-ssm/1.2.2/json',timeout=20);pypi.raise_for_status()
sd=next(x for x in pypi.json()['urls'] if x['packagetype']=='sdist')
def get(x):
    url,digest,target,strip=x
    f=d/url.split('/')[-1];target.mkdir(parents=True,exist_ok=True)
    if not f.exists() or hashlib.file_digest(f.open('rb'),'sha256').hexdigest()!=digest:
        r=requests.get(url,timeout=(15,90),stream=True);r.raise_for_status()
        with f.open('wb') as out:
            for chunk in r.iter_content(1024*1024):out.write(chunk)
    assert hashlib.file_digest(f.open('rb'),'sha256').hexdigest()==digest
    subprocess.run(['tar','xf',str(f),'-C',str(target),'--strip-components='+str(strip)],check=True)
    print('ready',f.name,flush=True)
jobs=[]
for key in ('cuda_nvcc','cuda_cudart','cuda_cccl','cuda_nvrtc'):
    if key in items:
        item=items[key]['linux-x86_64']; jobs.append((origin+item['relative_path'],item['sha256'],d/'cuda',1))
jobs.append((sd['url'],sd['digests']['sha256'],d/'mamba_1_2_2',1))
with ThreadPoolExecutor(5) as pool:list(pool.map(get,jobs))

