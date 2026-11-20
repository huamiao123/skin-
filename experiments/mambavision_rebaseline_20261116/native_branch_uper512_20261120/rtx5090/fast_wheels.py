from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, os, time
import requests
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
target=Path('/home/waas/mv_branch_sm120_20261120/wheels');target.mkdir(exist_ok=True)
priority={t:i for i,t in enumerate(sys_tags())}
def meta(name,v):
    for base in ('https://pypi.org/pypi','https://pypi.tuna.tsinghua.edu.cn/pypi'):
        try:
            r=requests.get(f'{base}/{name}/{v}/json',timeout=(5,16));r.raise_for_status();return r.json()
        except requests.RequestException:continue
    raise RuntimeError(name)
def wheel(name,v):
    j=meta(name,v);candidates=[]
    for w in j['urls']:
        if w['packagetype']!='bdist_wheel':continue
        tags=parse_wheel_filename(w['filename'])[-1]
        scores=[priority[t] for t in tags if t in priority]
        if scores:candidates.append((min(scores),w['filename'],w))
    w=min(candidates)[-1];p=target/w['filename'];dig=w['digests']['sha256']
    if p.exists() and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==dig:return
    for mirror in ('https://pypi.tuna.tsinghua.edu.cn/packages/','https://mirrors.cloud.tencent.com/pypi/packages/','https://files.pythonhosted.org/packages/'):
        try:
            url=w['url'].replace('https://files.pythonhosted.org/packages/',mirror)
            with requests.get(url,stream=True,timeout=(8,35)) as r:
                r.raise_for_status(); h=hashlib.sha256();tmp=p.with_suffix('.incomplete')
                with tmp.open('wb') as f:
                    for b in r.iter_content(2**20): f.write(b);h.update(b)
            assert h.hexdigest()==dig
            os.replace(tmp,p);print('ready',name,v,w['size'],flush=True);return
        except Exception as e:print('retry',name,str(e)[:120],flush=True)
    raise RuntimeError(name)
main=meta('torch','2.8.0')
deps={'torch':'2.8.0','torchvision':'0.23.0'}
for s in main['info']['requires_dist']:
    r=Requirement(s)
    if not (r.marker is None or r.marker.evaluate()):continue
    if r.name.startswith('nvidia') or r.name=='triton':
        deps[r.name]=next(x.version for x in r.specifier if x.operator=='==')
print('collecting',list(deps),flush=True)
with ThreadPoolExecutor(6) as pool:list(pool.map(lambda item:wheel(*item),deps.items()))

