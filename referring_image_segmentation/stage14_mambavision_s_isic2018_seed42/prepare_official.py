"""Restore only the pinned official S model and recorded official Mamba wheel."""
from common import *
import requests

def fetch(url,path,expected):
    if path.exists():assert sha(path)==expected;return
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.part')
    with requests.get(url,stream=True,timeout=(20,90)) as response:
        response.raise_for_status()
        with tmp.open('wb') as output:
            for block in response.iter_content(8*1024**2):output.write(block)
    assert sha(tmp)==expected;tmp.replace(path)

def main():
    record=json.loads((ROOT/'official/source_download.json').read_text())
    assert record['hf_revision']==HF_REVISION
    for row in record['files']:fetch(row['url'],ROOT/'official/hf'/row['path'],row['sha256'])
    row=json.loads((ROOT/'official/weight_download.json').read_text())
    fetch(row['url'],ROOT/'official/hf/model.safetensors',WEIGHT_SHA256)
    api=f'https://huggingface.co/api/models/nvidia/MambaVision-S-1K/revision/{HF_REVISION}?blobs=true'
    response=requests.get(api,timeout=45);response.raise_for_status()
    data=response.json();assert data['sha']==HF_REVISION
    weight=next(x for x in data['siblings'] if x['rfilename']=='model.safetensors')
    assert weight['lfs']['sha256']==WEIGHT_SHA256,weight
    dump(ROOT/'official/remote_weight_identity.json',dict(revision=HF_REVISION,api=api,weight=weight,
        downloaded_sha256=sha(ROOT/'official/hf/model.safetensors'),matches_remote_lfs_sha256=True))
    print('PINNED_OFFICIAL_S_VERIFIED',flush=True)

if __name__=='__main__':main()

