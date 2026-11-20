"""Fetch NVIDIA's pinned ADE20K artifact and retain only the verified decoder state for training."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, os, time, requests, torch
ROOT=Path(__file__).resolve().parent
TMP=Path('/tmp/uper512_ade_decoder_20261119')
TMP.mkdir(parents=True,exist_ok=True)
ID='775e92963709d0118c2a1312908e8fca83a116843158f9e345e44e9af01ebc1e'
URL='https://huggingface.co/nvidia/mamba_vision_160k_ade20k-512x512_small/resolve/c72844cda381341ce06f24cf87a9634c34839736/mamba_vision_160k_ade20k-512x512_small.pth'
SIZE=1008463676
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def download():
    path=TMP/'official_full.pth'
    if path.exists() and sha(path)==ID:return path
    step=16*1024*1024
    work=TMP/'ranges'
    work.mkdir(exist_ok=True)
    def fetch(i):
        lo,hi=i*step,min((i+1)*step,SIZE)-1
        dest=work/str(i)
        if dest.exists() and dest.stat().st_size==hi-lo+1:return dest
        for attempt in range(5):
            try:
                url=URL+f'?chunk={i}&attempt={attempt}'
                with requests.get(url,headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(10,38)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{SIZE}':raise ValueError((r.status_code,r.headers.get('Content-Range')))
                    with dest.with_suffix('.partial').open('wb') as f:
                        for block in r.iter_content(512*1024):f.write(block)
                assert dest.with_suffix('.partial').stat().st_size==hi-lo+1
                dest.with_suffix('.partial').replace(dest)
                return dest
            except Exception as e:
                print('retry',i,attempt,type(e).__name__,str(e).split(' for url:')[0][:150],flush=True)
                if attempt==4:raise
                time.sleep(1+attempt)
    with ThreadPoolExecutor(8) as pool: parts=list(pool.map(fetch,range((SIZE+step-1)//step)))
    import shutil
    with path.with_suffix('.assembling').open('wb') as f:
        for part in parts:
            with part.open('rb') as source:shutil.copyfileobj(source,f,1024*1024)
    assert sha(path.with_suffix('.assembling'))==ID
    path.with_suffix('.assembling').replace(path)
    return path
if __name__=='__main__':
    target=ROOT/'ade20k_decode_head.pth'
    if target.is_file():
        record=__import__('json').loads((ROOT/'checkpoint_source.json').read_text())
        assert record['full_sha256']==ID and sha(target)==record['decoder_sha256']
    else:
        path=download()
        state=torch.load(path,map_location='cpu',weights_only=False)['state_dict']
        head={k.removeprefix('decode_head.'):v for k,v in state.items() if k.startswith('decode_head.')}
        assert head['conv_seg.weight'].shape[:2]==(150,512)
        torch.save(head,target.with_suffix('.partial'))
        target.with_suffix('.partial').replace(target)
        record=dict(url=URL,full_sha256=ID,decoder_sha256=sha(target),tensors=len(head),keys=list(head))
        (ROOT/'checkpoint_source.json').write_text(__import__('json').dumps(record,indent=2))
        print('Official full and decoder verified',record['tensors'],record['decoder_sha256'],flush=True)
