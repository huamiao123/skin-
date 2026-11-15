from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures,zipfile,io
ROOT=Path(__file__).resolve().parent/'official_code'
REPOS=['SLDGroup/EMCAD','DengPingFan/PraNet','plemeri/UACANet','gorkemcanates/Dual-Cross-Attention','UARK-AICV/MEGANet','simzhangbest/HMT-Unet','ai4colonoscopy/PraNet-V2']
def fetch(repo):
 d=ROOT/repo.replace('/','__');d.mkdir(exist_ok=True);errors=[]
 for branch in ['main','master']:
  try:
   url='https://codeload.github.com/'+repo+'/zip/refs/heads/'+branch
   with urllib.request.urlopen(url,timeout=60) as f:data=f.read()
   z=zipfile.ZipFile(io.BytesIO(data));rows=[]
   for info in z.infolist():
    rel=Path(*Path(info.filename).parts[1:])
    if info.is_dir() or info.file_size>200000:continue
    if not(str(rel).endswith('.py') or str(rel).lower() in ['readme.md','license','license.txt']):continue
    p=(d/rel).resolve();assert p.is_relative_to(d.resolve())
    b=z.read(info);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    rows.append({'path':str(rel),'sha256':hashlib.sha256(b).hexdigest()})
   result={'repo':repo,'branch':branch,'archive_url':url,'archive_sha256':hashlib.sha256(data).hexdigest(),'files':rows,'previous_errors':errors}
   (d/'ARCHIVE_MANIFEST.json').write_text(json.dumps(result,indent=2));print(repo,len(rows),'OK',flush=True);return
  except Exception as e:errors.append(str(e))
 print(repo,errors,flush=True)
with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(fetch,REPOS))
