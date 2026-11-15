from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures,zipfile,io,datetime
ROOT=Path(__file__).resolve().parent
REPOS=['Inha-CVAI/MADGNet','saadwazir/MCADS-Decoder','SLDGroup/LoMix','Yejin0111/FPN-in-FPN','maklachur/MambaLiteUNet','KangBoce/PMRNet','Rayicer/TransFuse','McGregorWwww/UCTransNet','JCruan519/EGE-UNet','jeya-maria-jose/UNeXt-pytorch','bowang-lab/U-Mamba','Jiarun-Liu/Swin-UMamba','ge-xing/SegMamba','MIC-DKFZ/MedNeXt','SLDGroup/EffiDec3D','Reza-Safdari/MixStyleFlow','jeya-maria-jose/Medical-Transformer','yhygao/UTNet']
def fetch(repo):
 d=ROOT/'official_code'/repo.replace('/','__');d.mkdir(parents=True,exist_ok=True)
 errors=[]
 for branch in ['main','master']:
  try:
   url=f'https://codeload.github.com/{repo}/zip/refs/heads/{branch}'
   req=urllib.request.Request(url,headers={'User-Agent':'research-code-audit'})
   with urllib.request.urlopen(req,timeout=60) as f:
    data=f.read(50*1024*1024+1)
   if len(data)>50*1024*1024:raise RuntimeError('Archive exceeds50MiB: not downloaded beyond cap')
   z=zipfile.ZipFile(io.BytesIO(data));rows=[];allfiles=[]
   for info in z.infolist():
    rel=Path(*Path(info.filename).parts[1:])
    if info.is_dir():continue
    allfiles.append(dict(path=str(rel),bytes=info.file_size))
    allowed=rel.suffix.lower() in ['.py','.md','.txt','.yaml','.yml','.json','.sh','.toml','.cfg'] or rel.name.lower().startswith(('license','copying'))
    if not allowed or info.file_size>500000:continue
    p=(d/rel).resolve();assert p.is_relative_to(d.resolve())
    b=z.read(info);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    rows.append(dict(path=str(rel),sha256=hashlib.sha256(b).hexdigest()))
   result=dict(repo=repo,branch=branch,url=url,archive_sha256=hashlib.sha256(data).hexdigest(),archive_bytes=len(data),fetched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=rows,all_files=allfiles,previous_errors=errors,scope='Source snapshot only; no weights/datasets extracted. No training executed.')
   (d/'ARCHIVE_MANIFEST.json').write_text(json.dumps(result,indent=2))
   print('OK',repo,'source_files',len(rows),'archive_bytes',len(data),flush=True);return dict(repo=repo,status='source_snapshot',source_files=len(rows))
  except Exception as e:errors.append(type(e).__name__+': '+str(e))
 (d/'FETCH_FAILURE.json').write_text(json.dumps(dict(repo=repo,errors=errors),indent=2))
 print('FAIL',repo,errors,flush=True);return dict(repo=repo,status='fetch_failed',errors=errors)
with concurrent.futures.ThreadPoolExecutor(4) as pool:results=list(pool.map(fetch,REPOS))
old=ROOT.parent/'module_design_20261111/official_code'
for repo in ['SLDGroup/EMCAD','DengPingFan/PraNet','plemeri/UACANet','UARK-AICV/MEGANet','gorkemcanates/Dual-Cross-Attention','simzhangbest/HMT-Unet','ai4colonoscopy/PraNet-V2']:
 d=old/repo.replace('/','__')
 if (d/'ARCHIVE_MANIFEST.json').exists():results.append(dict(repo=repo,status='reuse_existing_source_snapshot',path=str(d)))
(ROOT/'code_fetch_summary.json').write_text(json.dumps(results,indent=2))
