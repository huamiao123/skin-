from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures
ROOT=Path(__file__).resolve().parent/'official_code';ROOT.mkdir(exist_ok=True)
REPOS=['SLDGroup/EMCAD','DengPingFan/PraNet','plemeri/UACANet','gorkemcanates/Dual-Cross-Attention','UARK-AICV/MEGANet','simzhangbest/HMT-Unet','ai4colonoscopy/PraNet-V2']
def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'research-method-audit'})
 with urllib.request.urlopen(req,timeout=40) as f:return f.read()
def fetch(repo):
 d=ROOT/repo.replace('/','__');d.mkdir(exist_ok=True)
 try:
  info=json.loads(get('https://api.github.com/repos/'+repo));branch=info['default_branch']
  tree=json.loads(get('https://api.github.com/repos/'+repo+'/git/trees/'+branch+'?recursive=1'))
  sha=tree['sha'];(d/'tree.json').write_text(json.dumps(tree,indent=2))
  files=[x['path'] for x in tree['tree'] if x['type']=='blob' and (x['path'].endswith('.py') or x['path'].lower() in ['readme.md','license','license.txt']) and x.get('size',0)<200000]
  manifest=[]
  for name in files:
   url='https://raw.githubusercontent.com/'+repo+'/'+sha+'/'+name
   try:
    data=get(url);p=d/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    manifest.append({'path':name,'url':url,'sha256':hashlib.sha256(data).hexdigest()})
   except Exception as e:manifest.append({'path':name,'error':str(e)})
  result={'repo':repo,'sha':sha,'files':manifest}
 except Exception as e:result={'repo':repo,'error':str(e)}
 (d/'SOURCE_MANIFEST.json').write_text(json.dumps(result,indent=2));print(json.dumps({'repo':repo,'files':len(result.get('files',[])),'error':result.get('error')}),flush=True)
with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(fetch,REPOS))
