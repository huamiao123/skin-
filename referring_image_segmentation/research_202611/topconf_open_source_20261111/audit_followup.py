from pathlib import Path
import json,re,hashlib,urllib.request,datetime,concurrent.futures
R=Path(__file__).resolve().parent
records=[]
for group in ['decoder','skin','generalization']:
 for p in json.loads((R/'api'/group/'papers.json').read_text()):
  records.append(dict(p,retrieval_query_group=group))
parent=list(range(len(records)))
def root(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
seen={}
for i,p in enumerate(records):
 title=re.sub(r'[^a-z0-9]','',p.get('title','').lower())
 doi=p.get('doi','').lower().replace('https://doi.org/','').strip()
 keys=['title:'+title] if title else []
 if doi:keys.append('doi:'+doi)
 ar=re.search(r'(\d{4}\.\d{4,5})',str(p.get('arxiv_id',''))+' '+doi+' '+p.get('url',''))
 if ar:keys.append('arxiv:'+ar.group(1))
 for k in keys:
  if k in seen:parent[root(i)]=root(seen[k])
  else:seen[k]=i
merged={}
for i,p in enumerate(records):merged.setdefault(root(i),[]).append(p)
result=[dict(representative=max(ps,key=lambda p:len(p.get('abstract','') or '')),provenance_records=ps) for ps in merged.values()]
(R/'api_merged.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
summary=dict(within_query_total=len(records),cross_query_unique=len(result),query_counts={g:sum(p['retrieval_query_group']==g for p in records) for g in ['decoder','skin','generalization']},dedupe='DOI/arXiv identifier/normalized exact title connected components; no fuzzy matching')
(R/'retrieval_summary.json').write_text(json.dumps(summary,indent=2));print('DEDUPE',summary,flush=True)
for g in ['decoder','skin','generalization']:
 for name in ['topic_brief.md','report.md']:
  p=R/'api'/g/name
  if p.exists() and ('algebraic connectivity' in p.read_text() or '多智能体' in p.read_text()):
   p.rename(p.with_name('INVALID_TEMPLATE_'+name))
   p.write_text('INVALID AUTOMATED TEMPLATE OUTPUT. Quarantined as INVALID_TEMPLATE_'+name+'. The helper contains unrelated connectivity-control boilerplate. Do not use it as medical evidence. Consult the manually audited main report instead.\n')
ns={'__file__':str(R/'fetch_code.py')};exec((R/'fetch_code.py').read_text().split('with concurrent.futures.ThreadPoolExecutor')[0],ns)
print(ns['fetch']('JiarunLiu/Swin-UMamba'),flush=True)
repo='McGregorWwww/UCTransNet';d=R/'official_code'/repo.replace('/','__');d.mkdir(exist_ok=True);out=[]
for rel in ['README.md','nets/CTrans.py','nets/UCTransNet.py','Config.py','train_model.py','test_model.py','LICENSE','LICENSE.md']:
 url=f'https://raw.githubusercontent.com/{repo}/main/{rel}'
 try:
  b=urllib.request.urlopen(url,timeout=25).read();p=d/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);out.append(dict(path=rel,url=url,sha256=hashlib.sha256(b).hexdigest()))
 except Exception as e:out.append(dict(path=rel,error=str(e)))
(d/'RAW_MANIFEST.json').write_text(json.dumps(dict(repo=repo,files=out,fetched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2));print('UCTransNet_RAW',[(x['path'],'error' not in x) for x in out],flush=True)
