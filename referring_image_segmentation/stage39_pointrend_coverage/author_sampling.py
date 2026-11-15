from pathlib import Path
import ast,hashlib
import torch
from torch.nn import functional as F
AUTHOR=Path('/home/waas/mambavision_stage22/research/pointrend_author_20261112/official_code/projects/PointRend/point_rend')
scope=dict(torch=torch,F=F,cat=torch.cat)
PROVENANCE=[]
for name,wanted in [('point_features.py',{'point_sample','get_uncertain_point_coords_on_grid','get_uncertain_point_coords_with_randomness'}),('semantic_seg.py',{'calculate_uncertainty'})]:
 path=AUTHOR/name
 nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
 assert {n.name for n in nodes}==wanted
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
 PROVENANCE.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),nodes=[n.name for n in nodes]))
point_sample=scope['point_sample']
select_grid=scope['get_uncertain_point_coords_on_grid']
select_train=scope['get_uncertain_point_coords_with_randomness']
uncertainty=scope['calculate_uncertainty']
