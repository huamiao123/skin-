"""Read-only source inventory for the independently stored Stage A adapter."""
import hashlib, importlib.metadata, json, os, platform, sys
from pathlib import Path
root=Path('/home/waas/mambavision_stage22/source/referring_image_segmentation')
workspace=Path(__file__).resolve().parents[1]
paths=['AGENTS.md','evaluation.py','stage1_visual/data.py','stage1_visual/models.py','stage1_visual/data_manifest.json','stage3_confirmation/data.py','stage28_drop_path/train.py','stage29_context_fusion/fusion.py','stage29_context_fusion/results/summary.json','stage37_lr_horizon100/PROTOCOL.json','stage25_problem_diagnosis/evidence_summary.json','stage26_coverage_probes/results/summary.json','stage45_upernet_decoder/author_modules.py','stage45_upernet_decoder/PROTOCOL.json','stage45_upernet_decoder/train.py','stage45_upernet_decoder/uper64/results/history.json','stage45_upernet_decoder/identity_partition.json','stage14_mambavision_s_isic2018_seed42/official/hf/config.json','stage14_mambavision_s_isic2018_seed42/official/hf/model.safetensors']
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
files={x:({'sha256':sha(root/x),'size':(root/x).stat().st_size} if (root/x).is_file() else {'missing':True}) for x in paths}
versions={}
for name in ('torch','torchvision','mmcv-lite','mmengine','transformers','timm','mamba-ssm','numpy','PyYAML'):
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]=None
partition=json.loads((root/'stage45_upernet_decoder/identity_partition.json').read_text())
config=json.loads((root/'stage14_mambavision_s_isic2018_seed42/official/hf/config.json').read_text())
report={'git_main_remote_observed':'7a23c1b8f601716a89c9b6a627fd9efbcf4812f4','git_main_remote_source':'read-only ls-remote from local Git, 2026-11-16; live server source tree is not itself a Git checkout','server_source_root':str(root),'file_hashes':files,'software':versions,'python':sys.version,'platform':platform.platform(),'pretrained_config':{k:config.get(k) for k in ('depths','num_heads','window_size','dim','in_dim','drop_path_rate')},'split_counts':partition['counts'],'patient_disjointness_verified':partition['patient_disjointness_verified'],'test_status':'exposed historical filtered test; not fresh independent test','dataset_numbering':'train and val IDs are split-local; exact decoded image hashes must be used for cross-split checks','official_segmentation_config_commit':'7860a506b2eb844eaaae676f08461ce8c3c26f43','mmseg_commit':'c685fe6767c4cadf6b051983ca6208f1b9d1ccb8','fcn_head_sha256':sha(workspace/'vendor/fcn_head.py')}
(workspace/'reports/source_inventory.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print('SOURCES',len(files),'MISSING',[k for k,v in files.items() if v.get('missing')]);print('VERSIONS',versions);print('WINDOW',report['pretrained_config']['window_size'])
