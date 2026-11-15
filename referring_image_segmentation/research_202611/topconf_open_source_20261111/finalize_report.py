from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent
ROOT=R.parent.parent
OLD=R.parent/'module_design_20261111/official_code'
NEW=R/'official_code'
rows=json.loads((R/'curated_catalog.json').read_text())
corrections={'EffiDec3D':'networks; main_train_MSD_Task01_10.py; main_train_BTCV_TU.py','SegMamba':'model_segmamba; 2_preprocessing_mri.py; 3_train.py; 4_predict.py; 5_compute_metrics.py','UTNet':'model; train_deep.py'}
for p in rows:
 if p['id'] in corrections:p['source_entrypoints']=corrections[p['id']]
 repo=p['official_repo'].removeprefix('https://github.com/')
 d=next((x for x in [NEW/repo.replace('/','__'),OLD/repo.replace('/','__')] if (x/'ARCHIVE_MANIFEST.json').exists() or (x/'RAW_MANIFEST.json').exists()),None)
 if d:
  p['server_source_snapshot']=str(d)
  p['available_python_files']=len(list(d.rglob('*.py')))
  p['manifest_paths']=[str(x) for x in [d/'ARCHIVE_MANIFEST.json',d/'RAW_MANIFEST.json'] if x.exists()]
 else:p['server_source_snapshot']=None
 p['audited_date']='2026-11-11'
(R/'curated_catalog.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
def table(rs):
 s='| 工作 | 场所 | 可复用机制/任务 | 公开代码核查 | 当前决策 |\n|---|---|---|---|---|\n'
 for p in rs:s+=f"| [{p['id']}]({p['official_paper']}) | {p['venue']} | {p['mechanism']} | [{p['code_status']}]({p['official_repo']}) | {p['decision']} |\n"
 return s
cat='**主会工作（25项）**\n\n'+table([p for p in rows if p['venue_tier']=='main'])+'\n**补充相关工作（4项，单独标注）**\n\n'+table([p for p in rows if p['venue_tier']!='main'])
detail='\n**逐项源码记录**\n\n'
for p in rows:detail+=f"**{p['id']} — {p['venue']}**\n\n任务：{p['task']}。入口：{p['source_entrypoints']}。\n\n核查深度：{p['review_depth']}；{p['notes']}\n\n来源：[论文/发表证据]({p['official_paper']})；[官方仓库]({p['official_repo']})。\n\n"
(R/'catalog_zh.md').write_text(cat+detail)
summary=json.loads((R/'retrieval_summary.json').read_text())
checks=json.loads((R/'compatibility_results.json').read_text())
inventory=[]
for base in [NEW,OLD]:
 for d in sorted(base.iterdir()):
  mans=[p for p in [d/'ARCHIVE_MANIFEST.json',d/'RAW_MANIFEST.json'] if p.exists()]
  if mans:inventory.append(dict(directory=str(d),manifests=[str(x) for x in mans],python_files=len(list(d.rglob('*.py')))))
(R/'source_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2))
plan=dict(status='original_module_wiring_prototypes_verified_not_trained',date='2026-11-11',new_original_modules=False,production_model_modified=False,
 project=str(ROOT/'source/referring_image_segmentation'),
 common_protocol='Same pretrained backbone, stage3/4 fine tuning, 64-channel adapters, train/val/test split, loss, LR trajectory and checkpoint selection. Native author normalization/initialization retained. No test-driven architecture selection.',
 arms=[
 dict(id='R0',name='MambaVision-S + existing Swin',change='control',reuse_prior_checkpoint_only_if_protocol_identical=True),
 dict(id='R1',name='MambaVision-S + original EMCAD decoder',source_repo='SLDGroup/EMCAD',module='EMCAD',kwargs={'channels':[64,64,64,64]},input_order=['adapter3_8x8','adapter2_16x16','adapter1_32x32','adapter0_64x64'],output='d1 -> existing 64-to-1 head -> 256x256',replaces_swin=True,decoder_module_parameters=284043),
 dict(id='R2',name='MambaVision-S + Swin + original CGA',source_repo='maklachur/MambaLiteUNet',module='CrossGatedAttentionGate',kwargs={'F_g':64,'F_l':64,'F_int':32},locations=[16,32,64],routing='SwinLevel(g + CGA(g,skip)); g=upsampled decoder feature',additional_module_parameters=205833),
 dict(id='R3',name='MambaVision-S + Swin + original MFMS',source_repo='Inha-CVAI/MADGNet',module='MFMSAttentionBlock',kwargs={'in_channels':64},locations=[16,32,64],routing='MFMS after skip adapter before original decoder addition',additional_module_parameters=258444)
 ],conditional_next_steps=['Use validation evidence to select stable single-module improvements.','Repeat promising comparisons with three fixed seeds.','Only combine modules if separate effects are complementary.','Report all locked best-checkpoint full-test results; acknowledge existing test exposure.'])
(R/'integration_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
sources=[
 ('EMCAD',OLD/'SLDGroup__EMCAD/lib/decoders.py',['EMCAD','LGAG','MSCB','EUCB']),
 ('MambaLiteUNet',NEW/'maklachur__MambaLiteUNet/models/MambaLiteUNet.py',['MambaSSMLayer','AdaptiveMultiBranchMambaFeatureFusion','LocalGlobalFeatureMixing','CrossGatedAttentionGate']),
 ('MADGNet',NEW/'Inha-CVAI__MADGNet/IS2D_models/mfmsnet.py',['MFMSAttentionBlock','MultiFrequencyChannelAttention']),
 ('EGE-UNet',NEW/'JCruan519__EGE-UNet/models/egeunet.py',['group_aggregation_bridge']),
 ('UCTransNet',NEW/'McGregorWwww__UCTransNet/nets/CTrans.py',['ChannelTransformer']),
 ('UCTransNet',NEW/'McGregorWwww__UCTransNet/nets/UCTransNet.py',['CCA']),
 ('TransFuse',NEW/'Rayicer__TransFuse/lib/TransFuse.py',['BiFusion_block']),
 ('UACANet',OLD/'plemeri__UACANet/lib/modules/attention_module.py',['UACA']),
 ('LoMix',NEW/'SLDGroup__LoMix/trainer.py',['WeightedFusion','CombinatorialMutationsLossModule'])
]
audit='**源码定位与校验**\n\n快照为作者原文件，未修改算法。分支压缩包/原始文件获取时间和SHA256见各manifest；没有伪造commit号。\n\n'
for name,p,names in sources:
 tree=ast.parse(p.read_text())
 found=[f'{n.name}:L{n.lineno}' for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 audit+=f"- {name}：{p}；{'; '.join(found)}；SHA256 {hashlib.sha256(p.read_bytes()).hexdigest()}。\n"
audit+='\n**执行检查**\n\n28项随机张量前向/反向检查全部通过；不代表准确率提升或完整训练显存需求。\n\n| 模块 | 尺寸 | 精度 | 参数量 | 结果 |\n|---|---|---|---|---|\n'
for x in checks['cases']:audit+=f"| {x['module']} | {x.get('height','multi')} | {x.get('precision','')} | {x.get('parameters','')} | {x['status']} |\n"
audit+='\nLoMix concat重复输出检查、UACA内存布局检查的原始数值与说明见compatibility_results.json。EMCAD全64通道额外检查的复现代码见compatibility_emcad64.py。\n'
(R/'code_audit.md').write_text(audit)
web='**定向网页补充证据**\n\n与API检索结果分开保存；访问日期2026-11-11。下面是人工筛选的官方论文与作者代码来源，不将未进API记录的条目伪装为API命中。\n\n'
for p in rows:web+=f"- {p['id']} / {p['venue']}：[论文/发表证据]({p['official_paper']})；[作者仓库]({p['official_repo']})；核查：{p['review_depth']}。\n"
web+='\n补充确认MedCLIPSeg CVPR 2026：[CVF补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Koleilat_MedCLIPSeg_Probabilistic_Vision-Language_CVPR_2026_supplemental.pdf)。\n'
(R/'web_supplement.md').write_text(web)
footer=f'''
**交付与未完成事项**

API去重候选：{summary['cross_query_unique']}；人工重点目录：{len(rows)}；服务器已保留源码/仓库资料目录：{len(inventory)}（包括复用前一轮快照，含README占位仓库，不能全部算完整开源模型）。
实际接口检查：{len(checks['cases'])}；通过：{sum(x['status']=='pass' for x in checks['cases'])}。作者模型文件未改；未修改生产训练模型；未启动新训练。

文件：
- deep_research_report.md / deep_research_report.txt：完整中文报告。
- curated_catalog.json / catalog_zh.md：29项论文、代码状态、入口、排除理由。
- code_audit.md / compatibility_results.json：源码定位与执行结果。
- integration_plan.json：三个现成实现的接入位置与控制变量。
- api_merged.json、api/*/search_log.md、errors.log：检索记录与限制。
- web_supplement.md、source_inventory.json、official_code/*/*MANIFEST.json：来源证据与快照。

尚未完成的事项是生产模型接线、完整训练、实际GPU预算和分割指标验证；不能把本报告写成已经验证精度提高。所有新增文件均在服务器目录{R}，未向本地桌面写文件。
'''
full=(R/'report_body.txt').read_text()+'\n'+cat+'\n'+footer
(R/'deep_research_report.md').write_text(full)
(R/'deep_research_report.txt').write_text(full)
(ROOT/'TOPCONF_REUSE_RESEARCH_LATEST.txt').write_text('2026-11-11\n'+str(R/'deep_research_report.md')+'\n29 focused papers; 28 original-module interface checks and 3 decoder wiring checks passed. No new training launched.\n')
print('REPORT_DONE',dict(catalog=len(rows),inventory=len(inventory),checks=len(checks['cases']),passed=sum(x['status']=='pass' for x in checks['cases']),characters=len(full)),flush=True)

