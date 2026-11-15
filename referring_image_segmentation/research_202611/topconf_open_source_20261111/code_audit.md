**源码定位与校验**

快照为作者原文件，未修改算法。分支压缩包/原始文件获取时间和SHA256见各manifest；没有伪造commit号。

- EMCAD：/home/waas/mambavision_stage22/research/module_design_20261111/official_code/SLDGroup__EMCAD/lib/decoders.py；MSCB:L113; EUCB:L193; LGAG:L220; EMCAD:L315；SHA256 4b36981117376ea005aed8844f4ea9c846f432286695b85c1530797a20157a18。
- MambaLiteUNet：/home/waas/mambavision_stage22/research/topconf_open_source_20261111/official_code/maklachur__MambaLiteUNet/models/MambaLiteUNet.py；MambaSSMLayer:L29; LocalGlobalFeatureMixing:L77; AdaptiveMultiBranchMambaFeatureFusion:L118; CrossGatedAttentionGate:L256；SHA256 e62b5b00136b10abef4729740cec7cacdfcd53d4d8247cf2ce680065e67bc535。
- MADGNet：/home/waas/mambavision_stage22/research/topconf_open_source_20261111/official_code/Inha-CVAI__MADGNet/IS2D_models/mfmsnet.py；MultiFrequencyChannelAttention:L34; MFMSAttentionBlock:L113；SHA256 8f1b4f8ac95313bc7d8043237cc161d529d5ee44d7a81eacbcd6e9526da61b03。
- EGE-UNet：/home/waas/mambavision_stage22/research/topconf_open_source_20261111/official_code/JCruan519__EGE-UNet/models/egeunet.py；group_aggregation_bridge:L47；SHA256 c9ae22ca5c6c13a74bb9f0b72958c35708e8b0853541f42e400df6c28ebfc310。
- UCTransNet：/home/waas/mambavision_stage22/research/topconf_open_source_20261111/official_code/McGregorWwww__UCTransNet/nets/CTrans.py；ChannelTransformer:L327；SHA256 5fdbc5d1f445b997fa57d2d1227ef78e5f49849c647d49e7faa62d07dd66eb89。
- UCTransNet：/home/waas/mambavision_stage22/research/topconf_open_source_20261111/official_code/McGregorWwww__UCTransNet/nets/UCTransNet.py；CCA:L55；SHA256 ebc8c12eade39b5f97f2f292440e3bd7d1c7a35a84320d98300a334a24d72491。
- TransFuse：/home/waas/mambavision_stage22/research/topconf_open_source_20261111/official_code/Rayicer__TransFuse/lib/TransFuse.py；BiFusion_block:L20；SHA256 bd452682dbec75cbc0782ff4e494127aa369f430d6d006331dd6fc7a0c7d40f7。
- UACANet：/home/waas/mambavision_stage22/research/module_design_20261111/official_code/plemeri__UACANet/lib/modules/attention_module.py；UACA:L112；SHA256 a4199dd15d8fe677b4cfcc9f40896ac32dfaeb9d8a660a7f94a7812ac18ff8ad。
- LoMix：/home/waas/mambavision_stage22/research/topconf_open_source_20261111/official_code/SLDGroup__LoMix/trainer.py；WeightedFusion:L28; CombinatorialMutationsLossModule:L65；SHA256 2209334a51b93aacd5a24362f6984a4df6b8b4a27c284faba46e87de1fec5ecf。

**执行检查**

28项随机张量前向/反向检查全部通过；不代表准确率提升或完整训练显存需求。

| 模块 | 尺寸 | 精度 | 参数量 | 结果 |
|---|---|---|---|---|
| LGAG | 16 | fp32 | 1379 | pass |
| LGAG | 16 | bf16 | 1379 | pass |
| LGAG | 32 | fp32 | 1379 | pass |
| LGAG | 32 | bf16 | 1379 | pass |
| LGAG | 64 | fp32 | 1379 | pass |
| LGAG | 64 | bf16 | 1379 | pass |
| CGA | 16 | fp32 | 68611 | pass |
| CGA | 16 | bf16 | 68611 | pass |
| CGA | 32 | fp32 | 68611 | pass |
| CGA | 32 | bf16 | 68611 | pass |
| CGA | 64 | fp32 | 68611 | pass |
| CGA | 64 | bf16 | 68611 | pass |
| MFMS | 16 | fp32 | 86148 | pass |
| MFMS | 16 | bf16 | 86148 | pass |
| MFMS | 32 | fp32 | 86148 | pass |
| MFMS | 32 | bf16 | 86148 | pass |
| MFMS | 64 | fp32 | 86148 | pass |
| MFMS | 64 | bf16 | 86148 | pass |
| GAB | 16 | fp32 | 14520 | pass |
| GAB | 16 | bf16 | 14520 | pass |
| GAB | 32 | fp32 | 14520 | pass |
| GAB | 32 | bf16 | 14520 | pass |
| GAB | 64 | fp32 | 14520 | pass |
| GAB | 64 | bf16 | 14520 | pass |
| EMCAD_full | 0 | fp32 | 10289723 | pass |
| EMCAD_full | 0 | bf16 | 10289723 | pass |
| EMCAD_full_64 | multi | fp32 | 284043 | pass |
| EMCAD_full_64 | multi | bf16 | 284043 | pass |

LoMix concat重复输出检查、UACA内存布局检查的原始数值与说明见compatibility_results.json。EMCAD全64通道额外检查的复现代码见compatibility_emcad64.py。
