# ISIC2016 五组统一300轮：seed42

无文本、后置固定/逐图文本、前置Cross-Attention固定/逐图文本，五组全部从相同seed42规则重新初始化，分别300轮/15300次更新。使用完整300轮cosine调度，不续接100轮分割权重。

数据806/90/379、batch16、lr1e-4、5轮warmup和原增强不变。DINOv3/CLIP冻结，只训练Adapter/融合/Decoder/Head。历史100轮结果保留；MosMed及其他seed不恢复。

配置PROTOCOL.md；运行python -u launch.py。进度run_status.json和train.log，最终analysis/REPORT.md/summary.json。数据与文本缓存只读复用，成功后保留5个新最佳权重。

## 已完成

五组均完成300轮/15300步，原图评估及独立计数复算通过。

| 模型 | 最佳轮 | 测试macro Dice | 测试pooled Dice |
|---|---:|---:|---:|
| E0_visual | 21 | 90.248% | 91.667% |
| E1_fixed_text | 157 | 90.771% | 91.724% |
| E2_image_text | 89 | 90.433% | 91.683% |
| predecoder_fixed_text | 21 | 90.311% | 91.754% |
| predecoder_image_text | 30 | 90.415% | 91.690% |

完整解释、配对区间与100/300预算对照见analysis/FINDINGS.md和REPORT.md。
