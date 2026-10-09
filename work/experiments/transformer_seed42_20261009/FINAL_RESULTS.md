# Transformer seed42 重训最终结果（2026-10-09）

两项均完成固定300轮。数据为旧 ISIC2018 划分（1886训练、808开发验证），256×256。
按最低验证 loss 选模型，最终评价 batch1，阈值0.5。DSC/IoU 为 pooled pixel 指标，不是独立测试结果。

| 模型 | 参数量 | 选中epoch | DSC | IoU | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| swin_pretrained | 41,392,836 | 16 | 0.89719584 | 0.81355863 | 0.90536744 | 0.96369338 |
| ptu_2262 | 15,764,801 | 174 | 0.83826920 | 0.72156923 | 0.84475256 | 0.94507574 |

训练期间的最高 DSC 另列，未用于改选报告模型：

- swin_pretrained：第141轮，DSC=0.90132426。
- ptu_2262：第206轮，DSC=0.84500865。

共同配置：seed42、FP32、train batch64、val batch8、workers4、AdamW lr1e-4/wd0.01、
Cosine300/eta_min1e-5、梯度裁剪1.0、最终输出 BCE+Dice。Swin 从 ImageNet 初始化并启用
激活重算；PTU 从零初始化。两者没有 EGE 的五路深监督，因此跨架构比较仍有损失差异。

Swin 于2026-10-09 10:32:50 UTC完成，PTU于11:28:43 UTC完成。初始 Swin OOM 和 PTU第5轮后的
暂停/恢复记录仍保留；Swin正式完整训练启用激活重算，PTU使用完整训练状态断点继续。

[完整结果CSV](final_results.csv) · [训练代码](train.py) · [冻结依赖源码](source/) ·
[逐epoch记录与最终DONE](snapshot/results/) · [队列状态](snapshot/queue_status.json) ·
[本地权重哈希清单](checkpoint_manifest.json)。模型权重和原始数据未上传。
