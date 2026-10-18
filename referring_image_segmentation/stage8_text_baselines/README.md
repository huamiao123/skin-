# RefCOCO：DINOv3 / VMamba / Concat 图文分割

2026-10-18：按用户要求停止，状态为 `stopped_by_user`，训练与两个等待后处理/清理的进程均已退出，四组已产生的最佳权重和日志保留。未完成30轮与最终测试，不能将其标记为完成结果。停止记录见 [STOPPED_BY_USER.json](STOPPED_BY_USER.json)。后续任务改为 [ISIC2016医学文本引导分割](../stage9_isic_text/README.md)。

本阶段仅seed42，三组主模型共享冻结CLIP Text Encoder、相同Swin Pixel Feature Decoder和文本—像素匹配头。额外一组从头训练的DINO固定文本对照检验语言收益。

协议见 [PROTOCOL.md](PROTOCOL.md)，代码为 [models.py](models.py)、[data.py](data.py)、[train.py](train.py)、[evaluate.py](evaluate.py)。新数据不沿用ISIC划分和几何增强；没有加载历史ISIC分割checkpoint。

`python launch.py` 依次监督完整图像/文本/mask准备、初始化/梯度/显存/原图评估预检、短程中断恢复、正式30轮训练和最终锁定checkpoint评估。当前状态见 `run_status.json`；数据下载进度见 `data_status.json`，训练中每500步更新 `results/seed42/status.json`。只有顶层validation.json passed和状态complete才表示所有工作完成。

统计单位为表达，额外给出image-balanced分数，区间按图聚类。train共120624表达，每epoch7539个batch16；30轮每个模型226170步。Validation原尺寸mean IoU选模，testA/testB在全部选择锁定后评估。第一轮为句子向量匹配的可行性基线，不能直接视为完整RIS方法复现或证明复杂关系推理能力。

`assets/`、`runtime/` 和本地`analysis/examples/`保留大型依赖和数据；数据镜像版本、COCO图像/实例、CLIP权重的来源与hash分别记录于data_manifest.json和text_provenance.json。原RefCOCO ZIP未验证逐字节一致性，已经明确披露，不将镜像误写成来自当前可连接的官方原ZIP。

预检发现官方UNC的train图137715与val图242807文件字节相同，影响val的9条表达，详见[data_overlap_audit.json](data_overlap_audit.json)。主结果保留官方划分。已安排独立、只读的补充后处理，在checkpoint选择锁定后生成 `analysis/OVERLAP_SENSITIVITY.md`，报告排除这张验证图的指标；该子集不用于重新选模，也不称为新的测试集。
