# MambaVision-S + VMamba SS2D式四路扫描

本轮只做一个seed42、300轮模型。使用VMamba官方四向cross_scan/cross_merge，四方向共享MambaVision原SSM预训练参数；保留窗口/CNN/Attention及非SSM分支，不新增参数、不解冻。这是四路扫描消融，**不是整块VMamba SS2D替换**。

与stage14相同的ISIC2018 1886/808开发划分、原Adapter/Swin Decoder、batch64、AdamW.001、Cosine300、原增强/loss及最高验证macro Dice选模。原始stage14结果只读复用，不重训；新模型的分割头从相同seed42初值重新训练。

配置见 [PROTOCOL.md](PROTOCOL.md)、config.json；扫描身份/次序/权重一致性见routing_preflight.json；初始化/梯度/显存见preflight.json。进度run_status.json与results/status.json；最终analysis/REPORT.md、summary.json和per_image.csv。

代码复用stage14训练、恢复、评价和分析实现，并记录本目录副本/变化；历史数据和指标仍直接调用原类/函数。official/与runtime/只读共享stage14依赖，原权重和官方源码不复制、不改动。VMamba的固定Git对象与算子SHA记于routing_preflight.json，实际CUDA SSM继续使用stage14官方mamba-ssm1.2.2扩展。

在本目录运行 `python -u launch.py`：依次预检、两轮独立smoke恢复、正式300轮、三个selector导出、完整计时、误差/图表及独立完成审计。恢复用 `python -u train.py` 或重新运行launch。评价用evaluate.py，随后benchmark.py、analyze.py、validate.py。依赖版本沿用stage14和requirements.txt。

保留best_dice/best_loss/last及latest恢复权重；808张概率与Mask保存在results/{criterion}/。含原图的示例仅保留analysis/examples，不推送GitHub。single-route模式仅用于预检身份测试，正式训练/导出必须四路。

## 完成结果

300轮/9000更新及全部独立校验已通过。主权重按验证macro Dice选模，完整结果见 [报告](analysis/REPORT.md) 与 [summary.json](analysis/summary.json)。

best_dice 第116轮，开发验证 macro Dice 88.3712%，pooled Dice 88.7314%。单seed开发验证结果，不作独立测试或架构机制归因。
