# HiRAM / MambaVision 特征诊断

这是一项对既有训练结果的事后诊断，不训练模型、不修改原始实验、权重或预测。分析使用 seed 42 已选的最佳 Macro Dice 权重和完整的 808 张 ISIC2018 开发验证图。输出留在本地，不上传数据或病例图。

完整结论见 [REPORT.md](REPORT.md)。脚本依赖现有的 stage14、stage16、stage17、stage18、stage19 代码、预训练权重、数据和 CUDA 环境。从本目录依次运行 `python -u diagnose.py`、`python -u route_audit.py`、`python -u encoder_scale_audit.py`、`python -u roi_intervention_audit.py`、`python -u dual_router_audit.py`、`python summarize_counterfactual.py` 和 `python verify_recovery.py`。输出包括各项审计 JSON、逐图 CSV、按规则选择的改善/退步样例和原始分析数组。

特征能量图和特征差异图仅表示通道均方根，不是分割概率、注意力、因果归因或单个通道语义。路由图是模型在推理时预测的三类标签；真实标注只用于事后评价。固定权重切换扫描顺序/旁路 HiRAM 属于推理期干预，可能带来训练和推理分布差异，不能替代重新训练的消融。

本分析使用已用于模型和 checkpoint 选择的开发验证集与单个随机种子。图片可能来自相关患者，未核验患者级独立性；不可据此宣称跨种子稳定或外部泛化。案例选择包含表现明显改善及明显退步的图，不代表总体频率。
