# MambaVision-S Decoder 融合实验

本目录包含 Swin 配对基线、仅门控与 AXIS-Bridge 的三组实验，以及 [SegMAN Decoder 适配版](segman_adaptation/README.md)。均为 ISIC2018 历史 1886/808 划分、seed 42 和 300 epochs；验证集用于选模，不是独立测试。

[AXIS 报告](analysis/REPORT.md) · [SegMAN 报告](segman_adaptation/analysis/REPORT.md) · [论文及源码核对](PAPER_REVIEW.md) · [统一实验汇总](../experiment_inventory_macro_dice_20261107.md)

公开归档包含每个模型的最高验证逐图平均 Dice 权重、源码、参数、日志、逐图指标和汇总；不包含预训练 backbone、其余 checkpoint、原图及逐像素预测。完整本地优化器断点未公开，不能从本归档直接断点续训。原评估脚本还依赖其他 checkpoint 及像素预测，重新运行需自行补齐或重训。

第三方源码不在仓库中；在 `referring_image_segmentation` 目录运行 `python publication/restore_stage20_21_references.py` 可以从固定 commit 下载并验证。原始数据、stage14 预训练模型及依赖也需要另行合法取得。AXIS 和 SegMAN 都只做了模块适配，不是完整论文复现。
