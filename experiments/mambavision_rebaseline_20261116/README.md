# MambaVision 皮肤病灶分割：已完成实验归档

本目录现已归档 15 个完整分割模型训练和 12 个冻结特征读出。下表保留 2026-11-16 至 11-17 首批 7 组训练的原始结果。旧 Stage1–51 的记录仍在原位置。

- [11 月 18–19 日 F3 语义锚定实验](method1_f3_20261118/RESULTS_zh.md) 收录 F3、同卡 UPer64 和 F4 三组，均完成 100 轮及最佳权重验证重放，本轮只使用开发数据。
- [11 月 17 日补录结果与诊断](followup_20261119/README.md) 收录后续 5 个模型、12 个分层读出及数据与误差诊断。

## 完成结果

指标均为逐图前景 Dice 的算术平均；最佳权重按验证集宏平均 Dice 选择。验证 762 张、测试 867 张，固定阈值 0.5，单尺度，无 TTA。

| 实验 | 最佳轮次 | 验证 Dice | 测试 Dice |
|---|---:|---:|---:|
| A：UPerNet-64，无辅助头 | 39 | 89.3603% | 89.6286% |
| B：UPerNet-512，无辅助头 | 53 | 88.9038% | 88.7106% |
| C：UPerNet-512 + FCN 辅助头 | 39 | 88.8986% | 89.4014% |
| Stage B 同卡重新训练对照 | 39 | 89.1695% | 89.5919% |
| 颜色增强 | 56 | 89.1871% | 88.9549% |
| 轻度缩小增强 | 39 | 89.0386% | 88.9671% |
| 解码器学习率 3e-4 | 39 | 88.9239% | 89.6745% |

A 原训练验证值为 89.36035098%，表中为当前 4090 重放值 89.36031394%；原值与设备信息保留在原始报告，未覆盖。

Stage B 使用同一块 4090 重新训练的 control 作为对照。颜色、尺度、降低解码器学习率三个方案均未通过预先设置的验证筛选。解码器学习率降低后测试略升，但验证下降，未因此改用该方案。这里只记录一次 seed=42 的观察，不能据此宣布方法优越或有统计稳定提升。

## 文件

- src/、tests/：执行过的模型、训练、评估、增强、诊断与核查代码原样快照；不包含未完成的编码器学习率实验。
- vendor/：FCN 作者源码与 MMSegmentation Apache-2.0 许可证。
- RESULTS.json：精确分数、最佳轮次、最佳权重 SHA256、设备信息。
- completed_artifacts.tar.gz：四组完整评估、逐图 CSV、训练配置、100 轮历史、恢复审计、验证阈值扫描、诊断和运行所需的旧项目辅助源码。包含所有对照和失败方案，没有只保留较好结果。
- ARTIFACT_MANIFEST.json：压缩包成员的大小和 SHA256；解包后可验证。
- ENVIRONMENT.json：实际训练虚拟环境的包版本。
- PROTOCOL.md、SOURCE_AUDIT.md：早期原始协议与源码审计，反映当时状态；本 README 和实际完成的报告用于解释后续执行情况。
- FILES_SHA256.json：本目录文件的校验清单（清单自身除外）。

## 协议和边界

输入 256×256，训练 1886 张，seed=42，100 轮，物理及有效 batch=64；AdamW，基线编码器 LR=1e-4、解码器 LR=1e-3、weight_decay=0.01，组内梯度裁剪 1，余弦调度。编码器前两级冻结，后两级更新；编码器保持 eval 模式，冻结 BN 统计与随机深度。图像几何增强在 FP32 执行，主要网络计算使用 BF16。精确配置以各实验 config.json 为准。

主干为 MambaVision-S。UPerNet / FCN 使用固定版本作者代码；C 对应官方头部结构的医学任务适配，不是 NVIDIA ADE20K 官方训练配方的原样复现。项目窗口为 14/7，官方 ADE20K 配方使用不同窗口配置。

训练、验证、测试使用已核查的 identity partition，归档中保留划分和测试输入核验；原始图片、mask、ImageNet 权重和训练 checkpoint 未上传。此历史测试集此前已被查看，不能称为从未接触的盲测；患者级独立性未经验证。

A/B 的训练涉及 3090 到 4090 的迁移，B 曾遇到 NFS 临时 checkpoint 写入失败后从成功保存点恢复。Stage B 各组采用同卡对照。BF16 全网络训练重放仍有微小数值差异，恢复时状态载入已审计，但不宣称逐步训练完全确定。相关原始审计保留在压缩包。

## 恢复和运行

这是原始实验记录，不是全新可移植安装包。源码保留了执行时的绝对路径和审计前置检查，未为发布而改写已运行代码。依赖旧仓库的辅助源码随包保留，不需要猜测旧版本。

先解包（建议在本目录执行）：

    tar -xzf completed_artifacts.tar.gz

恢复时，将 runtime/source/referring_image_segmentation/ 下内容放入 /home/waas/mambavision_stage22/source/referring_image_segmentation/；runtime/research/ 放入 /home/waas/mambavision_stage22/research/；runtime/local_audit_records/ 同样恢复到该根目录。将本目录代码和解包出的 results/、reports/ 放到 /home/featurize/work/mambavision_seg_rebaseline_20261116/。这些绝对路径也可在新环境明确调整，但需要重新运行预检、生成新的协议和哈希，不能声称与本归档逐字节相同。

按原配置准备 ISIC 图片与 mask，并从合法原始来源准备对应 ImageNet 权重。主干 model.safetensors 的预期 SHA256 为 006af39903f2d88c5b455431d087d0fcbf5d5932114aef52542c0d2f0b1ec180。数据和权重不包含在本次提交中。

原正式训练入口示例（必须先阅读训练器中的 smoke/preflight 检查并重新执行，而不是直接复用归档的通过标志）：

    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/waas/mambavision_stage22/env/bin/python src/train_mb64.py --arm A-64-noaux --micro-batch 64

Stage B 的 run_stageB_*.py 是本次已经执行过的有状态编排器，其中有一轮运行后重启等前置条件；不能在空目录直接当作一键入口。新复现应从相应 train_stageB_*.py 的 --help 和 config.json 开始，重跑前置核查。评估需要自行提供匹配 SHA256 的最佳权重；本归档提供结果和校验信息，不包含权重文件。

第三方代码的归属与许可参见 vendor/、runtime/research/ 内 LICENSE 及 runtime/source/.../official/ 中作者说明。
