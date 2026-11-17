# MambaVision-S 结构适配审计（2026-11-16）

## 版本与范围

- GitHub `main` 只读查询为 `7a23c1b8f601716a89c9b6a627fd9efbcf4812f4`。正在使用的服务器代码树并非 Git checkout，不能仅凭此 SHA 声称它与远端逐文件相同。19 个相关服务器文件的绝对位置、大小、SHA256 和依赖版本见 `reports/source_inventory.json`。
- 官方 MambaVision 配置固定在 commit `7860a506b2eb844eaaae676f08461ce8c3c26f43`；MMSegmentation 头实现固定在 `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`。Stage45 依赖的官方源码逐文件根据原 `SOURCE_MANIFEST.json` 验证。补充 FCNHead 源码的 SHA256 为 `c6fc931ac546d5d1f19bd2bf84679aa42be3245412b4d4c02e89bf2b16ba473a`，见 `vendor/` 及原许可证。
- 当前环境 Python 3.11.8、torch 2.2.2+cu121、mmcv-lite 2.1.0、mmengine 0.10.7、transformers 4.44.2、timm 1.0.22、mamba-ssm 1.2.2。此环境启动需设 `PYTHONNOUSERSITE=1`。本次未触发 GPU 计算。

## 结构与训练事实

- Stage45 的 `NativeUPerProbe` 从固定源码 AST 提取真正的 UPerHead，宽度限 64/128，固定四尺度与 256 输出；`train.System` 替换了 Stage28 的 `Probe`，未使用 FCN 辅助头。
- 新 `src/model.py` 仍运行同一官方 UPerHead 的提取 AST，增加 512 宽度、尺寸检查与动态输出。FCNHead 的类也从上述固定源码 AST 提取；其输入 f3、384 输入通道、256 中间通道、单卷积、无 concat。C 组训练返回两个 logits；eval 只返回主头。单通道 BCE + 逐图 Soft Dice、辅助权重 0.4 是本任务的二分类适配，不是 ADE20K 150 类训练配方。
- 使用 Stage28 `System` 的相同结构路径：冻结 patch embedding 与 levels 0/1；levels 2/3 参与梯度；整个 encoder 始终 eval；前两层及 mid 可缓存，后两层每次重算。解码器/辅助头与可训练 backbone 分别为优化器组，候选学习率 1e-3 / 1e-4。训练脚本尚未构建，候选配置仍不能用于正式运行。
- HF 模型配置窗口为 `[8,8,14,7]`、DropPath 0.2；官方分割配置为 `[8,8,160,56]`、DropPath 0.7。首轮沿用同一个项目 HF backbone，不把 backbone 差异误归因给 decoder。
- Stage29 是局部逐级与深层锚点的融合对照，不能直接当成 UPer FPN 融合实现。Stage37 的100轮调度修改和 Stage29 的300轮调度周期不同；Stage25 诊断是开发证据，Stage26 probe 不等于特征不存在。

## 数据和检查

- 划分为 train 1886、val 762、已暴露历史 test 867。训练/验证 ID 是各自从 0 编号的 split-local 名称，直接把 ID 交集视作重复是错误的。测试实际文件与历史清单的图像/掩膜 SHA 已核对；train/val/test 的 256 RGB 解码内容哈希未见跨划分重叠，两份历史重叠审计和测试清单也通过其清单 SHA。患者级互斥、全部近重复图像均未证实。
- 预训练权重 `model.safetensors` SHA256 `006af39903f2d88c5b455431d087d0fcbf5d5932114aef52542c0d2f0b1ec180`；同时核验7个源码/配置文件，CPU 严格加载且无 missing/unexpected keys。模型主体参数 50,140,584。
- CPU 真正运行 backbone 的 f1/f2，256 输入分别为 `(1,96,64,64)`、`(1,192,32,32)`。f3 selective-scan 使用 CUDA 扩展，不支持 CPU，因此 f3/f4 实际尺寸与完整 backbone 计算只能等待 GPU 授权验证。三组真实作者解码头均用合成 CPU 输入验证256/512尺寸。
- `reports/cpu_tests.log`：8 项测试通过，覆盖 A 与历史 Stage45 UPer64 初始化逐张量完全一致、B/C 主头初始化一致、头尺寸、主辅梯度、合成 backbone 冻结及 BN、损失及梯度与旧实现一致、CPU 断点恢复、指标、数据和身份审计哈希。分割性能尚未测试。
- `reports/parameter_counts.json` 是 CPU 实例化得到的参数数：A 总 51,229,673 / 可训练 47,078,849；B 总 81,561,513 / 可训练 77,410,689；C 总 82,447,018 / 可训练 78,296,194。各组可训练 backbone 均为 45,989,760。未实测显存、吞吐、结果或 GPU 梯度。
