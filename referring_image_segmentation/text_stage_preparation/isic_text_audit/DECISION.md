# ISIC 文本数据切换核查

## 当前执行状态（2026-10-18 后续更新）

用户已明确要求尝试ISIC并停止COCO。RefCOCO训练与等待进程已停止，四组最佳权重保留。ISIC官方图像/Mask已完整取得并与MedVLSM文本配对，去重后806训练/90验证/379测试。采用P3颜色/形状提示，将统一的`skin melanoma`类别词改为`skin lesion`，不使用Mask生成的大小、数量、位置字段。三视觉主组及DINO固定文本对照仅seed42，正式100epochs任务已启动；入口与冻结协议见 [stage9_isic_text](../../stage9_isic_text/README.md)。下面保留各次核查时的发现；此前“尚未切换”的表述是当时状态，已被本节更新。

2026-10-18 用户要求：如果 ISIC 有合适的图文分割数据，就切换；仅一个 seed。

最新结论（2026-10-18 补充核查）：已找到并下载 MedVLSM 作者公开的 ISIC 文本标注三份 JSON，不能再概括为“没有现成 ISIC 图文分割数据”。下面原有的 Kaggle 覆盖核查仍然成立，但只适用于该 Kaggle 数据，不能推广至所有 ISIC 文本资源。本次没有启动 ISIC 文本训练或切换正在运行的任务。

## 补充：已取得 MedVLSM 的 ISIC 文本标注

- 作者仓库：https://github.com/naamiinepal/medvlsm 。VLSM-Adapter 的 README 明确引用该仓库的数据准备流程。
- 固定来源版本：`dc36a3af83035efa344c8ce26397528ad173569e`。
- 原始文件：`data/isic/anns/train.json`、`val.json`、`test.json`；本地保存于本目录下 `medvlsm/`，总计 2,281,155 字节。
- 实际读取记录数：train 810、val 90、test 379。每条记录含图像名、Mask 名、bbox 和 p0–p9 文本提示，含类别、形状、颜色、大小、数量、位置等不同提示设定。p0 为空文本；不能只读取 `sentences` 中的空字符串而忽略 `prompts`。
- 这些是作者提供的模板化提示，不应称为医生逐图撰写的自然语言报告；使用前应明确提示生成来源和推理时提供哪些辅助信息。
- 标注已下载并保存 SHA256；图像/Mask 需要按作者 DATASETS.MD 从原始 ISIC 数据取得或与本地数据匹配。本次尚未声称完整图像/Mask 配对完成。
- 初步 ID 核查发现：训练 810 条记录对应 809 个不同 segment_id，训练与测试有 3 个相同 segment_id。正式实验前必须核对实际图像、处理重复并冻结划分，不能直接宣称该划分无泄漏。
- 下载清单、样本数和哈希见 `medvlsm/download_audit.json`。

## 已取得的证据

- 来源：Kaggle `zhaobinxu/skin-cancer-isic-and-gpt-based-descriptions`，版本 7，2025 年发布。
- 实际取得 `medical_descriptions_en.json`，1,990 条描述，908,296 字节；未下载 694 MB 的图像数据包。
- 作者明确说明：将已知疾病类别传入 GPT-4o-mini 的描述生成提示词。文本中也直接包含诊断信息。因此这是诊断条件下的生成描述，不能直接当作不额外提供诊断的实验输入。
- 与本机按官方 ISIC ID 保存的 ISIC2018 图像和 Mask 清单匹配，train 有 430 个匹配，val/test 均为 0。这里只是 ID 覆盖核对，尚未声称完成逐图语义审核。
- 旧 1,886/808 目录的文件已改为数字名称，不能用文件名缺少 ISIC ID 推断没有图像重叠。
- 该第三方数据与 2026 年论文中所称 ISIC-GPT 的同一性未得到证明；论文存在不等于完整标注文件已经取得。

## 适用范围与下一步

这份文本可以考虑用于明确声明“已知诊断辅助分割”的独立实验。若继续原定的文本引导病灶分割，需要取得或制作逐图描述，保证描述生成仅查看图像、不读取 GT Mask 和诊断标签，并先做质量审核及划分对应。

这是数据就绪条件未满足，不是判断 ISIC 永远不能做图文分割。完整证据、匹配 ID、来源元数据和文件 SHA256 见 `text_matching_audit.json`。
