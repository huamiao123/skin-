# MosMedData+ 文本信息增益实验

三组统一 DINOv3-S/16 + Swin：E0 无文本普通 Mask Head；E1 固定 `lung infection`；E2 逐图医学文本。正式种子42/43/44。CLIP文本和DINO均冻结，Adapter/Decoder/Head独立重新训练。

## 数据核查

已下载2729对图像/Mask，三张文本表2183/273/273。原始划分有整段CT切片跨集合。按可追溯CT体积和radiopaedia患者ID归组，全部bjorke来源保守归为一个未知患者桶，按test>val>train整体迁移；调整236条记录，最终 **2004 train / 216 val / 509 test**，来源组54/7/9，无精确重复图像。

原始集合含99张具有低灰度值的Mask，遵循LViT作者加载器 `mask > 0` 二值化。共有115种逐图描述，均为灰度图，1张空Mask在测试集。来源组不重叠；跨来源患者身份及bjorke内部患者映射仍不明确，因此不宣称完整患者级独立。

文本由公开Kaggle镜像取得，未与作者OneDrive原始文件做逐字节验证。LViT论文说明文本含病灶数量/位置、且用Mask质检；本实验评价额外文本标注条件下的分割能力，不等同于原始临床报告或通用RIS。

## 执行与进度

```bash
cd /home/featurize/referring_image_segmentation/stage10_mosmed_text
python -u launch.py
```

顺序：数据检查 → CLIP缓存 → 显存/梯度预检 → 两轮中断恢复及原尺寸评价检查 → seed42三组各50轮预实验 → 按事先规则选择统一100/150正式轮数 → 42/43/44各三组 → 全部选模锁定后测试。

配置见PROTOCOL.md；进度见run_status.json、pilot.log、train_seed42/43/44.log。最终汇总analysis/REPORT.md及summary.json。仅保留正式最佳权重9个及预实验最佳3个，完成后删除恢复和smoke权重。

原始图像放在 `/home/featurize/rsi_data/benchmarks/mosmed_plus`。历史实验文件未改动。冻结特征、数据和文本均可校验来源；代码依赖既有stage1/2/3的模型及stage8的CLIP资产/runtime。

## 当前状态：用户暂停

2026-10-20 用户要求停止其他 seeds 并核查低Dice。Seed42完成100轮；Seed43已停在第60轮部分更新，保留恢复点；Seed44未启动。低Dice只读核查见 `diagnostics/low_dice_20261020/FINDINGS.md`。恢复训练须由用户明确要求。
