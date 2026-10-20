# ISIC2016 seed42 三组文本对照

新增E0无文本 DINOv3+Swin；E1固定 `skin lesion`、E2逐图颜色/形状属性文本，核验并复用stage9已完成的同条件结果，不重复训练。

806训练／90验证／379测试，256输入，100轮，batch16，原亮度/对比度增强及AdamW设置。只有seed42，无VMamba，无错误文本实验。患者独立性未验证，测试集已用于既往开发。

执行 `python -u launch.py`；进度见run_status.json、train.log；最终三组结果见analysis/REPORT.md与summary.json。原始数据、文本、历史权重只读复用，不复制大型资产。MosMed保持用户暂停状态。

## 已完成

E0已完成seed42、100轮、5100次更新，选中第21轮，验证macro Dice92.828%。E1/E2历史验证及测试全部宏平均/汇总指标重放一致，六张结果表由逐图计数独立核验。

| 组别 | 测试macro Dice | 测试pooled Dice |
|---|---:|---:|
| E0_visual | 90.306% | 91.729% |
| E1_fixed_text | 90.180% | 91.573% |
| E2_image_text | 90.120% | 91.381% |

E2-E1为-0.060个百分点，图像配对区间跨0；本次未证明逐图属性文本带来额外收益。只有一个seed，不证明三组性能存在稳定差异。

完整解释见analysis/FINDINGS.md，学习曲线见analysis/learning_curves.png，固定ID对照图见analysis/fixed_id_prediction_panel.png。保留E0最佳权重，恢复及smoke权重已清理；历史权重保持原样。MosMed保持暂停。
