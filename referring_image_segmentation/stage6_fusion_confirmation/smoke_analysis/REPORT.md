# 第六轮：融合与容量的配对种子复验（流程预检，不能用于排名）

ISIC2018 1886/808开发划分，256×256；DINO基线校验后只读复用第三轮；六个新训练。最高验证mean Dice选模，±为训练seed样本标准差。

| Model | Mean Dice | Pooled Dice | mIoU | oIoU | Boundary F1 | HD95 px |
|---|---:|---:|---:|---:|---:|---:|
| dinov3 | 0.891294 | 0.898712 | 0.819122 | 0.816055 | 0.329662 | 15.142424 |
| capacity_concat | 0.863595 | 0.874746 | 0.778580 | 0.777376 | 0.259433 | 19.860786 |
| fusion_concat | 0.866761 | 0.874786 | 0.782830 | 0.777440 | 0.269269 | 20.466058 |

## 同seed主比较

| Comparison | Seed deltas (43/44/45) | Mean delta | Sample SD |
|---|---|---:|---:|
| fusion_concat_minus_dinov3 | [-0.02453315734999717] | -0.024533 | None |
| fusion_concat_minus_capacity_concat | [0.0031652833187948426] | 0.003165 | None |
| capacity_concat_minus_dinov3 | [-0.027698440668792013] | -0.027698 | None |

## 参数与完整推理

| Model | Trainable | Total | Batch1 ms | Batch64 images/s |
|---|---:|---:|---:|---:|
| dinov3 | 506465 | 22093409 | 10.999 | 1859.99 |
| capacity_concat | 638305 | 22225249 | 11.714 | 1691.37 |
| fusion_concat | 632161 | 51697633 | 27.807 | 834.29 |

全部macro/pooled Precision/Recall、Prec@X、Threshold Jaccard和HD95归一化见all_runs.csv与summary.json；三个checkpoint规则全部保留。
纠正/新增/净减少错误及其分母、分区和逐图配对记录见各seed目录。不同seed的像素计数是同一批图上的重复，不视为新的独立病例。
容量匹配仅针对可训练参数；此开发集及这些DINO seeds已有历史使用，不代表独立测试泛化。seed42单列为探索背景，没有混入主均值。
如为smoke，旧DINO训练300轮而新组只有2轮；表格只验证分析流程，不得作性能结论。
