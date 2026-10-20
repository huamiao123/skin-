# ISIC2016 seed42：Decoder之前的词级交互

文本只进入Decoder之前的四层Cross-Attention，Decoder之后是普通Mask Head。固定文本和逐图文本均新训练100轮；另外三个历史对照只读复用。

| Model | 新训练 | 最佳轮 | Val macro Dice | Test macro Dice | Test pooled Dice | Test BF1 | Test HD95 px |
|---|---|---:|---:|---:|---:|---:|---:|
| predecoder_fixed_text | 是 | 21 | 0.927175 | 0.903684 | 0.918839 | 0.340409 | 95.376 |
| predecoder_image_text | 是 | 25 | 0.925823 | 0.903097 | 0.916270 | 0.337375 | 84.799 |
| E0_visual | 历史 | 21 | 0.928283 | 0.903062 | 0.917288 | 0.338750 | 94.482 |
| E1_fixed_text | 历史 | 30 | 0.925196 | 0.901800 | 0.915731 | 0.334443 | 75.645 |
| E2_image_text | 历史 | 31 | 0.924761 | 0.901201 | 0.913811 | 0.326793 | 77.462 |

主要比较是predecoder_image_text - predecoder_fixed_text，解释同容量下逐图属性的额外价值。
与旧late-image比较是整体机制替换，包含位置/词级表示/输出头差异，不作为单独位置收益。
只有一个seed；既往使用过的测试集、未验证患者独立性，不宣称稳定提升或全新泛化验证。
完整macro/pooled区域指标、Prec@、threshold Jaccard、边界指标、参数量与延迟见summary.json。
