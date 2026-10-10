# 独立 CNN + 普通 Transformer，seed 42

不基于 EGE，不使用 Swin。普通卷积 U-Net 主路与独立标准全局多头自注意力 Transformer 分支，在 32/16/8/4 四个尺度融合：CNN + sigmoid(gamma) × Conv1x1(Transformer)，gamma 初始 0.1。

Transformer：patch8，维度64/128/256/512，每层2个标准 TransformerEncoderLayer，头数2/4/8/16，MLP ratio4，可学习绝对位置编码，无局部窗口、无空间缩减注意力，全部从头训练。CNN：普通双卷积+GroupNorm+GELU、池化编码器、插值+跳连解码器，不含 EGE 模块或深监督。输入先池化至128进入CNN，预测插值至256。

ISIC2018 已核对的1886/808开发划分，256输入，seed42，300epochs，物理batch64，FP32，CNN/fusion LR1e-3、Transformer LR1e-4，AdamW wd0.01，cosine300 eta_min1e-5，clip1，BCE+Dice。水平/垂直翻转与每次重新采样旋转概率均0.5，mask最近邻。最小验证loss选模型，最终batch1评估。smoke与正式结果完全分开，支持每epoch断点恢复。

本目录 results、smoke 和日志为已完成实验的实际文件副本。训练完成300轮，用时27分35秒；最小验证loss选中第59轮，最终验证 Dice 89.2243%、IoU 80.5449%。数据集和模型权重不随源码提交。

## 源码位置与复现

- `source/cnn_transformer.py`：完整独立 CNN＋普通 Transformer 模型。
- `train.py`：可移植的训练入口，使用环境变量 `ISIC2018_DATA_ROOT` 设置数据目录。
- `launch.py`：后台训练监督入口。
- `source/utils.py`、`source/datasets/dataset.py`：沿用 seed42 已修正的数据增强、loss 和数据读取工具；模型不导入 EGE 或 Swin。
- `snapshot/train.py`、`snapshot/launch.py`：实际运行的原始脚本；`source_audit.json` 记录原始脚本和模型源码哈希。可移植入口只修改路径解析。
- `results/cnn_transformer/`：正式配置、300轮历史、最佳模型最终评估；`smoke/`：独立测速结果。

需要 NVIDIA CUDA GPU。数据目录应包含 `train/images`、`train/masks`、`val/images`、`val/masks`，并使用此次1886/808划分；图像与mask按样本ID匹配。

```bash
cd work/experiments/cnn_transformer_seed42_20261010
pip install -r requirements.txt
export ISIC2018_DATA_ROOT=/your/path/to/isic2018_data
export PYTHONHASHSEED=42 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
```

归档结果已有 `DONE.json`，避免被覆盖，重新训练请将本目录的 `source/`、`train.py`、`launch.py`、`requirements.txt` 复制到一个新目录，再在新目录运行：

```bash
python -u train.py cnn_transformer
# 或后台监督运行
nohup python -u launch.py > supervisor.log 2>&1 &
```
