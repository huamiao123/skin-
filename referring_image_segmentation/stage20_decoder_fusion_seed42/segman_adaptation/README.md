# MambaVision-S + SegMAN decoder 适配实验

这是独立实验，不在当前 Swin 上叠加多个论文模块。冻结同一 MambaVision-S，使用 SegMAN-small 的 MMSCoPE 作为整个分割解码器，输入通道调整为 `[96,192,384,768]`，输出调整为 1 通道。官方代码这一解码路径读取四个输入，但只使用 c2/c3/c4。上游配置的头宽为 144、投影宽为 288，且自带 SyncBN、dropout；因此不是与 Swin 等参数或相同归一化的微小消融。

本目录不依赖 mmcv/mmseg，运行时从相邻 `references/segman_decoder_source.py` 提取所需原始定义，给 `BaseDecodeHead`、`ConvModule` 提供有限的独立兼容实现，原始 `oflex` CUDA kernel 则替换为项目现有标准 Mamba selective scan。BF16 输入上先转换到 FP32，保持 oflex=True 路径的 FP32 输出类型。单 GPU SyncBN、卷积初始化和算子实现都可能与官方框架有差异，所以明确标记为**适配版，不是完整官方复现**。上游 CUDA oflex 本机不可用，不能声称已与其输出逐元素等价。

预检包括官方 Triton 四方向扫描与直接张量公式的前向/梯度比较、当前 CUDA selective scan 与独立 PyTorch 参考的前向/梯度比较、真实图像的四尺度编码器特征和解码头前后向，以及 batch 64 的真实 GPU 显存检查。`preflight.json` 和 `capacity_preflight.json` 保存实际结果。因为头中有 SyncBN，不能把小 microbatch 梯度累积宣称和 batch 64 等价；本次实际使用完整 64（尾 batch 保留）。

数据、seed、300 epochs、优化器、学习率曲线、几何增强、损失、BF16、验证选模与上一目录的配对 Swin 一致。使用现成的 MambaVision + Swin 配对基线，不重新训练基线。报告将区分总体 decoder 能力差异与无法隔离的归一化、dropout、参数规模差异；不称之为扫描方式的单因素消融。

先独立执行两轮 smoke、验证从第一轮完整恢复，再由 `run_pipeline.py` 等待第一组 AXIS 训练、最终评估和 GPU 测速结束，然后独占 GPU 完成此实验及比较。`run_status.json` 指示等待、训练、评估或失败，正式结果不会使用 smoke 权重。完整命令：

```
python preflight.py
python capacity_preflight.py
python train_segman.py --smoke --stop-after-epoch 1
python train_segman.py --smoke
python run_pipeline.py
```

官方来源固定于 commit `9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8`；本地扫描及 head 源码的 SHA256 见 `preflight.json`。公开分享相关上游源码前需要核验根目录授权与源码文件头中单独标注的 NVIDIA 许可。
