# 数值复现说明

本轮与第一轮相同，设置 Python/NumPy/PyTorch seed 42、cudnn.benchmark=False、cudnn.deterministic=True；训练顺序和几何增强使用独立、固定的逐 epoch Generator，Decoder 和 Adapter 初始权重 hash 均检查。

没有启用 `torch.use_deterministic_algorithms(True)`。本地 PyTorch 2.2.2 的 `torch.nn.functional.interpolate` 文档明确说明 CUDA 上可能产生非确定性梯度。Decoder 和 Adapter 均包含双线性上采样，因此不能把固定 seed 解释为逐位可复现。

独立 smoke 与正式运行的配置除 epochs=2/300 外一致，但前两轮数值并非完全相同。例如 DINOv3 第一轮 train loss 差约 1.23e-5；VMamba 第一轮差约 2.55e-4。这一检查未通过逐位相等要求，因而本轮只声称初始化、样本/增强、训练预算与超参数一致，不声称完全相同的数值轨迹。CUDA 非确定性是已知可能来源，并未逐算子定位全部差异。

本轮沿用第一轮的数值设置，不在中途改变算子或重启以挑选更高分结果。正式候选方案需要至少三个 seeds 确认；需要严格确定性时，应为所有对照统一新建协议并重新运行。
