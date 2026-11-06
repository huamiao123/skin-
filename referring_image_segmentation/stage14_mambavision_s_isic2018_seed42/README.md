# MambaVision-S + Swin：ISIC2018 seed42

单模型、单 seed42、冻结 NVIDIA 官方 MambaVision-S-1K，仅训练历史 Adapter、Swin Pixel Decoder 和输出层，300 epochs / 9000 updates。数据、增强、初始化、损失、选模和指标直接复用原项目；不改 stage1–13，不自动上传 GitHub。

- [冻结协议](PROTOCOL.md)、[实际配置与环境](config.json)、[真实数据预检](preflight.json)。
- [仓库十个指定文件核查](repository_inspection.json)：与 GitHub commit `833430d15a58e01889f886cddaec164ffe64d1fb` 逐字节一致。
- 官方 HF revision `e4f9ea93877fba6f1e2d231fd0f0a391368ad895`。未修改官方模型代码；strict 加载并逐张量验证预训练参数和缓冲区，权重远端 LFS 身份见 official/remote_weight_identity.json。
- 原 ISIC2018 图像/Mask 路径为 `../data/isic2018`，1886/808 开发划分；无需下载 ISIC2016 或文本数据。
- 依赖复用历史 PyTorch2.2.2+cu121、timm1.0.22、Transformers4.44.2；官方 mamba-ssm1.2.2 CUDA wheel 安装至本阶段 runtime。版本/扩展实际路径与 SHA 均记录于 config.json。

## 恢复环境和执行

在独立环境恢复 requirements.txt；PyTorch 需 CUDA 构建，官方 mamba wheel 适用于 CPython3.11、torch2.2、CXX11 ABI False。原机只向本目录安装 mamba-ssm，不修改全局或历史依赖。原 wheel URL/SHA 见 official/mamba_wheel.json，可用 `python -m pip install --no-deps --target runtime <wheel路径>` 安装。官方 MambaVision 调用 torch Conv1d，未要求额外 causal-conv1d 扩展，真实 scan CUDA 前向已验收。

恢复数据和依赖后在本目录运行：

```bash
python -u prepare_official.py
python -u preflight.py
python -u launch.py
```

监督入口会隔离执行两轮 smoke 与跨进程恢复预检，再从头正式训练、三种 checkpoint 导出、完整模型计时、报告和独立完成审计。已存在完成记录属于历史存档，不等同于另一台机器已经复现；跨机器复现请建立独立副本、记录路径调整和新的 provenance，不覆盖本目录历史结果。

正式断点恢复命令：`python -u train.py`，只读本阶段 results/latest.pth，核验模型/optimizer/scheduler/全部 RNG、配置与更新数，从下一完整 epoch 继续。或重新启动 `python -u launch.py`，自动接续并完成最终分析。单独导出用 `python -u evaluate.py`，随后 `python -u benchmark.py`、`python -u analyze.py`、`python -u validate.py`。

## 结果与本地资产

进度 run_status.json、results/status.json、train.log；逐轮 history.json / results/history.json；最终 analysis/summary.json、REPORT.md、per_image.csv、learning_curves.png。正式 best_dice/best_loss/last 三权重和 latest 恢复点均保留。

三种 checkpoint 的808张256网格概率和二值Mask保存在 results/{best_dice,best_loss,last}/。含原图的改善、退步、失败示例仅在 analysis/examples/；不自动公开原图、权重、下载的模型或运行依赖。主要对照只读 stage4 同 seed/同 Swin/同选模的 DINOv3 与 VMamba，不重跑历史模型。

只有正式300轮/9000更新、预训练身份、冻结状态、三选模规则、808图完整指标复核与报告一致性均通过，run_status 才会标 complete。

## 完成结果

300轮/9000更新及全部独立校验已通过。主权重按验证macro Dice选模，完整结果见 [报告](analysis/REPORT.md) 与 [summary.json](analysis/summary.json)。

best_dice 第159轮，开发验证 macro Dice 88.3598%，pooled Dice 88.9288%。单seed开发验证结果，不作独立测试或架构机制归因。
