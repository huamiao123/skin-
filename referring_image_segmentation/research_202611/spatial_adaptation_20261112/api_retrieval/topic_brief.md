# 主题调研简报：Mamba medical image segmentation local spatial enhancement

## 一句话结论

`Mamba medical image segmentation local spatial enhancement` 的检索结果显示，该主题的主线不是单纯的多智能体一致性，而是在编队、避障、通信受限、攻击或不确定性下维持通信/感知图的连通裕度。当前语料包含 33 条去重记录，年份覆盖 2024-2026，主要落在连通度保持与代数连通度控制、编队、队形与一致性控制、避障、安全约束与有限感知、事件触发、通信受限与分布式实现、鲁棒性、攻击与不确定系统等方向。

## 背景与问题定义

本简报基于可复现学术 API 检索生成，用于快速理解一个研究主题的核心问题、代表工作和后续阅读路径。在多智能体系统语境下，连通度控制通常关注通信图或感知图在运动、编队、避障、攻击、事件触发通信等条件下能否保持足够连通，常见度量包括图连通性、Laplacian 的 algebraic connectivity、邻接边保持、有限感知半径内的连通维护等。

## 检索策略与覆盖范围

- 原始查询：`Mamba medical image segmentation local spatial enhancement`
- 年份范围：2024 到 2026
- 去重后文献数：33
- 各源返回数量：openalex=15, arxiv=5, crossref=15
- 说明：该结果依赖 API 覆盖、查询式质量和元数据完整度；最终用于论文写作前仍需人工筛选全文。

## 技术路线分类

### 连通度保持与代数连通度控制

该类包含 11 条候选文献，覆盖 2024-2026。排名靠前的代表工作是《Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation》，可作为进入该分支的入口。

- Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation (2024), Ziyang Wang, Jian-Qing Zheng, Yichi Zhang, et al., arXiv (Cornell University); citations=350, DOI/ID=10.48550/arxiv.2402.05079.
- FMTNet: A Fourier‐Mamba–Transformer Enhanced Network for Medical Image Segmentation (2026), Shaoqiang Wang, Guiling Shi, Yuanyuan Zhang, et al., CAAI Transactions on Intelligence Technology; citations=2, DOI/ID=10.1049/cit2.70141.
- U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation (2024), Jun Ma, Li, Feifei, Bo Wang, arXiv (Cornell University); citations=1267, DOI/ID=10.48550/arxiv.2401.04722.
- DPM-UNet: A Mamba-Based Network with Dynamic Perception Feature Enhancement for Medical Image Segmentation (2025), Shangyu Xu, Xiaohang Liu, Hongsheng Lei, et al., Sensors; citations=1, DOI/ID=10.3390/s25227053.
- PF-Mamba-UNet: An Anatomical-Prior-Guided Medical Image Segmentation Network Based on Mamba and Spatial-Channel Collaborative Attention Fusion (2026), Leyi Xiao, Kangshun Liao, Baoxian Zhou, et al., crossref; citations=0, DOI/ID=10.2139/ssrn.7553347.

### 编队、队形与一致性控制

该类包含 9 条候选文献，覆盖 2024-2026。排名靠前的代表工作是《Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation》，可作为进入该分支的入口。

- Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation (2024), Ziyang Wang, Jian-Qing Zheng, Yichi Zhang, et al., arXiv (Cornell University); citations=350, DOI/ID=10.48550/arxiv.2402.05079.
- FMTNet: A Fourier‐Mamba–Transformer Enhanced Network for Medical Image Segmentation (2026), Shaoqiang Wang, Guiling Shi, Yuanyuan Zhang, et al., CAAI Transactions on Intelligence Technology; citations=2, DOI/ID=10.1049/cit2.70141.
- VM-UNet: Vision Mamba UNet for Medical Image Segmentation (2025), Jiacheng Ruan, Jincheng Li, Suncheng Xiang, ACM Transactions on Multimedia Computing Communications and Applications; citations=1323, DOI/ID=10.1145/3767748.
- DPM-UNet: A Mamba-Based Network with Dynamic Perception Feature Enhancement for Medical Image Segmentation (2025), Shangyu Xu, Xiaohang Liu, Hongsheng Lei, et al., Sensors; citations=1, DOI/ID=10.3390/s25227053.
- Mamba-Sea: A Mamba-Based Framework With Global-to-Local Sequence Augmentation for Generalizable Medical Image Segmentation (2025), Zihan Cheng, Jintao Guo, Jian Zhang, et al., IEEE Transactions on Medical Imaging; citations=44, DOI/ID=10.1109/tmi.2025.3564765.

### 避障、安全约束与有限感知

该类包含 1 条候选文献，覆盖 2026-2026。排名靠前的代表工作是《Large-Kernel Spatial-Frequency Fusion Deep Mamba Network for Remote Sensing Image Segmentation》，可作为进入该分支的入口。

- Large-Kernel Spatial-Frequency Fusion Deep Mamba Network for Remote Sensing Image Segmentation (2026), ChenZhe Xia, Eziz Tursun, crossref; citations=0, DOI/ID=10.2139/ssrn.6190562.

### 事件触发、通信受限与分布式实现

该类包含 2 条候选文献，覆盖 2025-2025。排名靠前的代表工作是《VM-UNet: Vision Mamba UNet for Medical Image Segmentation》，可作为进入该分支的入口。

- VM-UNet: Vision Mamba UNet for Medical Image Segmentation (2025), Jiacheng Ruan, Jincheng Li, Suncheng Xiang, ACM Transactions on Multimedia Computing Communications and Applications; citations=1323, DOI/ID=10.1145/3767748.
- SCM-UNet: Spatial-channel Mamba UNet for medical image segmentation (2025), Haijie Yan, Qiuhong Hong, Shoulin Wei, et al., Digital Signal Processing; citations=8, DOI/ID=10.1016/j.dsp.2025.105550.

### 鲁棒性、攻击与不确定系统

该类包含 5 条候选文献，覆盖 2024-2026。排名靠前的代表工作是《FMTNet: A Fourier‐Mamba–Transformer Enhanced Network for Medical Image Segmentation》，可作为进入该分支的入口。

- FMTNet: A Fourier‐Mamba–Transformer Enhanced Network for Medical Image Segmentation (2026), Shaoqiang Wang, Guiling Shi, Yuanyuan Zhang, et al., CAAI Transactions on Intelligence Technology; citations=2, DOI/ID=10.1049/cit2.70141.
- Mamba-Sea: A Mamba-Based Framework With Global-to-Local Sequence Augmentation for Generalizable Medical Image Segmentation (2025), Zihan Cheng, Jintao Guo, Jian Zhang, et al., IEEE Transactions on Medical Imaging; citations=44, DOI/ID=10.1109/tmi.2025.3564765.
- SFma-Unet: A Mamba-Based Spatial-Frequency Fusion Network for Medical Image Segmentation (2025), Zhanpeng Liu, Yuqiang Zhang, Bin Wang, et al., IEEE International Conference on Acoustics Speech and Signal Processing; citations=4, DOI/ID=10.1109/icassp49660.2025.10889117.
- Fréchet Radiomic Distance (FRD): A Versatile Metric for Comparing Medical Imaging Datasets (2024), Nicholas Konz, Richard Osuala, Preeti Verma, et al., arXiv; citations=0, DOI/ID=10.1016/j.media.2026.103943.
- Test-time generative augmentation for medical image segmentation (2024), Xiao Ma, Yuhui Tao, Zetian Zhang, et al., arXiv; citations=0, DOI/ID=10.1016/j.media.2025.103902.

### 优化、MPC 与性能约束

该类包含 9 条候选文献，覆盖 2024-2026。排名靠前的代表工作是《Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation》，可作为进入该分支的入口。

- Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation (2024), Ziyang Wang, Jian-Qing Zheng, Yichi Zhang, et al., arXiv (Cornell University); citations=350, DOI/ID=10.48550/arxiv.2402.05079.
- FMTNet: A Fourier‐Mamba–Transformer Enhanced Network for Medical Image Segmentation (2026), Shaoqiang Wang, Guiling Shi, Yuanyuan Zhang, et al., CAAI Transactions on Intelligence Technology; citations=2, DOI/ID=10.1049/cit2.70141.
- LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation (2024), Weibin Liao, Yinghao Zhu, Xinyuan Wang, et al., arXiv (Cornell University); citations=238, DOI/ID=10.48550/arxiv.2403.05246.
- PF-Mamba-UNet: An Anatomical-Prior-Guided Medical Image Segmentation Network Based on Mamba and Spatial-Channel Collaborative Attention Fusion (2026), Leyi Xiao, Kangshun Liao, Baoxian Zhou, et al., crossref; citations=0, DOI/ID=10.2139/ssrn.7553347.
- HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation (2024), Jiashu Xu, arXiv (Cornell University); citations=74, DOI/ID=10.48550/arxiv.2405.05007.

### 其他相关工作

该类包含 13 条候选文献，覆盖 2024-2026。排名靠前的代表工作是《LKM-UNet: Large Kernel Vision Mamba UNet for Medical Image Segmentation》，可作为进入该分支的入口。

- LKM-UNet: Large Kernel Vision Mamba UNet for Medical Image Segmentation (2024), Jinhong Wang, Jintai Chen, Danny Z. Chen, et al., Lecture notes in computer science; citations=161, DOI/ID=10.1007/978-3-031-72111-3_34.
- HLSM-UNet:Hybrid Local Spatial Mamba UNet for Medical Image Segmentation (2024), Xude Zhang, Xiaoping Wu, Changzhen Zhang, et al., 2024 International Conference on Neuromorphic Computing (ICNC); citations=2, DOI/ID=10.1109/icnc64304.2024.10987734.
- Vision Mamba and xLSTM-UNet for medical image segmentation (2025), Xin Zhong, Gehao Lu, Hao Li, Scientific Reports; citations=32, DOI/ID=10.1038/s41598-025-88967-5.
- GL-Mamba UNet: Global-to-Local Context-Guided Vision Mamba for Medical Image Segmentation (2026), Longhui Wang, Jie Li, Yue Ruan, Proceedings of the 2026 6th International Conference on Bioinformatics and Intelligent Computing; citations=0, DOI/ID=10.1145/3809986.3810131.
- ASRM-Net: Multi-Scale Feature Enhancement with Adaptive Spatial Recalibration for Medical Image Segmentation (2026), Xinyang Chen, 2026 7th International Conference on Computer Vision, Image and Deep Learning (CVIDL); citations=0, DOI/ID=10.1109/cvidl70130.2026.11637809.

## 代表性论文表

| 角色 | 论文 | 年份 | 来源/期刊会议 | 为什么重要 |
| --- | --- | --- | --- | --- |
| 高被引基础 | Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation | 2024 | arXiv (Cornell University) | 引用量较高，适合作为理论背景或问题定义入口。 |
| 近期进展 | FMTNet: A Fourier‐Mamba–Transformer Enhanced Network for Medical Image Segmentation | 2026 | CAAI Transactions on Intelligence Technology | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 高被引基础 | VM-UNet: Vision Mamba UNet for Medical Image Segmentation | 2025 | ACM Transactions on Multimedia Computing Communications and Applications | 引用量较高，适合作为理论背景或问题定义入口。 |
| 高被引基础 | LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation | 2024 | arXiv (Cornell University) | 引用量较高，适合作为理论背景或问题定义入口。 |
| 高被引基础 | U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation | 2024 | arXiv (Cornell University) | 引用量较高，适合作为理论背景或问题定义入口。 |
| 近期进展 | DPM-UNet: A Mamba-Based Network with Dynamic Perception Feature Enhancement for Medical Image Segmentation | 2025 | Sensors | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 近期进展 | Mamba-Sea: A Mamba-Based Framework With Global-to-Local Sequence Augmentation for Generalizable Medical Image Segmentation | 2025 | IEEE Transactions on Medical Imaging | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 高被引基础 | LKM-UNet: Large Kernel Vision Mamba UNet for Medical Image Segmentation | 2024 | Lecture notes in computer science | 引用量较高，适合作为理论背景或问题定义入口。 |
| 近期进展 | HLSM-UNet:Hybrid Local Spatial Mamba UNet for Medical Image Segmentation | 2024 | 2024 International Conference on Neuromorphic Computing (ICNC) | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 近期进展 | PF-Mamba-UNet: An Anatomical-Prior-Guided Medical Image Segmentation Network Based on Mamba and Spatial-Channel Collaborative Attention Fusion | 2026 | crossref | 反映近年约束控制、鲁棒性或应用场景的新变化。 |

## 研究脉络

- 2023-2026：约 33 条；代表工作为《VM-UNet: Vision Mamba UNet for Medical Image Segmentation》，该阶段重点可从 高被引基础 切入。

## 主要方法对比

| 方法族 | 典型关注点 | 优势 | 常见限制 |
| --- | --- | --- | --- |
| Potential/barrier/navigation functions | 保持已有边、避免距离约束破坏 | 直观，适合编队与避障 | 参数调节和局部极值问题明显 |
| Algebraic connectivity / Fiedler value | 直接优化或估计图连通度 | 能刻画全局连通裕度 | 分布式估计和可扩展性是难点 |
| Event-triggered connectivity control | 减少通信和控制更新 | 适合资源受限网络 | 需要证明无 Zeno 行为和鲁棒性 |
| Prescribed performance / constrained control | 把连通、碰撞、误差约束统一处理 | 约束表达清晰 | 对模型假设和扰动界较敏感 |
| MPC / optimization-based methods | 在预测窗口内优化连通和任务性能 | 可处理多目标权衡 | 计算量和实时性压力较高 |

## 重要论文阅读顺序

1. VM-UNet: Vision Mamba UNet for Medical Image Segmentation (2025), Jiacheng Ruan, Jincheng Li, Suncheng Xiang, ACM Transactions on Multimedia Computing Communications and Applications; citations=1323, DOI/ID=10.1145/3767748.
2. U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation (2024), Jun Ma, Li, Feifei, Bo Wang, arXiv (Cornell University); citations=1267, DOI/ID=10.48550/arxiv.2401.04722.
3. Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation (2024), Ziyang Wang, Jian-Qing Zheng, Yichi Zhang, et al., arXiv (Cornell University); citations=350, DOI/ID=10.48550/arxiv.2402.05079.
4. LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation (2024), Weibin Liao, Yinghao Zhu, Xinyuan Wang, et al., arXiv (Cornell University); citations=238, DOI/ID=10.48550/arxiv.2403.05246.
5. LKM-UNet: Large Kernel Vision Mamba UNet for Medical Image Segmentation (2024), Jinhong Wang, Jintai Chen, Danny Z. Chen, et al., Lecture notes in computer science; citations=161, DOI/ID=10.1007/978-3-031-72111-3_34.
6. HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation (2024), Jiashu Xu, arXiv (Cornell University); citations=74, DOI/ID=10.48550/arxiv.2405.05007.
7. Mamba-Sea: A Mamba-Based Framework With Global-to-Local Sequence Augmentation for Generalizable Medical Image Segmentation (2025), Zihan Cheng, Jintao Guo, Jian Zhang, et al., IEEE Transactions on Medical Imaging; citations=44, DOI/ID=10.1109/tmi.2025.3564765.
8. Vision Mamba and xLSTM-UNet for medical image segmentation (2025), Xin Zhong, Gehao Lu, Hao Li, Scientific Reports; citations=32, DOI/ID=10.1038/s41598-025-88967-5.
9. FMTNet: A Fourier‐Mamba–Transformer Enhanced Network for Medical Image Segmentation (2026), Shaoqiang Wang, Guiling Shi, Yuanyuan Zhang, et al., CAAI Transactions on Intelligence Technology; citations=2, DOI/ID=10.1049/cit2.70141.
10. DPM-UNet: A Mamba-Based Network with Dynamic Perception Feature Enhancement for Medical Image Segmentation (2025), Shangyu Xu, Xiaohang Liu, Hongsheng Lei, et al., Sensors; citations=1, DOI/ID=10.3390/s25227053.

## 局限与开放问题

- 从理论保证走向真实机器人/无人系统实验仍是关键缺口，尤其是有限感知、通信丢包和动态障碍同时存在时。
- 代数连通度类方法需要更可扩展的分布式估计与控制实现，避免全局谱信息成为隐含假设。
- 连通保持、碰撞避免、队形精度和能耗之间存在多目标冲突，需要更清晰的权衡机制。
- 事件触发和通信受限控制需要同时说明无 Zeno、鲁棒性和实际网络延迟下的性能。
- 面向攻击或故障的连通度控制还需要把安全检测、拓扑重构和控制律设计更紧密地结合。

## 网页补充建议

- `"Mamba medical image segmentation local spatial enhancement" PDF`
- `"multi-agent systems" "connectivity preservation" control`
- `"algebraic connectivity" "multi-agent" formation control`
- `"connectivity maintenance" "multi-robot" consensus`
- `"Fiedler eigenvalue" "distributed connectivity control"`
- `"connectivity preserving formation control" author page`
- `"multi-agent connectivity control" arXiv`

将网页搜索结果单独记录到 `web_supplement.md`，只收录作者主页、项目页、开放 PDF、近期预印本、实验室页面和可靠综述资料；不要把 Google Scholar 抓取结果混入 API 检索语料。

## 可复核来源

- `papers.json`：归一化元数据和排序分数。
- `papers.csv`：适合人工筛选的表格。
- `papers.bib`：BibTeX 导入文件。
- `search_log.md`：检索参数、来源计数、去重和跳过信息。
