# 主题调研简报：Mamba visual segmentation spatial adapter

## 一句话结论

`Mamba visual segmentation spatial adapter` 的检索结果显示，该主题的主线不是单纯的多智能体一致性，而是在编队、避障、通信受限、攻击或不确定性下维持通信/感知图的连通裕度。当前语料包含 86 条去重记录，年份覆盖 2024-2026，主要落在连通度保持与代数连通度控制、编队、队形与一致性控制、避障、安全约束与有限感知、事件触发、通信受限与分布式实现、鲁棒性、攻击与不确定系统等方向。

## 背景与问题定义

本简报基于可复现学术 API 检索生成，用于快速理解一个研究主题的核心问题、代表工作和后续阅读路径。在多智能体系统语境下，连通度控制通常关注通信图或感知图在运动、编队、避障、攻击、事件触发通信等条件下能否保持足够连通，常见度量包括图连通性、Laplacian 的 algebraic connectivity、邻接边保持、有限感知半径内的连通维护等。

## 检索策略与覆盖范围

- 原始查询：`Mamba visual segmentation spatial adapter`
- 年份范围：2024 到 2026
- 去重后文献数：86
- 说明：该结果依赖 API 覆盖、查询式质量和元数据完整度；最终用于论文写作前仍需人工筛选全文。

## 技术路线分类

### 连通度保持与代数连通度控制

该类包含 31 条候选文献，覆盖 2024-2026。排名靠前的代表工作是《Spatial-Mamba: Effective Visual State Space Models via Structure-aware State Fusion》，可作为进入该分支的入口。

- Spatial-Mamba: Effective Visual State Space Models via Structure-aware State Fusion (2024), Chaodong Xiao, Minghan Li, Zhengqiang Zhang, et al., arXiv (Cornell University); citations=45, DOI/ID=10.48550/arxiv.2410.15091.
- Sigma-Fusion: Synergistic Semantic Segmentation of Optical and SAR Remote Sensing Images Based on a Siamese Mamba Architecture (2026), Lun Ma, Yanqi Zhao, Yuhuan Niu, Lecture notes in computer science; citations=0, DOI/ID=10.1007/978-981-92-3520-9_42.
- A Comprehensive Survey of Mamba Architectures for Medical Image Analysis: Classification, Segmentation, Restoration and Beyond (2024), Shubhi Bansal, Sreeharish Arikrishnan, J. Madhava Prasath, et al., arXiv (Cornell University); citations=48, DOI/ID=10.48550/arxiv.2410.02362.
- Vision Mamba: A Comprehensive Survey and Taxonomy (2025), Xiao Liu, Chenxu Zhang, Fuxiang Huang, et al., IEEE Transactions on Neural Networks and Learning Systems; citations=262, DOI/ID=10.1109/tnnls.2025.3610435.
- SAM-Mamba: Mamba Guided SAM Architecture for Generalized Zero-Shot Polyp Segmentation (2025), Tapas Kumar Dutta, Snehashis Majhi, Deepak Ranjan Nayak, et al., IEEE Winter Conference on Applications of Computer Vision; citations=19, DOI/ID=10.1109/wacv61041.2025.00457.

### 编队、队形与一致性控制

该类包含 28 条候选文献，覆盖 2024-2026。排名靠前的代表工作是《Spatial-Mamba: Effective Visual State Space Models via Structure-aware State Fusion》，可作为进入该分支的入口。

- Spatial-Mamba: Effective Visual State Space Models via Structure-aware State Fusion (2024), Chaodong Xiao, Minghan Li, Zhengqiang Zhang, et al., arXiv (Cornell University); citations=45, DOI/ID=10.48550/arxiv.2410.15091.
- PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition (2024), Chenhongyi Yang, Zehui Chen, Miguel Espinosa, et al., arXiv (Cornell University); citations=183, DOI/ID=10.48550/arxiv.2403.17695.
- MFil-Mamba: Multi-Filter Scanning for Spatial Redundancy-Aware Visual State Space Models (2026), Puskal Khadka, KC Santosh, arXiv (Cornell University); citations=0, DOI/ID=10.48550/arxiv.2603.20074.
- A Hybrid Mamba-SAM Architecture for Efficient 3D Medical Image Segmentation (2026), Mohammadreza Gholipour Shahraki, Mehdi Rezaeian, Mohammad Ghasemzadeh, arXiv (Cornell University); citations=0, DOI/ID=10.48550/arxiv.2602.00650.
- 3D Segment Anything Model With Visual Mamba for Diagnosing Placenta Accreta Spectrum (2026), Yuliang Zhang, Fang He, Lulu Peng, et al., IEEE Transactions on Image Processing; citations=0, DOI/ID=10.1109/tip.2026.3694185.

### 避障、安全约束与有限感知

该类包含 17 条候选文献，覆盖 2024-2026。排名靠前的代表工作是《A Survey on Visual Mamba》，可作为进入该分支的入口。

- A Survey on Visual Mamba (2024), Hanwei Zhang, Ying Zhu, Dan Wang, et al., Applied Sciences; citations=265, DOI/ID=10.3390/app14135683.
- Sigma-Fusion: Synergistic Semantic Segmentation of Optical and SAR Remote Sensing Images Based on a Siamese Mamba Architecture (2026), Lun Ma, Yanqi Zhao, Yuhuan Niu, Lecture notes in computer science; citations=0, DOI/ID=10.1007/978-981-92-3520-9_42.
- Mamba-Adaptor: State Space Model Adaptor for Visual Recognition (2025), Fei Xie, Jiahao Nie, Yong Tan, et al., IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=8, DOI/ID=10.1109/cvpr52734.2025.01874.
- Vision Mamba: A Comprehensive Survey and Taxonomy (2025), Xiao Liu, Chenxu Zhang, Fuxiang Huang, et al., IEEE Transactions on Neural Networks and Learning Systems; citations=262, DOI/ID=10.1109/tnnls.2025.3610435.
- Frequency-Assisted Mamba for Remote Sensing Image Super-Resolution (2024), Yi Xiao, Qiangqiang Yuan, Kui Jiang, et al., IEEE Transactions on Multimedia; citations=282, DOI/ID=10.1109/tmm.2024.3521798.

### 事件触发、通信受限与分布式实现

该类包含 5 条候选文献，覆盖 2025-2026。排名靠前的代表工作是《MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation》，可作为进入该分支的入口。

- MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation (2026), Fuqiang Gu, Yuanke Li, Xianlei Long, et al., Proceedings of the AAAI Conference on Artificial Intelligence; citations=3, DOI/ID=10.1609/aaai.v40i6.42427.
- DAP-Mamba: Dynamic Anchor Prompting with Bi-Phasic Mamba for Referring Video Object Segmentation (2026), Wei He, Jiaqing Fan, Zhengtong Zhu, et al., crossref; citations=0, DOI/ID=10.2139/ssrn.7549571.
- AS-Mamba: Multimodal Semantic Segmentation via Mamba with Asymmetric Cross and Spatial Perception (2025), Guohao Shen, Hongzhao Li, Hao Yang, et al., 2025 IEEE 31th International Conference on Parallel and Distributed Systems (ICPADS); citations=0, DOI/ID=10.1109/icpads67057.2025.11323107.
- SAM-FireAdapter: An adapter for fire segmentation with SAM (2026), Yanan Wu, Chaoqun Hong, Yongfeng Chen, et al., Journal of Visual Communication and Image Representation; citations=4, DOI/ID=10.1016/j.jvcir.2025.104678.
- Visual Boosting Techniques for Spatiotemporal Dense Pixel Visualizations (2026), Julius Rauscher, Frederik L. Dennig, Udo Schlegel, et al., arXiv; citations=0, DOI/ID=10.2312/eurova.20261004.

### 鲁棒性、攻击与不确定系统

该类包含 11 条候选文献，覆盖 2025-2026。排名靠前的代表工作是《Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation》，可作为进入该分支的入口。

- Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation (2025), Xin Zhang, Robby T. Tan, IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=26, DOI/ID=10.1109/cvpr52734.2025.01354.
- Mamba-Sea: A Mamba-Based Framework With Global-to-Local Sequence Augmentation for Generalizable Medical Image Segmentation (2025), Zihan Cheng, Jintao Guo, Jian Zhang, et al., IEEE Transactions on Medical Imaging; citations=44, DOI/ID=10.1109/tmi.2025.3564765.
- MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation (2026), Fuqiang Gu, Yuanke Li, Xianlei Long, et al., Proceedings of the AAAI Conference on Artificial Intelligence; citations=3, DOI/ID=10.1609/aaai.v40i6.42427.
- FMTNet: A Fourier‐Mamba–Transformer Enhanced Network for Medical Image Segmentation (2026), Shaoqiang Wang, Guiling Shi, Yuanyuan Zhang, et al., CAAI Transactions on Intelligence Technology; citations=2, DOI/ID=10.1049/cit2.70141.
- Automated Pneumothorax Segmentation with a Spatial Prior Contrast Adapter (2025), Yiming Jia, Essam A. Rashed, Applied Sciences; citations=2, DOI/ID=10.3390/app15126598.

### 优化、MPC 与性能约束

该类包含 45 条候选文献，覆盖 2024-2026。排名靠前的代表工作是《MambaOut: Do We Really Need Mamba for Vision?》，可作为进入该分支的入口。

- MambaOut: Do We Really Need Mamba for Vision? (2025), Weihao Yu, Xinchao Wang, IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=376, DOI/ID=10.1109/cvpr52734.2025.00423.
- PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition (2024), Chenhongyi Yang, Zehui Chen, Miguel Espinosa, et al., arXiv (Cornell University); citations=183, DOI/ID=10.48550/arxiv.2403.17695.
- Mamba-Adaptor: State Space Model Adaptor for Visual Recognition (2025), Fei Xie, Jiahao Nie, Yong Tan, et al., IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=8, DOI/ID=10.1109/cvpr52734.2025.01874.
- A Comprehensive Survey of Mamba Architectures for Medical Image Analysis: Classification, Segmentation, Restoration and Beyond (2024), Shubhi Bansal, Sreeharish Arikrishnan, J. Madhava Prasath, et al., arXiv (Cornell University); citations=48, DOI/ID=10.48550/arxiv.2410.02362.
- SAM-Mamba: Mamba Guided SAM Architecture for Generalized Zero-Shot Polyp Segmentation (2025), Tapas Kumar Dutta, Snehashis Majhi, Deepak Ranjan Nayak, et al., IEEE Winter Conference on Applications of Computer Vision; citations=19, DOI/ID=10.1109/wacv61041.2025.00457.

### 综述、基础理论与背景材料

该类包含 11 条候选文献，覆盖 2024-2026。排名靠前的代表工作是《A Survey on Visual Mamba》，可作为进入该分支的入口。

- A Survey on Visual Mamba (2024), Hanwei Zhang, Ying Zhu, Dan Wang, et al., Applied Sciences; citations=265, DOI/ID=10.3390/app14135683.
- A Comprehensive Survey of Mamba Architectures for Medical Image Analysis: Classification, Segmentation, Restoration and Beyond (2024), Shubhi Bansal, Sreeharish Arikrishnan, J. Madhava Prasath, et al., arXiv (Cornell University); citations=48, DOI/ID=10.48550/arxiv.2410.02362.
- Vision Mamba: A Comprehensive Survey and Taxonomy (2025), Xiao Liu, Chenxu Zhang, Fuxiang Huang, et al., IEEE Transactions on Neural Networks and Learning Systems; citations=262, DOI/ID=10.1109/tnnls.2025.3610435.
- Visual Mamba: A Survey and New Outlooks (2024), Rui Xu, Shu Yang, Yihui Wang, et al., arXiv (Cornell University); citations=157, DOI/ID=10.48550/arxiv.2404.18861.
- A Survey of Mamba (2026), Haohao Qu, Liangbo Ning, Rui An, et al., ACM Transactions on Intelligent Systems and Technology; citations=127, DOI/ID=10.1145/3819582.

### 其他相关工作

该类包含 20 条候选文献，覆盖 2024-2026。排名靠前的代表工作是《Semi-Mamba-UNet: Pixel-level contrastive and cross-supervised visual Mamba-based UNet for semi-supervised medical image segmentation》，可作为进入该分支的入口。

- Semi-Mamba-UNet: Pixel-level contrastive and cross-supervised visual Mamba-based UNet for semi-supervised medical image segmentation (2024), Chao Ma, Ziyang Wang, Knowledge-Based Systems; citations=127, DOI/ID=10.1016/j.knosys.2024.112203.
- LKM-UNet: Large Kernel Vision Mamba UNet for Medical Image Segmentation (2024), Jinhong Wang, Jintai Chen, Danny Z. Chen, et al., Lecture notes in computer science; citations=161, DOI/ID=10.1007/978-3-031-72111-3_34.
- FSE-Mamba: A novel Frequency-Spatial Entanglement Mamba model for retinal vessel segmentation (2025), Xutao Sun, Junwen Liu, Xiaolu Xu, et al., Computers in Biology and Medicine; citations=6, DOI/ID=10.1016/j.compbiomed.2025.110776.
- HLSM-UNet:Hybrid Local Spatial Mamba UNet for Medical Image Segmentation (2024), Xude Zhang, Xiaoping Wu, Changzhen Zhang, et al., 2024 International Conference on Neuromorphic Computing (ICNC); citations=2, DOI/ID=10.1109/icnc64304.2024.10987734.
- Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding (2025), Guo Chen, Yifei Huang, Jilan Xu, et al., International Journal of Computer Vision; citations=176, DOI/ID=10.1007/s11263-025-02597-y.

## 代表性论文表

| 角色 | 论文 | 年份 | 来源/期刊会议 | 为什么重要 |
| --- | --- | --- | --- | --- |
| 近期进展 | Spatial-Mamba: Effective Visual State Space Models via Structure-aware State Fusion | 2024 | arXiv (Cornell University) | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 高被引基础 | MambaOut: Do We Really Need Mamba for Vision? | 2025 | IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings | 引用量较高，适合作为理论背景或问题定义入口。 |
| 近期进展 | Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation | 2025 | IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 综述入口 | A Survey on Visual Mamba | 2024 | Applied Sciences | 覆盖面广，适合快速建立术语和分支地图。 |
| 安全约束 | Sigma-Fusion: Synergistic Semantic Segmentation of Optical and SAR Remote Sensing Images Based on a Siamese Mamba Architecture | 2026 | Lecture notes in computer science | 把连通保持与碰撞/障碍物约束放在同一控制问题中。 |
| 高被引基础 | PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition | 2024 | arXiv (Cornell University) | 引用量较高，适合作为理论背景或问题定义入口。 |
| 近期进展 | Mamba-Adaptor: State Space Model Adaptor for Visual Recognition | 2025 | IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 高被引基础 | Semi-Mamba-UNet: Pixel-level contrastive and cross-supervised visual Mamba-based UNet for semi-supervised medical image segmentation | 2024 | Knowledge-Based Systems | 引用量较高，适合作为理论背景或问题定义入口。 |
| 综述入口 | A Comprehensive Survey of Mamba Architectures for Medical Image Analysis: Classification, Segmentation, Restoration and Beyond | 2024 | arXiv (Cornell University) | 覆盖面广，适合快速建立术语和分支地图。 |
| 综述入口 | Vision Mamba: A Comprehensive Survey and Taxonomy | 2025 | IEEE Transactions on Neural Networks and Learning Systems | 覆盖面广，适合快速建立术语和分支地图。 |

## 研究脉络

- 2023-2026：约 86 条；代表工作为《MambaOut: Do We Really Need Mamba for Vision?》，该阶段重点可从 高被引基础 切入。

## 主要方法对比

| 方法族 | 典型关注点 | 优势 | 常见限制 |
| --- | --- | --- | --- |
| Potential/barrier/navigation functions | 保持已有边、避免距离约束破坏 | 直观，适合编队与避障 | 参数调节和局部极值问题明显 |
| Algebraic connectivity / Fiedler value | 直接优化或估计图连通度 | 能刻画全局连通裕度 | 分布式估计和可扩展性是难点 |
| Event-triggered connectivity control | 减少通信和控制更新 | 适合资源受限网络 | 需要证明无 Zeno 行为和鲁棒性 |
| Prescribed performance / constrained control | 把连通、碰撞、误差约束统一处理 | 约束表达清晰 | 对模型假设和扰动界较敏感 |
| MPC / optimization-based methods | 在预测窗口内优化连通和任务性能 | 可处理多目标权衡 | 计算量和实时性压力较高 |

## 重要论文阅读顺序

1. MambaOut: Do We Really Need Mamba for Vision? (2025), Weihao Yu, Xinchao Wang, IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=376, DOI/ID=10.1109/cvpr52734.2025.00423.
2. Frequency-Assisted Mamba for Remote Sensing Image Super-Resolution (2024), Yi Xiao, Qiangqiang Yuan, Kui Jiang, et al., IEEE Transactions on Multimedia; citations=282, DOI/ID=10.1109/tmm.2024.3521798.
3. A Survey on Visual Mamba (2024), Hanwei Zhang, Ying Zhu, Dan Wang, et al., Applied Sciences; citations=265, DOI/ID=10.3390/app14135683.
4. Vision Mamba: A Comprehensive Survey and Taxonomy (2025), Xiao Liu, Chenxu Zhang, Fuxiang Huang, et al., IEEE Transactions on Neural Networks and Learning Systems; citations=262, DOI/ID=10.1109/tnnls.2025.3610435.
5. Demystify Mamba in Vision: A Linear Attention Perspective (2024), Dongchen Han, Ziyi Wang, Zhuofan Xia, et al., neural information processing systems; citations=187, DOI/ID=10.52202/079017-4039.
6. PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition (2024), Chenhongyi Yang, Zehui Chen, Miguel Espinosa, et al., arXiv (Cornell University); citations=183, DOI/ID=10.48550/arxiv.2403.17695.
7. Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding (2025), Guo Chen, Yifei Huang, Jilan Xu, et al., International Journal of Computer Vision; citations=176, DOI/ID=10.1007/s11263-025-02597-y.
8. LKM-UNet: Large Kernel Vision Mamba UNet for Medical Image Segmentation (2024), Jinhong Wang, Jintai Chen, Danny Z. Chen, et al., Lecture notes in computer science; citations=161, DOI/ID=10.1007/978-3-031-72111-3_34.
9. Spatial-Mamba: Effective Visual State Space Models via Structure-aware State Fusion (2024), Chaodong Xiao, Minghan Li, Zhengqiang Zhang, et al., arXiv (Cornell University); citations=45, DOI/ID=10.48550/arxiv.2410.15091.
10. Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation (2025), Xin Zhang, Robby T. Tan, IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=26, DOI/ID=10.1109/cvpr52734.2025.01354.

## 局限与开放问题

- 从理论保证走向真实机器人/无人系统实验仍是关键缺口，尤其是有限感知、通信丢包和动态障碍同时存在时。
- 代数连通度类方法需要更可扩展的分布式估计与控制实现，避免全局谱信息成为隐含假设。
- 连通保持、碰撞避免、队形精度和能耗之间存在多目标冲突，需要更清晰的权衡机制。
- 事件触发和通信受限控制需要同时说明无 Zeno、鲁棒性和实际网络延迟下的性能。
- 面向攻击或故障的连通度控制还需要把安全检测、拓扑重构和控制律设计更紧密地结合。

## 网页补充建议

- `"Mamba visual segmentation spatial adapter" PDF`
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
