# 主题调研简报：skin lesion segmentation transformer mamba

## 一句话结论

`skin lesion segmentation transformer mamba` 的检索结果显示，该主题的主线不是单纯的多智能体一致性，而是在编队、避障、通信受限、攻击或不确定性下维持通信/感知图的连通裕度。当前语料包含 102 条去重记录，年份覆盖 2020-2026，主要落在连通度保持与代数连通度控制、编队、队形与一致性控制、避障、安全约束与有限感知、事件触发、通信受限与分布式实现、鲁棒性、攻击与不确定系统等方向。

## 背景与问题定义

本简报基于可复现学术 API 检索生成，用于快速理解一个研究主题的核心问题、代表工作和后续阅读路径。在多智能体系统语境下，连通度控制通常关注通信图或感知图在运动、编队、避障、攻击、事件触发通信等条件下能否保持足够连通，常见度量包括图连通性、Laplacian 的 algebraic connectivity、邻接边保持、有限感知半径内的连通维护等。

## 检索策略与覆盖范围

- 原始查询：`skin lesion segmentation transformer mamba`
- 年份范围：2020 到 2026
- 去重后文献数：102
- 各源返回数量：openalex=45, crossref=45, semanticscholar=0, arxiv=32
- 检索错误：semanticscholar: HTTPError: HTTP Error 429: 
- 说明：该结果依赖 API 覆盖、查询式质量和元数据完整度；最终用于论文写作前仍需人工筛选全文。

## 技术路线分类

### 连通度保持与代数连通度控制

该类包含 47 条候选文献，覆盖 2021-2026。排名靠前的代表工作是《SkinMamba: A Precision Skin Lesion Segmentation Architecture with Cross-Scale Global State Modeling and Frequency Boundary Guidance》，可作为进入该分支的入口。

- SkinMamba: A Precision Skin Lesion Segmentation Architecture with Cross-Scale Global State Modeling and Frequency Boundary Guidance (2024), Shun Zou, Zhang, Mingya, Fan, Bingjian, et al., arXiv (Cornell University); citations=22, DOI/ID=10.48550/arxiv.2409.10890.
- SGNet: A Structure-Guided Network with Dual-Domain Boundary Enhancement and Semantic Fusion for Skin Lesion Segmentation (2025), Haijiao Yun, Qingyu Du, Ziqing Han, et al., Sensors; citations=7, DOI/ID=10.3390/s25154652.
- AttmNet: a hybrid Transformer integrating self-attention, Mamba, and multi-layer convolution for enhanced lesion segmentation (2025), Hancan Zhu, Yibing Huang, Kelin Yao, et al., Quantitative Imaging in Medicine and Surgery; citations=6, DOI/ID=10.21037/qims-2024-2561.
- VML-UNet: Fusing Vision Mamba and Lightweight Attention Mechanism for Skin Lesion Segmentation (2025), Tang Tang, Haihui Wang, Qiang Rao, et al., Electronics; citations=2, DOI/ID=10.3390/electronics14142866.
- HBTMNet: A Tri-Hybrid Boundary-Aware Transformer Mamba Network for Precise Skin Lesion Segmentation (2026), Sanjyoti Kumari Tarai, Renuka Arora, International Journal of Drug Delivery Technology; citations=0, DOI/ID=10.25258/ijddt.16.74s.27.

### 编队、队形与一致性控制

该类包含 30 条候选文献，覆盖 2021-2026。排名靠前的代表工作是《SkinMamba: A Precision Skin Lesion Segmentation Architecture with Cross-Scale Global State Modeling and Frequency Boundary Guidance》，可作为进入该分支的入口。

- SkinMamba: A Precision Skin Lesion Segmentation Architecture with Cross-Scale Global State Modeling and Frequency Boundary Guidance (2024), Shun Zou, Zhang, Mingya, Fan, Bingjian, et al., arXiv (Cornell University); citations=22, DOI/ID=10.48550/arxiv.2409.10890.
- VM-UNet: Vision Mamba UNet for Medical Image Segmentation (2025), Jiacheng Ruan, Jincheng Li, Suncheng Xiang, ACM Transactions on Multimedia Computing Communications and Applications; citations=1308, DOI/ID=10.1145/3767748.
- VML-UNet: Fusing Vision Mamba and Lightweight Attention Mechanism for Skin Lesion Segmentation (2025), Tang Tang, Haihui Wang, Qiang Rao, et al., Electronics; citations=2, DOI/ID=10.3390/electronics14142866.
- HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation (2024), Jiashu Xu, arXiv (Cornell University); citations=72, DOI/ID=10.48550/arxiv.2405.05007.
- A Deep Feature Fusion Visual Mamba for Skin Lesion Segmentation (2026), Sisi Yan, Hao Zheng, Yaru Fu, et al., International Conference on Image, Signal Processing and Pattern Recognition (ISPP); citations=0, DOI/ID=10.1109/ispp69262.2026.11542987.

### 避障、安全约束与有限感知

该类包含 1 条候选文献，覆盖 2024-2024。排名靠前的代表工作是《Lesion Boundary Detection for Skin Lesion Segmentation Based on Boundary Sensing and Cnn-Transformer Fusion Networks》，可作为进入该分支的入口。

- Lesion Boundary Detection for Skin Lesion Segmentation Based on Boundary Sensing and Cnn-Transformer Fusion Networks (2024), XuZhen Huang, Yuliang Ma, Xiajin Mei, et al., crossref; citations=0, DOI/ID=10.2139/ssrn.4970410.

### 事件触发、通信受限与分布式实现

该类包含 14 条候选文献，覆盖 2023-2026。排名靠前的代表工作是《SkinMamba: A Precision Skin Lesion Segmentation Architecture with Cross-Scale Global State Modeling and Frequency Boundary Guidance》，可作为进入该分支的入口。

- SkinMamba: A Precision Skin Lesion Segmentation Architecture with Cross-Scale Global State Modeling and Frequency Boundary Guidance (2024), Shun Zou, Zhang, Mingya, Fan, Bingjian, et al., arXiv (Cornell University); citations=22, DOI/ID=10.48550/arxiv.2409.10890.
- VM-UNet: Vision Mamba UNet for Medical Image Segmentation (2025), Jiacheng Ruan, Jincheng Li, Suncheng Xiang, ACM Transactions on Multimedia Computing Communications and Applications; citations=1308, DOI/ID=10.1145/3767748.
- SUTrans-NET: a hybrid transformer approach to skin lesion segmentation (2024), Yaqin Li, Tonghe Tian, Jing Hu, et al., PeerJ Computer Science; citations=25, DOI/ID=10.7717/peerj-cs.1935.
- U-SwinTrans: Automated Skin Lesion Segmentation Using Swin Transformer (2025), Aaditya Lochan Sharma, Kalpana Sharma, Utkarsh Prakash Srivastava, et al., 2025 3rd International Conference on Intelligent Systems, Advanced Computing and Communication (ISACC); citations=9, DOI/ID=10.1109/isacc65211.2025.10969397.
- FITrans: Skin Lesion Segmentation Based on Feature Integration and Transformer (2023), Likun Zhang, 2023 3rd International Conference on Neural Networks, Information and Communication Engineering (NNICE); citations=2, DOI/ID=10.1109/nnice58320.2023.10105777.

### 鲁棒性、攻击与不确定系统

该类包含 14 条候选文献，覆盖 2021-2026。排名靠前的代表工作是《SGNet: A Structure-Guided Network with Dual-Domain Boundary Enhancement and Semantic Fusion for Skin Lesion Segmentation》，可作为进入该分支的入口。

- SGNet: A Structure-Guided Network with Dual-Domain Boundary Enhancement and Semantic Fusion for Skin Lesion Segmentation (2025), Haijiao Yun, Qingyu Du, Ziqing Han, et al., Sensors; citations=7, DOI/ID=10.3390/s25154652.
- HBTMNet: A Tri-Hybrid Boundary-Aware Transformer Mamba Network for Precise Skin Lesion Segmentation (2026), Sanjyoti Kumari Tarai, Renuka Arora, International Journal of Drug Delivery Technology; citations=0, DOI/ID=10.25258/ijddt.16.74s.27.
- Skin-lesion segmentation using boundary-aware segmentation network and classification based on a mixture of convolutional and transformer neural networks (2025), Javaria Amin, Marium Azhar, Habiba Arshad, et al., Frontiers in Medicine; citations=25, DOI/ID=10.3389/fmed.2025.1524146.
- KM-UNet: medical image segmentation with selective-scan Mamba and Kolmogorov-Arnold networks (2026), Yibo Zhang, Jingwen Zhao, Xiang Liu, et al., PeerJ Computer Science; citations=190, DOI/ID=10.7717/peerj-cs.3727.
- UltraLBM-UNet: Ultralight bidirectional mamba-based model for skin lesion segmentation (2026), 范琳煊, Juntao Jiang, Weixuan Liu, et al., Neurocomputing; citations=4, DOI/ID=10.1016/j.neucom.2026.134608.

### 优化、MPC 与性能约束

该类包含 38 条候选文献，覆盖 2021-2026。排名靠前的代表工作是《UltraLight VM-UNet: Parallel Vision Mamba significantly reduces parameters for skin lesion segmentation》，可作为进入该分支的入口。

- UltraLight VM-UNet: Parallel Vision Mamba significantly reduces parameters for skin lesion segmentation (2025), Renkai Wu, Yinghao Liu, Guochen Ning, et al., Patterns; citations=293, DOI/ID=10.1016/j.patter.2025.101298.
- SGNet: A Structure-Guided Network with Dual-Domain Boundary Enhancement and Semantic Fusion for Skin Lesion Segmentation (2025), Haijiao Yun, Qingyu Du, Ziqing Han, et al., Sensors; citations=7, DOI/ID=10.3390/s25154652.
- AttmNet: a hybrid Transformer integrating self-attention, Mamba, and multi-layer convolution for enhanced lesion segmentation (2025), Hancan Zhu, Yibing Huang, Kelin Yao, et al., Quantitative Imaging in Medicine and Surgery; citations=6, DOI/ID=10.21037/qims-2024-2561.
- VML-UNet: Fusing Vision Mamba and Lightweight Attention Mechanism for Skin Lesion Segmentation (2025), Tang Tang, Haihui Wang, Qiang Rao, et al., Electronics; citations=2, DOI/ID=10.3390/electronics14142866.
- HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation (2024), Jiashu Xu, arXiv (Cornell University); citations=72, DOI/ID=10.48550/arxiv.2405.05007.

### 综述、基础理论与背景材料

该类包含 5 条候选文献，覆盖 2021-2026。排名靠前的代表工作是《MaxCAT-Net: Hybrid transformer-based skin lesion classification and lightweight segmentation with calibration and cross-dataset validation》，可作为进入该分支的入口。

- MaxCAT-Net: Hybrid transformer-based skin lesion classification and lightweight segmentation with calibration and cross-dataset validation (2026), Sadek Al Prince, Israt Fatema Shorna, crossref; citations=0, DOI/ID=10.64898/2026.11.01.763389.
- Skin Lesion Segmentation using Vision Transformer and UNet (2025), Bibat Thokar, International Journal on Engineering Technology and Infrastructure Development; citations=0, DOI/ID=10.3126/injet-indev.v2i1.82195.
- Research Review of Deep Learning in Skin Lesions Image Segmentation (2026), MENG Xiangfu, LI Jiaxun, YU Chunlin, LU Yunxuan, DOAJ (DOAJ: Directory of Open Access Journals); citations=0, DOI/ID=10.3778/j.issn.1673-9418.2504019.
- A Survey on Deep Learning for Skin Lesion Segmentation (2022), Zahra Mirikharaji, Kumar Abhishek, Alceu Bissoto, et al., arXiv; citations=0, DOI/ID=10.1016/j.media.2023.102863.
- Team NeuroPoly: Description of the Pipelines for the MICCAI 2021 MS New Lesions Segmentation Challenge (2021), Uzay Macar, Enamundram Naga Karthik, Charley Gros, et al., arXiv; citations=0, DOI/ID=2109.05409v2.

### 其他相关工作

该类包含 23 条候选文献，覆盖 2020-2026。排名靠前的代表工作是《Seg-MambaDiff: Skin Lesion Segmentation Based on Mamba and Diffusion Model》，可作为进入该分支的入口。

- Seg-MambaDiff: Skin Lesion Segmentation Based on Mamba and Diffusion Model (2025), Jiacheng Zheng, Xinghong Chen, Weilin Wu, et al., International Conference on Computer Graphics, Image and Virtualization (ICCGIV); citations=0, DOI/ID=10.1109/iccgiv65419.2025.11085387.
- Semi-Supervised Skin Lesion Segmentation With Coupling CNN and Transformer Features (2022), Mohammad D. Alahmadi, Wajdi Alghamdi, IEEE Access; citations=42, DOI/ID=10.1109/access.2022.3224005.
- BEFNet: A Hybrid CNN-Mamba Architecture for Accurate Skin Lesion Image Segmentation (2024), Wei Wang, Yuewen Luo, Xin Wang, 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM); citations=5, DOI/ID=10.1109/bibm62325.2024.10822106.
- Transformer and CNN-Based Deep Learning Models for Skin Lesion Segmentation (2025), Abdelkader Alrabai, Amira Echtioui, Fathi Kallel, 2025 IEEE International Conference on Advanced Systems and Emergent Technologies (IC_ASET); citations=1, DOI/ID=10.1109/ic_aset65966.2025.11232106.
- S³-Mamba: Small-Size-Sensitive Mamba for Lesion Segmentation (2025), Gui Wang, Yuexiang Li, Wenting Chen, et al., Proceedings of the AAAI Conference on Artificial Intelligence; citations=21, DOI/ID=10.1609/aaai.v39i7.32824.

## 代表性论文表

| 角色 | 论文 | 年份 | 来源/期刊会议 | 为什么重要 |
| --- | --- | --- | --- | --- |
| 高被引基础 | UltraLight VM-UNet: Parallel Vision Mamba significantly reduces parameters for skin lesion segmentation | 2025 | Patterns | 引用量较高，适合作为理论背景或问题定义入口。 |
| 近期进展 | SkinMamba: A Precision Skin Lesion Segmentation Architecture with Cross-Scale Global State Modeling and Frequency Boundary Guidance | 2024 | arXiv (Cornell University) | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 近期进展 | SGNet: A Structure-Guided Network with Dual-Domain Boundary Enhancement and Semantic Fusion for Skin Lesion Segmentation | 2025 | Sensors | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 近期进展 | AttmNet: a hybrid Transformer integrating self-attention, Mamba, and multi-layer convolution for enhanced lesion segmentation | 2025 | Quantitative Imaging in Medicine and Surgery | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 高被引基础 | VM-UNet: Vision Mamba UNet for Medical Image Segmentation | 2025 | ACM Transactions on Multimedia Computing Communications and Applications | 引用量较高，适合作为理论背景或问题定义入口。 |
| 近期进展 | VML-UNet: Fusing Vision Mamba and Lightweight Attention Mechanism for Skin Lesion Segmentation | 2025 | Electronics | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 近期进展 | HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation | 2024 | arXiv (Cornell University) | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 近期进展 | HBTMNet: A Tri-Hybrid Boundary-Aware Transformer Mamba Network for Precise Skin Lesion Segmentation | 2026 | International Journal of Drug Delivery Technology | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 近期进展 | A Deep Feature Fusion Visual Mamba for Skin Lesion Segmentation | 2026 | International Conference on Image, Signal Processing and Pattern Recognition (ISPP) | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 近期进展 | An Adaptive Multi-Scale Fusion Model Based on Mamba for Skin Lesion Image Segmentation | 2026 | International Conference on Computer Supported Cooperative Work in Design (CSCWD) | 反映近年约束控制、鲁棒性或应用场景的新变化。 |

## 研究脉络

- 2019-2022：约 10 条；代表工作为《Semi-Supervised Skin Lesion Segmentation With Coupling CNN and Transformer Features》，该阶段重点可从 代表工作 切入。
- 2023-2026：约 92 条；代表工作为《VM-UNet: Vision Mamba UNet for Medical Image Segmentation》，该阶段重点可从 高被引基础 切入。

## 主要方法对比

| 方法族 | 典型关注点 | 优势 | 常见限制 |
| --- | --- | --- | --- |
| Potential/barrier/navigation functions | 保持已有边、避免距离约束破坏 | 直观，适合编队与避障 | 参数调节和局部极值问题明显 |
| Algebraic connectivity / Fiedler value | 直接优化或估计图连通度 | 能刻画全局连通裕度 | 分布式估计和可扩展性是难点 |
| Event-triggered connectivity control | 减少通信和控制更新 | 适合资源受限网络 | 需要证明无 Zeno 行为和鲁棒性 |
| Prescribed performance / constrained control | 把连通、碰撞、误差约束统一处理 | 约束表达清晰 | 对模型假设和扰动界较敏感 |
| MPC / optimization-based methods | 在预测窗口内优化连通和任务性能 | 可处理多目标权衡 | 计算量和实时性压力较高 |

## 重要论文阅读顺序

1. VM-UNet: Vision Mamba UNet for Medical Image Segmentation (2025), Jiacheng Ruan, Jincheng Li, Suncheng Xiang, ACM Transactions on Multimedia Computing Communications and Applications; citations=1308, DOI/ID=10.1145/3767748.
2. UltraLight VM-UNet: Parallel Vision Mamba significantly reduces parameters for skin lesion segmentation (2025), Renkai Wu, Yinghao Liu, Guochen Ning, et al., Patterns; citations=293, DOI/ID=10.1016/j.patter.2025.101298.
3. H-vmunet: High-order Vision Mamba UNet for medical image segmentation (2025), Renkai Wu, Yinghao Liu, Pengchen Liang, et al., Neurocomputing; citations=276, DOI/ID=10.1016/j.neucom.2025.129447.
4. LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation (2024), Weibin Liao, Yinghao Zhu, Xinyuan Wang, et al., arXiv (Cornell University); citations=233, DOI/ID=10.48550/arxiv.2403.05246.
5. VM-UNET-V2: Rethinking Vision Mamba UNet for Medical Image Segmentation (2024), Mingya Zhang, Yue Yu, Jin Sun, et al., Lecture notes in computer science; citations=213, DOI/ID=10.1007/978-981-97-5128-0_27.
6. KM-UNet: medical image segmentation with selective-scan Mamba and Kolmogorov-Arnold networks (2026), Yibo Zhang, Jingwen Zhao, Xiang Liu, et al., PeerJ Computer Science; citations=190, DOI/ID=10.7717/peerj-cs.3727.
7. HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation (2024), Jiashu Xu, arXiv (Cornell University); citations=72, DOI/ID=10.48550/arxiv.2405.05007.
8. Transformer-based multi-attention hybrid networks for skin lesion segmentation (2024), Zhiwei Dong, Jinjiang Li, Zhen Hua, Expert Systems with Applications; citations=45, DOI/ID=10.1016/j.eswa.2023.123016.
9. SkinMamba: A Precision Skin Lesion Segmentation Architecture with Cross-Scale Global State Modeling and Frequency Boundary Guidance (2024), Shun Zou, Zhang, Mingya, Fan, Bingjian, et al., arXiv (Cornell University); citations=22, DOI/ID=10.48550/arxiv.2409.10890.
10. SGNet: A Structure-Guided Network with Dual-Domain Boundary Enhancement and Semantic Fusion for Skin Lesion Segmentation (2025), Haijiao Yun, Qingyu Du, Ziqing Han, et al., Sensors; citations=7, DOI/ID=10.3390/s25154652.

## 局限与开放问题

- 从理论保证走向真实机器人/无人系统实验仍是关键缺口，尤其是有限感知、通信丢包和动态障碍同时存在时。
- 代数连通度类方法需要更可扩展的分布式估计与控制实现，避免全局谱信息成为隐含假设。
- 连通保持、碰撞避免、队形精度和能耗之间存在多目标冲突，需要更清晰的权衡机制。
- 事件触发和通信受限控制需要同时说明无 Zeno、鲁棒性和实际网络延迟下的性能。
- 面向攻击或故障的连通度控制还需要把安全检测、拓扑重构和控制律设计更紧密地结合。

## 网页补充建议

- `"skin lesion segmentation transformer mamba" PDF`
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
