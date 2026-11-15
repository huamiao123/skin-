# 主题调研简报：medical image segmentation domain generalization

## 一句话结论

`medical image segmentation domain generalization` 的检索结果显示，该主题的主线不是单纯的多智能体一致性，而是在编队、避障、通信受限、攻击或不确定性下维持通信/感知图的连通裕度。当前语料包含 101 条去重记录，年份覆盖 2020-2026，主要落在连通度保持与代数连通度控制、编队、队形与一致性控制、避障、安全约束与有限感知、事件触发、通信受限与分布式实现、鲁棒性、攻击与不确定系统等方向。

## 背景与问题定义

本简报基于可复现学术 API 检索生成，用于快速理解一个研究主题的核心问题、代表工作和后续阅读路径。在多智能体系统语境下，连通度控制通常关注通信图或感知图在运动、编队、避障、攻击、事件触发通信等条件下能否保持足够连通，常见度量包括图连通性、Laplacian 的 algebraic connectivity、邻接边保持、有限感知半径内的连通维护等。

## 检索策略与覆盖范围

- 原始查询：`medical image segmentation domain generalization`
- 年份范围：2020 到 2026
- 去重后文献数：101
- 各源返回数量：openalex=45, crossref=45, semanticscholar=0, arxiv=36
- 检索错误：semanticscholar: HTTPError: HTTP Error 429: 
- 说明：该结果依赖 API 覆盖、查询式质量和元数据完整度；最终用于论文写作前仍需人工筛选全文。

## 技术路线分类

### 连通度保持与代数连通度控制

该类包含 36 条候选文献，覆盖 2020-2026。排名靠前的代表工作是《Causality-Inspired Single-Source Domain Generalization for Medical Image Segmentation》，可作为进入该分支的入口。

- Causality-Inspired Single-Source Domain Generalization for Medical Image Segmentation (2022), Cheng Ouyang, Chen Chen, Surui Li, et al., IEEE Transactions on Medical Imaging; citations=291, DOI/ID=10.1109/tmi.2022.3224067.
- Dynamic domain generalization for medical image segmentation (2024), Zhiming Cheng, Mingxia Liu, Chenggang Clarence Yan, et al., Neural Networks; citations=35, DOI/ID=10.1016/j.neunet.2024.107073.
- Treasure in Distribution: A Domain Randomization Based Multi-source Domain Generalization for 2D Medical Image Segmentation (2023), Ziyang Chen, Yongsheng Pan, Yiwen Ye, et al., Lecture notes in computer science; citations=45, DOI/ID=10.1007/978-3-031-43901-8_9.
- Hcdg: A Hierarchical Consistency Framework for Domain Generalization on Medical Image Segmentation (2022), Yijun Yang, Shujun Wang, Lei Zhu, et al., SSRN Electronic Journal; citations=19, DOI/ID=10.2139/ssrn.4188684.
- Domain generalization of deep networks for medical image segmentation via meta learning (2022), Quande Liu, Qi Dou, Cheng Chen, et al., Elsevier eBooks; citations=14, DOI/ID=10.1016/b978-0-32-399851-2.00016-8.

### 编队、队形与一致性控制

该类包含 26 条候选文献，覆盖 2020-2026。排名靠前的代表工作是《FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space》，可作为进入该分支的入口。

- FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space (2021), Quande Liu, Cheng Chen, Jing Qin, et al., IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=615, DOI/ID=10.1109/cvpr46437.2021.00107.
- Causality-Inspired Single-Source Domain Generalization for Medical Image Segmentation (2022), Cheng Ouyang, Chen Chen, Surui Li, et al., IEEE Transactions on Medical Imaging; citations=291, DOI/ID=10.1109/tmi.2022.3224067.
- Rethinking Data Augmentation for Single-Source Domain Generalization in Medical Image Segmentation (2023), Zixian Su, Kai Yao, Xi Yang, et al., Proceedings of the AAAI Conference on Artificial Intelligence; citations=128, DOI/ID=10.1609/aaai.v37i2.25332.
- Single-Domain Generalization in Medical Image Segmentation via Test-Time Adaptation from Shape Dictionary (2022), Quande Liu, Cheng Chen, Qi Dou, et al., Proceedings of the AAAI Conference on Artificial Intelligence; citations=51, DOI/ID=10.1609/aaai.v36i2.20068.
- Medical Image Segmentation via Single-Source Domain Generalization with Random Amplitude Spectrum Synthesis (2024), Qiang Qiao, Wenyu Wang, Meixia Qu, et al., Lecture notes in computer science; citations=15, DOI/ID=10.1007/978-3-031-72114-4_42.

### 避障、安全约束与有限感知

该类包含 10 条候选文献，覆盖 2021-2025。排名靠前的代表工作是《FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space》，可作为进入该分支的入口。

- FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space (2021), Quande Liu, Cheng Chen, Jing Qin, et al., IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=615, DOI/ID=10.1109/cvpr46437.2021.00107.
- Domain and Content Adaptive Convolution Based Multi-Source Domain Generalization for Medical Image Segmentation (2022), Shishuai Hu, Zehui Liao, Jianpeng Zhang, et al., IEEE Transactions on Medical Imaging; citations=121, DOI/ID=10.1109/tmi.2022.3210133.
- Frequency‐aware domain randomization for single‐source domain generalization in medical image segmentation (2025), Jiayi Wu, Zikai Chen, Yiyi Chen, et al., Medical Physics; citations=2, DOI/ID=10.1002/mp.70118.
- Integrated Domain Augmentation and Structural Content Reconstruction Based Multisource Domain Generalization on Medical Image Segmentation (2025), Yihan Ren, Yanfeng Li, Jia Sun, et al., IEEE Transactions on Instrumentation and Measurement; citations=2, DOI/ID=10.1109/tim.2025.3548070.
- A Comparative Domain Generalization Study for SAR Image-Based Flood Segmentation (2025), Ilter Turkmenli, Koray Kayabol, Erchan Aptoula, 2025 Joint Urban Remote Sensing Event (JURSE); citations=0, DOI/ID=10.1109/jurse60372.2025.11075977.

### 事件触发、通信受限与分布式实现

该类包含 6 条候选文献，覆盖 2021-2025。排名靠前的代表工作是《FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space》，可作为进入该分支的入口。

- FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space (2021), Quande Liu, Cheng Chen, Jing Qin, et al., IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=615, DOI/ID=10.1109/cvpr46437.2021.00107.
- Domain Generalization with Adversarial Intensity Attack for Medical Image Segmentation (2023), Zheyuan Zhang, Bin Wang, Lanhong Yao, et al., arXiv (Cornell University); citations=14, DOI/ID=10.48550/arxiv.2304.02720.
- Invariant Content Synergistic Learning for Domain Generalization on Medical Image Segmentation (2022), Yuxin Kang, Hansheng Li, Xuan Zhao, et al., IEEE International Conference on Bioinformatics and Biomedicine (BIBM); citations=9, DOI/ID=10.1109/bibm55620.2022.9995034.
- Fully Automated SAM for Single-source Domain Generalization in Medical Image Segmentation (2025), Huanli Zhuo, Leilei Ma, Haifeng Zhao, et al., Conference proceedings/Conference proceedings - IEEE International Conference on Systems, Man, and Cybernetics; citations=1, DOI/ID=10.1109/smc58881.2025.11343166.
- A Comparative Domain Generalization Study for SAR Image-Based Flood Segmentation (2025), Ilter Turkmenli, Koray Kayabol, Erchan Aptoula, 2025 Joint Urban Remote Sensing Event (JURSE); citations=0, DOI/ID=10.1109/jurse60372.2025.11075977.

### 鲁棒性、攻击与不确定系统

该类包含 30 条候选文献，覆盖 2020-2026。排名靠前的代表工作是《Causality-Inspired Single-Source Domain Generalization for Medical Image Segmentation》，可作为进入该分支的入口。

- Causality-Inspired Single-Source Domain Generalization for Medical Image Segmentation (2022), Cheng Ouyang, Chen Chen, Surui Li, et al., IEEE Transactions on Medical Imaging; citations=291, DOI/ID=10.1109/tmi.2022.3224067.
- Treasure in Distribution: A Domain Randomization Based Multi-source Domain Generalization for 2D Medical Image Segmentation (2023), Ziyang Chen, Yongsheng Pan, Yiwen Ye, et al., Lecture notes in computer science; citations=45, DOI/ID=10.1007/978-3-031-43901-8_9.
- Domain Generalization with Adversarial Intensity Attack for Medical Image Segmentation (2023), Zheyuan Zhang, Bin Wang, Lanhong Yao, et al., arXiv (Cornell University); citations=14, DOI/ID=10.48550/arxiv.2304.02720.
- Towards robust medical image segmentation: Spectro-spatial domain generalization with MRAM and DMIR (2026), Yuqi Li, Junhao Dong, Hansheng Zeng, et al., Computer Vision and Image Understanding; citations=4, DOI/ID=10.1016/j.cviu.2026.104822.
- Frequency‐aware domain randomization for single‐source domain generalization in medical image segmentation (2025), Jiayi Wu, Zikai Chen, Yiyi Chen, et al., Medical Physics; citations=2, DOI/ID=10.1002/mp.70118.

### 优化、MPC 与性能约束

该类包含 42 条候选文献，覆盖 2020-2026。排名靠前的代表工作是《FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space》，可作为进入该分支的入口。

- FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space (2021), Quande Liu, Cheng Chen, Jing Qin, et al., IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=615, DOI/ID=10.1109/cvpr46437.2021.00107.
- Causality-Inspired Single-Source Domain Generalization for Medical Image Segmentation (2022), Cheng Ouyang, Chen Chen, Surui Li, et al., IEEE Transactions on Medical Imaging; citations=291, DOI/ID=10.1109/tmi.2022.3224067.
- Medical Image Segmentation via Single-Source Domain Generalization with Random Amplitude Spectrum Synthesis (2024), Qiang Qiao, Wenyu Wang, Meixia Qu, et al., Lecture notes in computer science; citations=15, DOI/ID=10.1007/978-3-031-72114-4_42.
- Leveraging SAM for Single-Source Domain Generalization in Medical Image Segmentation (2024), Hanhui Wang, Huaize Ye, Yi Xia, et al., arXiv (Cornell University); citations=7, DOI/ID=10.48550/arxiv.2401.02076.
- CoSAM: Self-Correcting SAM for Domain Generalization in 2D Medical Image Segmentation (2024), Yihang Fu, Ziyang Chen, Yiwen Ye, et al., arXiv (Cornell University); citations=7, DOI/ID=10.48550/arxiv.2411.10136.

### 综述、基础理论与背景材料

该类包含 5 条候选文献，覆盖 2021-2023。排名靠前的代表工作是《Devil is in Channels: Contrastive Single Domain Generalization for Medical Image Segmentation》，可作为进入该分支的入口。

- Devil is in Channels: Contrastive Single Domain Generalization for Medical Image Segmentation (2023), Shishuai Hu, Zehui Liao, Yong Xia, Lecture notes in computer science; citations=77, DOI/ID=10.1007/978-3-031-43901-8_2.
- Domain Generalization in Medical Image Segmentation via Meta-Learners (2022), Hugo Oliveira, Roberto Marcondes Cesar Junior, Pedro H. T. Gama, et al., SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI); citations=9, DOI/ID=10.1109/sibgrapi55357.2022.9991767.
- DivergentNets: Medical Image Segmentation by Network Ensemble (2021), Vajira Thambawita, Steven A. Hicks, Pål Halvorsen, et al., arXiv; citations=0, DOI/ID=2107.00283v1.
- Attention Mechanisms in Medical Image Segmentation: A Survey (2023), Yutong Xie, Bing Yang, Qingbiao Guan, et al., arXiv; citations=0, DOI/ID=2305.17937v1.
- Data synthesis and adversarial networks: A review and meta-analysis in cancer imaging (2021), Richard Osuala, Kaisar Kushibar, Lidia Garrucho, et al., arXiv; citations=0, DOI/ID=10.1016/j.media.2022.102704.

### 其他相关工作

该类包含 28 条候选文献，覆盖 2020-2026。排名靠前的代表工作是《From Denoising Training to Test-Time Adaptation: Enhancing Domain Generalization for Medical Image Segmentation》，可作为进入该分支的入口。

- From Denoising Training to Test-Time Adaptation: Enhancing Domain Generalization for Medical Image Segmentation (2024), Ruxue Wen, Hangjie Yuan, Dong Ni, et al., IEEE Winter Conference on Applications of Computer Vision; citations=42, DOI/ID=10.1109/wacv57701.2024.00052.
- Generative feature style augmentation for domain generalization in medical image segmentation (2025), Yunzhi Huang, Luyi Han, Haoran Dou, Pattern Recognition; citations=33, DOI/ID=10.1016/j.patcog.2025.111416.
- Frequency-Mixed Single-Source Domain Generalization for Medical Image Segmentation (2023), Heng Li, Haojin Li, Wei Zhao, et al., Lecture notes in computer science; citations=52, DOI/ID=10.1007/978-3-031-43987-2_13.
- Adversarial Consistency for Single Domain Generalization in Medical Image Segmentation (2022), Yanwu Xu, Shaoan Xie, Maxwell Reynolds, et al., Lecture notes in computer science; citations=45, DOI/ID=10.1007/978-3-031-16449-1_64.
- Aegis: A domain generalization framework for medical image segmentation by mitigating feature misalignment (2025), Yuheng Xu, Taiping Zhang, Yuqi Fang, Pattern Recognition; citations=9, DOI/ID=10.1016/j.patcog.2025.112406.

## 代表性论文表

| 角色 | 论文 | 年份 | 来源/期刊会议 | 为什么重要 |
| --- | --- | --- | --- | --- |
| 高被引基础 | FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space | 2021 | IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings | 引用量较高，适合作为理论背景或问题定义入口。 |
| 高被引基础 | Causality-Inspired Single-Source Domain Generalization for Medical Image Segmentation | 2022 | IEEE Transactions on Medical Imaging | 引用量较高，适合作为理论背景或问题定义入口。 |
| 高被引基础 | Rethinking Data Augmentation for Single-Source Domain Generalization in Medical Image Segmentation | 2023 | Proceedings of the AAAI Conference on Artificial Intelligence | 引用量较高，适合作为理论背景或问题定义入口。 |
| 高被引基础 | Domain and Content Adaptive Convolution Based Multi-Source Domain Generalization for Medical Image Segmentation | 2022 | IEEE Transactions on Medical Imaging | 引用量较高，适合作为理论背景或问题定义入口。 |
| 近期进展 | Devil is in Channels: Contrastive Single Domain Generalization for Medical Image Segmentation | 2023 | Lecture notes in computer science | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 近期进展 | From Denoising Training to Test-Time Adaptation: Enhancing Domain Generalization for Medical Image Segmentation | 2024 | IEEE Winter Conference on Applications of Computer Vision | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 近期进展 | Generative feature style augmentation for domain generalization in medical image segmentation | 2025 | Pattern Recognition | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 近期进展 | Dynamic domain generalization for medical image segmentation | 2024 | Neural Networks | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 代表工作 | Single-Domain Generalization in Medical Image Segmentation via Test-Time Adaptation from Shape Dictionary | 2022 | Proceedings of the AAAI Conference on Artificial Intelligence | 与主题词和核心技术路线高度相关，可作为分支阅读材料。 |
| 近期进展 | Treasure in Distribution: A Domain Randomization Based Multi-source Domain Generalization for 2D Medical Image Segmentation | 2023 | Lecture notes in computer science | 反映近年约束控制、鲁棒性或应用场景的新变化。 |

## 研究脉络

- 2019-2022：约 31 条；代表工作为《FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space》，该阶段重点可从 高被引基础 切入。
- 2023-2026：约 70 条；代表工作为《Rethinking Data Augmentation for Single-Source Domain Generalization in Medical Image Segmentation》，该阶段重点可从 高被引基础 切入。

## 主要方法对比

| 方法族 | 典型关注点 | 优势 | 常见限制 |
| --- | --- | --- | --- |
| Potential/barrier/navigation functions | 保持已有边、避免距离约束破坏 | 直观，适合编队与避障 | 参数调节和局部极值问题明显 |
| Algebraic connectivity / Fiedler value | 直接优化或估计图连通度 | 能刻画全局连通裕度 | 分布式估计和可扩展性是难点 |
| Event-triggered connectivity control | 减少通信和控制更新 | 适合资源受限网络 | 需要证明无 Zeno 行为和鲁棒性 |
| Prescribed performance / constrained control | 把连通、碰撞、误差约束统一处理 | 约束表达清晰 | 对模型假设和扰动界较敏感 |
| MPC / optimization-based methods | 在预测窗口内优化连通和任务性能 | 可处理多目标权衡 | 计算量和实时性压力较高 |

## 重要论文阅读顺序

1. FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space (2021), Quande Liu, Cheng Chen, Jing Qin, et al., IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=615, DOI/ID=10.1109/cvpr46437.2021.00107.
2. Causality-Inspired Single-Source Domain Generalization for Medical Image Segmentation (2022), Cheng Ouyang, Chen Chen, Surui Li, et al., IEEE Transactions on Medical Imaging; citations=291, DOI/ID=10.1109/tmi.2022.3224067.
3. Source free domain adaptation for medical image segmentation with fourier style mining (2022), Chen Yang, Xiaoqing Guo, Zhen Chen, et al., Medical Image Analysis; citations=148, DOI/ID=10.1016/j.media.2022.102457.
4. Rethinking Data Augmentation for Single-Source Domain Generalization in Medical Image Segmentation (2023), Zixian Su, Kai Yao, Xi Yang, et al., Proceedings of the AAAI Conference on Artificial Intelligence; citations=128, DOI/ID=10.1609/aaai.v37i2.25332.
5. Domain and Content Adaptive Convolution Based Multi-Source Domain Generalization for Medical Image Segmentation (2022), Shishuai Hu, Zehui Liao, Jianpeng Zhang, et al., IEEE Transactions on Medical Imaging; citations=121, DOI/ID=10.1109/tmi.2022.3210133.
6. Devil is in Channels: Contrastive Single Domain Generalization for Medical Image Segmentation (2023), Shishuai Hu, Zehui Liao, Yong Xia, Lecture notes in computer science; citations=77, DOI/ID=10.1007/978-3-031-43901-8_2.
7. Frequency-Mixed Single-Source Domain Generalization for Medical Image Segmentation (2023), Heng Li, Haojin Li, Wei Zhao, et al., Lecture notes in computer science; citations=52, DOI/ID=10.1007/978-3-031-43987-2_13.
8. Single-Domain Generalization in Medical Image Segmentation via Test-Time Adaptation from Shape Dictionary (2022), Quande Liu, Cheng Chen, Qi Dou, et al., Proceedings of the AAAI Conference on Artificial Intelligence; citations=51, DOI/ID=10.1609/aaai.v36i2.20068.
9. From Denoising Training to Test-Time Adaptation: Enhancing Domain Generalization for Medical Image Segmentation (2024), Ruxue Wen, Hangjie Yuan, Dong Ni, et al., IEEE Winter Conference on Applications of Computer Vision; citations=42, DOI/ID=10.1109/wacv57701.2024.00052.
10. Generative feature style augmentation for domain generalization in medical image segmentation (2025), Yunzhi Huang, Luyi Han, Haoran Dou, Pattern Recognition; citations=33, DOI/ID=10.1016/j.patcog.2025.111416.

## 局限与开放问题

- 从理论保证走向真实机器人/无人系统实验仍是关键缺口，尤其是有限感知、通信丢包和动态障碍同时存在时。
- 代数连通度类方法需要更可扩展的分布式估计与控制实现，避免全局谱信息成为隐含假设。
- 连通保持、碰撞避免、队形精度和能耗之间存在多目标冲突，需要更清晰的权衡机制。
- 事件触发和通信受限控制需要同时说明无 Zeno、鲁棒性和实际网络延迟下的性能。
- 面向攻击或故障的连通度控制还需要把安全检测、拓扑重构和控制律设计更紧密地结合。

## 网页补充建议

- `"medical image segmentation domain generalization" PDF`
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
