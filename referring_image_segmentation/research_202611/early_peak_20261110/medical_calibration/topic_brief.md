# 主题调研简报：medical image segmentation confidence calibration overfitting

## 一句话结论

`medical image segmentation confidence calibration overfitting` 的检索结果显示，该主题的主线不是单纯的多智能体一致性，而是在编队、避障、通信受限、攻击或不确定性下维持通信/感知图的连通裕度。当前语料包含 50 条去重记录，年份覆盖 2016-2026，主要落在连通度保持与代数连通度控制、编队、队形与一致性控制、避障、安全约束与有限感知、事件触发、通信受限与分布式实现、鲁棒性、攻击与不确定系统等方向。

## 背景与问题定义

本简报基于可复现学术 API 检索生成，用于快速理解一个研究主题的核心问题、代表工作和后续阅读路径。在多智能体系统语境下，连通度控制通常关注通信图或感知图在运动、编队、避障、攻击、事件触发通信等条件下能否保持足够连通，常见度量包括图连通性、Laplacian 的 algebraic connectivity、邻接边保持、有限感知半径内的连通维护等。

## 检索策略与覆盖范围

- 原始查询：`medical image segmentation confidence calibration overfitting`
- 年份范围：2015 到 2026
- 去重后文献数：50
- 各源返回数量：openalex=18, semanticscholar=0, crossref=18, arxiv=17
- 检索错误：semanticscholar: HTTPError: HTTP Error 429: 
- 说明：该结果依赖 API 覆盖、查询式质量和元数据完整度；最终用于论文写作前仍需人工筛选全文。

## 技术路线分类

### 连通度保持与代数连通度控制

该类包含 24 条候选文献，覆盖 2018-2026。排名靠前的代表工作是《Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation》，可作为进入该分支的入口。

- Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation (2020), Alireza Mehrtash, William M. Wells, Clare M. Tempany, et al., IEEE Transactions on Medical Imaging; citations=475, DOI/ID=10.1109/tmi.2020.3006437.
- Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation (2024), Theodore Barfoot, Luis C. García-Peraza-Herrera, Ben Glocker, et al., Lecture notes in computer science; citations=11, DOI/ID=10.1007/978-3-031-72114-4_14.
- CALSeg: Improving Calibration of Medical Image Segmentation Via Variational Label Smoothing (2024), Xutao Guo, Yanwu Yang, Chenfei Ye, et al., IEEE International Conference on Acoustics Speech and Signal Processing; citations=3, DOI/ID=10.1109/icassp48485.2024.10446030.
- Studying the Impact of Augmentations on Medical Confidence Calibration (2023), Adrit Rao, Joon‐Young Lee, Oliver Aalami, IEEE International Conference on Computer Vision Workshops; citations=10, DOI/ID=10.1109/iccvw60793.2023.00260.
- Medical Image Segmentation With Limited Supervision: A Review of Deep Network Models (2021), Jialin Peng, Ye Wang, IEEE Access; citations=115, DOI/ID=10.1109/access.2021.3062380.

### 编队、队形与一致性控制

该类包含 7 条候选文献，覆盖 2020-2024。排名靠前的代表工作是《CALSeg: Improving Calibration of Medical Image Segmentation Via Variational Label Smoothing》，可作为进入该分支的入口。

- CALSeg: Improving Calibration of Medical Image Segmentation Via Variational Label Smoothing (2024), Xutao Guo, Yanwu Yang, Chenfei Ye, et al., IEEE International Conference on Acoustics Speech and Signal Processing; citations=3, DOI/ID=10.1109/icassp48485.2024.10446030.
- Multivariate Confidence Calibration for Object Detection (2020), Fabian Küppers, Jan Kronenberger, Amirhossein Shantia, et al., IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW); citations=132, DOI/ID=10.1109/cvprw50498.2020.00171.
- Diminishing Uncertainty Within the Training Pool: Active Learning for Medical Image Segmentation (2020), Vishwesh Nath, Dong Yang, Bennett A. Landman, et al., IEEE Transactions on Medical Imaging; citations=118, DOI/ID=10.1109/tmi.2020.3048055.
- Test-time generative augmentation for medical image segmentation (2024), Xiao Ma, Yuhui Tao, Zetian Zhang, et al., arXiv; citations=0, DOI/ID=10.1016/j.media.2025.103902.
- Anomaly Detection-Inspired Few-Shot Medical Image Segmentation Through Self-Supervision With Supervoxels (2022), Stine Hansen, Srishti Gautam, Robert Jenssen, et al., arXiv; citations=0, DOI/ID=10.1016/j.media.2022.102385.

### 避障、安全约束与有限感知

该类包含 4 条候选文献，覆盖 2020-2026。排名靠前的代表工作是《Multivariate Confidence Calibration for Object Detection》，可作为进入该分支的入口。

- Multivariate Confidence Calibration for Object Detection (2020), Fabian Küppers, Jan Kronenberger, Amirhossein Shantia, et al., IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW); citations=132, DOI/ID=10.1109/cvprw50498.2020.00171.
- Reliability-Preserving Compression for Deployable Medical Image Segmentation (2026), Sagar Kumar, Shailja Sharma, Tirthajit Baruah, et al., arXiv (Cornell University); citations=0, DOI/ID=10.48550/arxiv.2610.25460.
- Learning with imperfect datasets in medical image segmentation (2024), Yuhang Ding, OPUS - Open Publications of UTS Scholars (University of Technology Sydney); citations=0, DOI/ID=-.
- Cross-dimensional transfer learning in medical image segmentation with deep learning (2023), Hicham Messaoudi, Ahror Belaid, Douraied Ben Salem, et al., arXiv; citations=0, DOI/ID=10.1016/j.media.2023.102868.

### 事件触发、通信受限与分布式实现

该类包含 2 条候选文献，覆盖 2019-2024。排名靠前的代表工作是《CALSeg: Improving Calibration of Medical Image Segmentation Via Variational Label Smoothing》，可作为进入该分支的入口。

- CALSeg: Improving Calibration of Medical Image Segmentation Via Variational Label Smoothing (2024), Xutao Guo, Yanwu Yang, Chenfei Ye, et al., IEEE International Conference on Acoustics Speech and Signal Processing; citations=3, DOI/ID=10.1109/icassp48485.2024.10446030.
- CNN Architecture for Surgical Image Segmentation Systems with Recursive Network Structure to Mitigate Overfitting (2019), Taito Manabe, Koki Tomonaga, Yuichiro Shibata, 2019 Seventh International Symposium on Computing and Networking (CANDAR); citations=3, DOI/ID=10.1109/candar.2019.00029.

### 鲁棒性、攻击与不确定系统

该类包含 20 条候选文献，覆盖 2020-2026。排名靠前的代表工作是《Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation》，可作为进入该分支的入口。

- Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation (2020), Alireza Mehrtash, William M. Wells, Clare M. Tempany, et al., IEEE Transactions on Medical Imaging; citations=475, DOI/ID=10.1109/tmi.2020.3006437.
- Calibrating ensembles for scalable uncertainty quantification in deep learning-based medical image segmentation (2023), Thomas Buddenkotte, Lorena Escudero Sánchez, Mireia Crispin‐Ortuzar, et al., Computers in Biology and Medicine; citations=50, DOI/ID=10.1016/j.compbiomed.2023.107096.
- Studying the Impact of Augmentations on Medical Confidence Calibration (2023), Adrit Rao, Joon‐Young Lee, Oliver Aalami, IEEE International Conference on Computer Vision Workshops; citations=10, DOI/ID=10.1109/iccvw60793.2023.00260.
- Diminishing Uncertainty Within the Training Pool: Active Learning for Medical Image Segmentation (2020), Vishwesh Nath, Dong Yang, Bennett A. Landman, et al., IEEE Transactions on Medical Imaging; citations=118, DOI/ID=10.1109/tmi.2020.3048055.
- Maximum Entropy on Erroneous Predictions (MEEP): Improving model calibration for medical image segmentation (2021), Agostina J. Larrazabal, César E. Martínez, José Dolz, et al., arXiv (Cornell University); citations=9, DOI/ID=10.48550/arxiv.2112.12218.

### 优化、MPC 与性能约束

该类包含 23 条候选文献，覆盖 2018-2026。排名靠前的代表工作是《Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation》，可作为进入该分支的入口。

- Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation (2020), Alireza Mehrtash, William M. Wells, Clare M. Tempany, et al., IEEE Transactions on Medical Imaging; citations=475, DOI/ID=10.1109/tmi.2020.3006437.
- CALSeg: Improving Calibration of Medical Image Segmentation Via Variational Label Smoothing (2024), Xutao Guo, Yanwu Yang, Chenfei Ye, et al., IEEE International Conference on Acoustics Speech and Signal Processing; citations=3, DOI/ID=10.1109/icassp48485.2024.10446030.
- Studying the Impact of Augmentations on Medical Confidence Calibration (2023), Adrit Rao, Joon‐Young Lee, Oliver Aalami, IEEE International Conference on Computer Vision Workshops; citations=10, DOI/ID=10.1109/iccvw60793.2023.00260.
- Medical Image Segmentation With Limited Supervision: A Review of Deep Network Models (2021), Jialin Peng, Ye Wang, IEEE Access; citations=115, DOI/ID=10.1109/access.2021.3062380.
- Uncertainty Quantification in Medical Image Segmentation: A Comprehensive Survey (2026), Seyed Sina Ziaee, Katie L. Ovens, Journal of Imaging; citations=1, DOI/ID=10.3390/jimaging12080341.

### 综述、基础理论与背景材料

该类包含 7 条候选文献，覆盖 2021-2026。排名靠前的代表工作是《Learning with limited annotations: A survey on deep semi-supervised learning for medical image segmentation》，可作为进入该分支的入口。

- Learning with limited annotations: A survey on deep semi-supervised learning for medical image segmentation (2023), Rushi Jiao, Yichi Zhang, Le Ding, et al., Computers in Biology and Medicine; citations=403, DOI/ID=10.1016/j.compbiomed.2023.107840.
- Medical Image Segmentation With Limited Supervision: A Review of Deep Network Models (2021), Jialin Peng, Ye Wang, IEEE Access; citations=115, DOI/ID=10.1109/access.2021.3062380.
- Maximum Entropy on Erroneous Predictions (MEEP): Improving model calibration for medical image segmentation (2021), Agostina J. Larrazabal, César E. Martínez, José Dolz, et al., arXiv (Cornell University); citations=9, DOI/ID=10.48550/arxiv.2112.12218.
- Uncertainty Quantification in Medical Image Segmentation: A Comprehensive Survey (2026), Seyed Sina Ziaee, Katie L. Ovens, Journal of Imaging; citations=1, DOI/ID=10.3390/jimaging12080341.
- Medical image segmentation method based on confidence learning and collaborative training (2024), Zhao Hongwei, Chen Yuhan, Liu Pingping, et al., Scientific Insights and Discoveries Review; citations=0, DOI/ID=10.59782/sidr.v2i1.117.

### 其他相关工作

该类包含 10 条候选文献，覆盖 2016-2026。排名靠前的代表工作是《Generalizable Medical Image Segmentation via Random Amplitude Mixup and Domain-Specific Image Restoration》，可作为进入该分支的入口。

- Generalizable Medical Image Segmentation via Random Amplitude Mixup and Domain-Specific Image Restoration (2022), Ziqi Zhou, Lei Qi, Yinghuan Shi, Lecture notes in computer science; citations=70, DOI/ID=10.1007/978-3-031-19803-8_25.
- Training-Free One-Shot Medical Image Segmentation with SAM via Confidence and Consistency-Based Bidirectional Prompt Evolution (2026), Muyang Li, Kang Li, Guotai Wang, 2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI); citations=0, DOI/ID=10.1109/isbi61048.2026.11515319.
- Semi-supervised Medical Image Segmentation with Low-Confidence Consistency and Class Separation (2023), Zhimin Gao, Tianyou Yu, 2023 International Conference on Intelligent Supercomputing and BioPharma (ISBP); citations=1, DOI/ID=10.1109/isbp57705.2023.10061306.
- Invariant Content Representation for Generalizable Medical Image Segmentation (2024), Zhiming Cheng, Shuai Wang, Yuhan Gao, et al., Journal of Imaging Informatics in Medicine; citations=4, DOI/ID=10.1007/s10278-024-01088-9.
- Non-equivalent images and pixels: Confidence-aware resampling with meta-learning mixup for polyp segmentation (2022), Xiaoqing Guo, Zhen Chen, Jun Liu, et al., Medical Image Analysis; citations=30, DOI/ID=10.1016/j.media.2022.102394.

## 代表性论文表

| 角色 | 论文 | 年份 | 来源/期刊会议 | 为什么重要 |
| --- | --- | --- | --- | --- |
| 高被引基础 | Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation | 2020 | IEEE Transactions on Medical Imaging | 引用量较高，适合作为理论背景或问题定义入口。 |
| 近期进展 | Calibrating ensembles for scalable uncertainty quantification in deep learning-based medical image segmentation | 2023 | Computers in Biology and Medicine | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 代表工作 | Generalizable Medical Image Segmentation via Random Amplitude Mixup and Domain-Specific Image Restoration | 2022 | Lecture notes in computer science | 与主题词和核心技术路线高度相关，可作为分支阅读材料。 |
| 综述入口 | Learning with limited annotations: A survey on deep semi-supervised learning for medical image segmentation | 2023 | Computers in Biology and Medicine | 覆盖面广，适合快速建立术语和分支地图。 |
| 近期进展 | Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation | 2024 | Lecture notes in computer science | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 事件触发 | CALSeg: Improving Calibration of Medical Image Segmentation Via Variational Label Smoothing | 2024 | IEEE International Conference on Acoustics Speech and Signal Processing | 连接连通度保持与低通信/低更新频率实现。 |
| 近期进展 | Studying the Impact of Augmentations on Medical Confidence Calibration | 2023 | IEEE International Conference on Computer Vision Workshops | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 综述入口 | Medical Image Segmentation With Limited Supervision: A Review of Deep Network Models | 2021 | IEEE Access | 覆盖面广，适合快速建立术语和分支地图。 |
| 高被引基础 | Multivariate Confidence Calibration for Object Detection | 2020 | IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) | 引用量较高，适合作为理论背景或问题定义入口。 |
| 高被引基础 | Diminishing Uncertainty Within the Training Pool: Active Learning for Medical Image Segmentation | 2020 | IEEE Transactions on Medical Imaging | 引用量较高，适合作为理论背景或问题定义入口。 |

## 研究脉络

- 2015-2018：约 3 条；代表工作为《Image Segmentation Based on Statistical Confidence Intervals》，该阶段重点可从 代表工作 切入。
- 2019-2022：约 19 条；代表工作为《Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation》，该阶段重点可从 高被引基础 切入。
- 2023-2026：约 28 条；代表工作为《Learning with limited annotations: A survey on deep semi-supervised learning for medical image segmentation》，该阶段重点可从 综述入口 切入。

## 主要方法对比

| 方法族 | 典型关注点 | 优势 | 常见限制 |
| --- | --- | --- | --- |
| Potential/barrier/navigation functions | 保持已有边、避免距离约束破坏 | 直观，适合编队与避障 | 参数调节和局部极值问题明显 |
| Algebraic connectivity / Fiedler value | 直接优化或估计图连通度 | 能刻画全局连通裕度 | 分布式估计和可扩展性是难点 |
| Event-triggered connectivity control | 减少通信和控制更新 | 适合资源受限网络 | 需要证明无 Zeno 行为和鲁棒性 |
| Prescribed performance / constrained control | 把连通、碰撞、误差约束统一处理 | 约束表达清晰 | 对模型假设和扰动界较敏感 |
| MPC / optimization-based methods | 在预测窗口内优化连通和任务性能 | 可处理多目标权衡 | 计算量和实时性压力较高 |

## 重要论文阅读顺序

1. Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation (2020), Alireza Mehrtash, William M. Wells, Clare M. Tempany, et al., IEEE Transactions on Medical Imaging; citations=475, DOI/ID=10.1109/tmi.2020.3006437.
2. Learning with limited annotations: A survey on deep semi-supervised learning for medical image segmentation (2023), Rushi Jiao, Yichi Zhang, Le Ding, et al., Computers in Biology and Medicine; citations=403, DOI/ID=10.1016/j.compbiomed.2023.107840.
3. Multivariate Confidence Calibration for Object Detection (2020), Fabian Küppers, Jan Kronenberger, Amirhossein Shantia, et al., IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW); citations=132, DOI/ID=10.1109/cvprw50498.2020.00171.
4. Diminishing Uncertainty Within the Training Pool: Active Learning for Medical Image Segmentation (2020), Vishwesh Nath, Dong Yang, Bennett A. Landman, et al., IEEE Transactions on Medical Imaging; citations=118, DOI/ID=10.1109/tmi.2020.3048055.
5. Medical Image Segmentation With Limited Supervision: A Review of Deep Network Models (2021), Jialin Peng, Ye Wang, IEEE Access; citations=115, DOI/ID=10.1109/access.2021.3062380.
6. Generalizable Medical Image Segmentation via Random Amplitude Mixup and Domain-Specific Image Restoration (2022), Ziqi Zhou, Lei Qi, Yinghuan Shi, Lecture notes in computer science; citations=70, DOI/ID=10.1007/978-3-031-19803-8_25.
7. Calibrating ensembles for scalable uncertainty quantification in deep learning-based medical image segmentation (2023), Thomas Buddenkotte, Lorena Escudero Sánchez, Mireia Crispin‐Ortuzar, et al., Computers in Biology and Medicine; citations=50, DOI/ID=10.1016/j.compbiomed.2023.107096.
8. Non-equivalent images and pixels: Confidence-aware resampling with meta-learning mixup for polyp segmentation (2022), Xiaoqing Guo, Zhen Chen, Jun Liu, et al., Medical Image Analysis; citations=30, DOI/ID=10.1016/j.media.2022.102394.
9. Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation (2024), Theodore Barfoot, Luis C. García-Peraza-Herrera, Ben Glocker, et al., Lecture notes in computer science; citations=11, DOI/ID=10.1007/978-3-031-72114-4_14.
10. CALSeg: Improving Calibration of Medical Image Segmentation Via Variational Label Smoothing (2024), Xutao Guo, Yanwu Yang, Chenfei Ye, et al., IEEE International Conference on Acoustics Speech and Signal Processing; citations=3, DOI/ID=10.1109/icassp48485.2024.10446030.

## 局限与开放问题

- 从理论保证走向真实机器人/无人系统实验仍是关键缺口，尤其是有限感知、通信丢包和动态障碍同时存在时。
- 代数连通度类方法需要更可扩展的分布式估计与控制实现，避免全局谱信息成为隐含假设。
- 连通保持、碰撞避免、队形精度和能耗之间存在多目标冲突，需要更清晰的权衡机制。
- 事件触发和通信受限控制需要同时说明无 Zeno、鲁棒性和实际网络延迟下的性能。
- 面向攻击或故障的连通度控制还需要把安全检测、拓扑重构和控制律设计更紧密地结合。

## 网页补充建议

- `"medical image segmentation confidence calibration overfitting" PDF`
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
