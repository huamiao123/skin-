# 主题调研简报：fine tuning pretrained features overfitting regularization

## 一句话结论

`fine tuning pretrained features overfitting regularization` 的检索结果显示，该主题的主线不是单纯的多智能体一致性，而是在编队、避障、通信受限、攻击或不确定性下维持通信/感知图的连通裕度。当前语料包含 48 条去重记录，年份覆盖 2018-2026，主要落在连通度保持与代数连通度控制、编队、队形与一致性控制、避障、安全约束与有限感知、事件触发、通信受限与分布式实现、鲁棒性、攻击与不确定系统等方向。

## 背景与问题定义

本简报基于可复现学术 API 检索生成，用于快速理解一个研究主题的核心问题、代表工作和后续阅读路径。在多智能体系统语境下，连通度控制通常关注通信图或感知图在运动、编队、避障、攻击、事件触发通信等条件下能否保持足够连通，常见度量包括图连通性、Laplacian 的 algebraic connectivity、邻接边保持、有限感知半径内的连通维护等。

## 检索策略与覆盖范围

- 原始查询：`fine tuning pretrained features overfitting regularization`
- 年份范围：2015 到 2026
- 去重后文献数：48
- 各源返回数量：openalex=18, semanticscholar=0, crossref=18, arxiv=17
- 检索错误：semanticscholar: HTTPError: HTTP Error 429: 
- 说明：该结果依赖 API 覆盖、查询式质量和元数据完整度；最终用于论文写作前仍需人工筛选全文。

## 技术路线分类

### 连通度保持与代数连通度控制

该类包含 9 条候选文献，覆盖 2019-2025。排名靠前的代表工作是《Improving Pretrained Language Model Fine-Tuning With Noise Stability Regularization》，可作为进入该分支的入口。

- Improving Pretrained Language Model Fine-Tuning With Noise Stability Regularization (2023), Hang Hua, Xingjian Li, Dejing Dou, et al., IEEE Transactions on Neural Networks and Learning Systems; citations=24, DOI/ID=10.1109/tnnls.2023.3330926.
- Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (2022), Ananya Kumar, Aditi Raghunathan, Jones, Robbie, et al., arXiv (Cornell University); citations=580, DOI/ID=10.48550/arxiv.2202.10054.
- CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features (2019), Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, et al., IEEE/CVF International Conference on Computer Vision (ICCV); citations=5690, DOI/ID=10.1109/iccv.2019.00612.
- Fine-tune the pretrained ATST model for sound event detection (2023), Nian Shao, Xian Li, Xiaofei Li, arXiv; citations=0, DOI/ID=2309.08153v2.
- Domain Generalization using Pretrained Models without Fine-tuning (2022), Ziyue Li, Ren, Kan, Xinyang Jiang, et al., arXiv (Cornell University); citations=43, DOI/ID=10.48550/arxiv.2203.04600.

### 编队、队形与一致性控制

该类包含 4 条候选文献，覆盖 2019-2025。排名靠前的代表工作是《CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features》，可作为进入该分支的入口。

- CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features (2019), Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, et al., IEEE/CVF International Conference on Computer Vision (ICCV); citations=5690, DOI/ID=10.1109/iccv.2019.00612.
- Quality Regularization-Based Semisupervised Adversarial Transfer Model With Unlabeled Data for Industrial Soft Sensing (2023), Yan‐Lin He, Lei Chen, Qun‐Xiong Zhu, IEEE Transactions on Industrial Informatics; citations=36, DOI/ID=10.1109/tii.2023.3272690.
- PARAMETER EFFICIENT FINE-TUNING AND OVERFITTING IN GPT LARGE LANGUAGE MODELS: A METRIC-BASED COMPARISON (2025), Bohdan Pavlyshenko, Ivan Bulka, Electronics and Information Technologies; citations=1, DOI/ID=10.30970/eli.30.3.
- Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs (2025), Zhixin Xie, Xurui Song, Jun Luo, Advances in Neural Information Processing Systems 38; citations=0, DOI/ID=10.52202/085713-1046.

### 避障、安全约束与有限感知

该类包含 5 条候选文献，覆盖 2021-2026。排名靠前的代表工作是《DR-Tune: Improving Fine-tuning of Pretrained Visual Models by Distribution Regularization with Semantic Calibration》，可作为进入该分支的入口。

- DR-Tune: Improving Fine-tuning of Pretrained Visual Models by Distribution Regularization with Semantic Calibration (2023), Nan Zhou, Jiaxin Chen, Di Huang, IEEE/CVF International Conference on Computer Vision (ICCV); citations=7, DOI/ID=10.1109/iccv51070.2023.00149.
- Quality Regularization-Based Semisupervised Adversarial Transfer Model With Unlabeled Data for Industrial Soft Sensing (2023), Yan‐Lin He, Lei Chen, Qun‐Xiong Zhu, IEEE Transactions on Industrial Informatics; citations=36, DOI/ID=10.1109/tii.2023.3272690.
- Pretraining Curricula Enable Selective Fine-tuning (2026), Sebastian A. Bruijns, Jirko Rubruck, Mia H. Whitefield, et al., arXiv (Cornell University); citations=0, DOI/ID=10.48550/arxiv.2607.04846.
- Differentially Private Fine-tuning of Language Models (2021), Da Yu, Saurabh Naik, Arturs Backurs, et al., arXiv; citations=0, DOI/ID=2110.06500v2.
- Legged Robots that Keep on Learning: Fine-Tuning Locomotion Policies in the Real World (2021), Laura Smith, J. Chase Kew, Xue Bin Peng, et al., arXiv; citations=0, DOI/ID=2110.05457v1.

### 事件触发、通信受限与分布式实现

该类包含 6 条候选文献，覆盖 2020-2025。排名靠前的代表工作是《Domain Adversarial Fine-Tuning as an Effective Regularizer》，可作为进入该分支的入口。

- Domain Adversarial Fine-Tuning as an Effective Regularizer (2020), Giorgos Vernikos, Katerina Margatina, Alexandra Chronopoulou, et al., Findings of the Association for Computational Linguistics: EMNLP (Findings of EMNLP); citations=11, DOI/ID=10.18653/v1/2020.findings-emnlp.278.
- DR-Tune: Improving Fine-tuning of Pretrained Visual Models by Distribution Regularization with Semantic Calibration (2023), Nan Zhou, Jiaxin Chen, Di Huang, IEEE/CVF International Conference on Computer Vision (ICCV); citations=7, DOI/ID=10.1109/iccv51070.2023.00149.
- Quality Regularization-Based Semisupervised Adversarial Transfer Model With Unlabeled Data for Industrial Soft Sensing (2023), Yan‐Lin He, Lei Chen, Qun‐Xiong Zhu, IEEE Transactions on Industrial Informatics; citations=36, DOI/ID=10.1109/tii.2023.3272690.
- Fine-tune the pretrained ATST model for sound event detection (2023), Nian Shao, Xian Li, Xiaofei Li, arXiv; citations=0, DOI/ID=2309.08153v2.
- Fine-Tuning Pretrained Language Models for Automated Research Papers Classification (2025), Mohammed Mghari, Abdelilah Mhamedi, Abdelaaziz El Hibaoui, 2025 International Conference on Circuit, Systems and Communication (ICCSC); citations=1, DOI/ID=10.1109/iccsc66714.2025.11135408.

### 鲁棒性、攻击与不确定系统

该类包含 6 条候选文献，覆盖 2019-2026。排名靠前的代表工作是《CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features》，可作为进入该分支的入口。

- CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features (2019), Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, et al., IEEE/CVF International Conference on Computer Vision (ICCV); citations=5690, DOI/ID=10.1109/iccv.2019.00612.
- Deep Learning Enhancements Using Pretraining and Fine-Tuning (2023), Kwame Nkosi, International Journal of Applied Data Science & Modern Computing; citations=0, DOI/ID=10.67228/30713498/ijadsmc-2023pii0q6w.
- Pretraining Curricula Enable Selective Fine-tuning (2026), Sebastian A. Bruijns, Jirko Rubruck, Mia H. Whitefield, et al., arXiv (Cornell University); citations=0, DOI/ID=10.48550/arxiv.2607.04846.
- Noise-Robust Fine-Tuning of Pretrained Language Models via External Guidance (2023), Song Wang, Zhen Tan, Ruocheng Guo, et al., Findings of the Association for Computational Linguistics: EMNLP 2023; citations=10, DOI/ID=10.18653/v1/2023.findings-emnlp.834.
- Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs (2025), Zhixin Xie, Xurui Song, Jun Luo, Advances in Neural Information Processing Systems 38; citations=0, DOI/ID=10.52202/085713-1046.

### 优化、MPC 与性能约束

该类包含 23 条候选文献，覆盖 2018-2026。排名靠前的代表工作是《Improving Pretrained Language Model Fine-Tuning With Noise Stability Regularization》，可作为进入该分支的入口。

- Improving Pretrained Language Model Fine-Tuning With Noise Stability Regularization (2023), Hang Hua, Xingjian Li, Dejing Dou, et al., IEEE Transactions on Neural Networks and Learning Systems; citations=24, DOI/ID=10.1109/tnnls.2023.3330926.
- Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment (2026), Lingling Xu, Haoran Xie, S. Joe Qin, et al., IEEE Transactions on Pattern Analysis and Machine Intelligence; citations=567, DOI/ID=10.1109/tpami.2026.3657354.
- Domain Adversarial Fine-Tuning as an Effective Regularizer (2020), Giorgos Vernikos, Katerina Margatina, Alexandra Chronopoulou, et al., Findings of the Association for Computational Linguistics: EMNLP (Findings of EMNLP); citations=11, DOI/ID=10.18653/v1/2020.findings-emnlp.278.
- CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features (2019), Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, et al., IEEE/CVF International Conference on Computer Vision (ICCV); citations=5690, DOI/ID=10.1109/iccv.2019.00612.
- Universal Language Model Fine-tuning for Text Classification (2018), Jeremy Howard, Sebastian Ruder, Proceedings of the conference - Association for Computational Linguistics. Meeting; citations=5075, DOI/ID=10.18653/v1/p18-1031.

### 综述、基础理论与背景材料

该类包含 3 条候选文献，覆盖 2018-2026。排名靠前的代表工作是《Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment》，可作为进入该分支的入口。

- Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment (2026), Lingling Xu, Haoran Xie, S. Joe Qin, et al., IEEE Transactions on Pattern Analysis and Machine Intelligence; citations=567, DOI/ID=10.1109/tpami.2026.3657354.
- Deep Learning Enhancements Using Pretraining and Fine-Tuning (2023), Kwame Nkosi, International Journal of Applied Data Science & Modern Computing; citations=0, DOI/ID=10.67228/30713498/ijadsmc-2023pii0q6w.
- Is The Universal Matter - Antimatter Asymmetry Fine Tuned? (2018), Gary Steigman, Robert J. Scherrer, arXiv; citations=0, DOI/ID=1801.10059v1.

### 其他相关工作

该类包含 14 条候选文献，覆盖 2020-2026。排名靠前的代表工作是《Parameter-Efficient Fine-Tuning via Meta-Regularizer》，可作为进入该分支的入口。

- Parameter-Efficient Fine-Tuning via Meta-Regularizer (2026), Jinyoung Park, Juyeon Ko, Sanghyeok Lee, et al., International Journal of Computer Vision; citations=0, DOI/ID=10.1007/s11263-025-02693-z.
- Noise Stability Regularization for Improving BERT Fine-tuning (2021), Hang Hua, Xingjian Li, Dejing Dou, et al., Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT); citations=32, DOI/ID=10.18653/v1/2021.naacl-main.258.
- LexFit: Lexical Fine-Tuning of Pretrained Language Models (2021), Ivan Vulić, Edoardo Maria Ponti, Anna Korhonen, et al., openalex; citations=26, DOI/ID=10.18653/v1/2021.acl-long.410.
- A tutorial on fine-tuning pretrained language models: Applications in social and behavioral science research (2025), Yu Wang, Wen Qu, Behavior Research Methods; citations=2, DOI/ID=10.3758/s13428-025-02868-z.
- Fine-Tuning Pre-trained Language Model with Weak Supervision: A Contrastive-Regularized Self-Training Approach (2021), Yue Yu, Simiao Zuo, Haoming Jiang, et al., Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT); citations=118, DOI/ID=10.18653/v1/2021.naacl-main.84.

## 代表性论文表

| 角色 | 论文 | 年份 | 来源/期刊会议 | 为什么重要 |
| --- | --- | --- | --- | --- |
| 近期进展 | Improving Pretrained Language Model Fine-Tuning With Noise Stability Regularization | 2023 | IEEE Transactions on Neural Networks and Learning Systems | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 高被引基础 | Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution | 2022 | arXiv (Cornell University) | 引用量较高，适合作为理论背景或问题定义入口。 |
| 综述入口 | Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment | 2026 | IEEE Transactions on Pattern Analysis and Machine Intelligence | 覆盖面广，适合快速建立术语和分支地图。 |
| 事件触发 | Domain Adversarial Fine-Tuning as an Effective Regularizer | 2020 | Findings of the Association for Computational Linguistics: EMNLP (Findings of EMNLP) | 连接连通度保持与低通信/低更新频率实现。 |
| 高被引基础 | CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features | 2019 | IEEE/CVF International Conference on Computer Vision (ICCV) | 引用量较高，适合作为理论背景或问题定义入口。 |
| 高被引基础 | Universal Language Model Fine-tuning for Text Classification | 2018 | Proceedings of the conference - Association for Computational Linguistics. Meeting | 引用量较高，适合作为理论背景或问题定义入口。 |
| 事件触发 | DR-Tune: Improving Fine-tuning of Pretrained Visual Models by Distribution Regularization with Semantic Calibration | 2023 | IEEE/CVF International Conference on Computer Vision (ICCV) | 连接连通度保持与低通信/低更新频率实现。 |
| 事件触发 | Quality Regularization-Based Semisupervised Adversarial Transfer Model With Unlabeled Data for Industrial Soft Sensing | 2023 | IEEE Transactions on Industrial Informatics | 连接连通度保持与低通信/低更新频率实现。 |
| 事件触发 | Fine-tune the pretrained ATST model for sound event detection | 2023 | arXiv | 连接连通度保持与低通信/低更新频率实现。 |
| 近期进展 | Parameter-Efficient Fine-Tuning via Meta-Regularizer | 2026 | International Journal of Computer Vision | 反映近年约束控制、鲁棒性或应用场景的新变化。 |

## 研究脉络

- 2015-2018：约 2 条；代表工作为《Universal Language Model Fine-tuning for Text Classification》，该阶段重点可从 高被引基础 切入。
- 2019-2022：约 16 条；代表工作为《CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features》，该阶段重点可从 高被引基础 切入。
- 2023-2026：约 30 条；代表工作为《Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment》，该阶段重点可从 综述入口 切入。

## 主要方法对比

| 方法族 | 典型关注点 | 优势 | 常见限制 |
| --- | --- | --- | --- |
| Potential/barrier/navigation functions | 保持已有边、避免距离约束破坏 | 直观，适合编队与避障 | 参数调节和局部极值问题明显 |
| Algebraic connectivity / Fiedler value | 直接优化或估计图连通度 | 能刻画全局连通裕度 | 分布式估计和可扩展性是难点 |
| Event-triggered connectivity control | 减少通信和控制更新 | 适合资源受限网络 | 需要证明无 Zeno 行为和鲁棒性 |
| Prescribed performance / constrained control | 把连通、碰撞、误差约束统一处理 | 约束表达清晰 | 对模型假设和扰动界较敏感 |
| MPC / optimization-based methods | 在预测窗口内优化连通和任务性能 | 可处理多目标权衡 | 计算量和实时性压力较高 |

## 重要论文阅读顺序

1. CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features (2019), Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, et al., IEEE/CVF International Conference on Computer Vision (ICCV); citations=5690, DOI/ID=10.1109/iccv.2019.00612.
2. Universal Language Model Fine-tuning for Text Classification (2018), Jeremy Howard, Sebastian Ruder, Proceedings of the conference - Association for Computational Linguistics. Meeting; citations=5075, DOI/ID=10.18653/v1/p18-1031.
3. Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (2022), Ananya Kumar, Aditi Raghunathan, Jones, Robbie, et al., arXiv (Cornell University); citations=580, DOI/ID=10.48550/arxiv.2202.10054.
4. Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment (2026), Lingling Xu, Haoran Xie, S. Joe Qin, et al., IEEE Transactions on Pattern Analysis and Machine Intelligence; citations=567, DOI/ID=10.1109/tpami.2026.3657354.
5. Fine-Tuning Pre-trained Language Model with Weak Supervision: A Contrastive-Regularized Self-Training Approach (2021), Yue Yu, Simiao Zuo, Haoming Jiang, et al., Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT); citations=118, DOI/ID=10.18653/v1/2021.naacl-main.84.
6. LoRA dropout as a sparsity regularizer for overfitting reduction (2025), Yang Lin, Xinyu Ma, Xu Chu, et al., Knowledge-Based Systems; citations=76, DOI/ID=10.1016/j.knosys.2025.114241.
7. Domain Generalization using Pretrained Models without Fine-tuning (2022), Ziyue Li, Ren, Kan, Xinyang Jiang, et al., arXiv (Cornell University); citations=43, DOI/ID=10.48550/arxiv.2203.04600.
8. Quality Regularization-Based Semisupervised Adversarial Transfer Model With Unlabeled Data for Industrial Soft Sensing (2023), Yan‐Lin He, Lei Chen, Qun‐Xiong Zhu, IEEE Transactions on Industrial Informatics; citations=36, DOI/ID=10.1109/tii.2023.3272690.
9. Improving Pretrained Language Model Fine-Tuning With Noise Stability Regularization (2023), Hang Hua, Xingjian Li, Dejing Dou, et al., IEEE Transactions on Neural Networks and Learning Systems; citations=24, DOI/ID=10.1109/tnnls.2023.3330926.
10. Domain Adversarial Fine-Tuning as an Effective Regularizer (2020), Giorgos Vernikos, Katerina Margatina, Alexandra Chronopoulou, et al., Findings of the Association for Computational Linguistics: EMNLP (Findings of EMNLP); citations=11, DOI/ID=10.18653/v1/2020.findings-emnlp.278.

## 局限与开放问题

- 从理论保证走向真实机器人/无人系统实验仍是关键缺口，尤其是有限感知、通信丢包和动态障碍同时存在时。
- 代数连通度类方法需要更可扩展的分布式估计与控制实现，避免全局谱信息成为隐含假设。
- 连通保持、碰撞避免、队形精度和能耗之间存在多目标冲突，需要更清晰的权衡机制。
- 事件触发和通信受限控制需要同时说明无 Zeno、鲁棒性和实际网络延迟下的性能。
- 面向攻击或故障的连通度控制还需要把安全检测、拓扑重构和控制律设计更紧密地结合。

## 网页补充建议

- `"fine tuning pretrained features overfitting regularization" PDF`
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
