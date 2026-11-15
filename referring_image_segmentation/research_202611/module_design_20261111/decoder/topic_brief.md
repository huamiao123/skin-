> INVALID AUTO-GENERATED PROSE: unrelated fixed template. Do not use as research evidence. See ../deep_research_report_zh.txt. Raw retrieval metadata remains in papers.json.

# 主题调研简报：medical image segmentation prediction guided decoder multi scale

## 一句话结论

`medical image segmentation prediction guided decoder multi scale` 的检索结果显示，该主题的主线不是单纯的多智能体一致性，而是在编队、避障、通信受限、攻击或不确定性下维持通信/感知图的连通裕度。当前语料包含 33 条去重记录，年份覆盖 2020-2026，主要落在连通度保持与代数连通度控制、编队、队形与一致性控制、避障、安全约束与有限感知、事件触发、通信受限与分布式实现、鲁棒性、攻击与不确定系统等方向。

## 背景与问题定义

本简报基于可复现学术 API 检索生成，用于快速理解一个研究主题的核心问题、代表工作和后续阅读路径。在多智能体系统语境下，连通度控制通常关注通信图或感知图在运动、编队、避障、攻击、事件触发通信等条件下能否保持足够连通，常见度量包括图连通性、Laplacian 的 algebraic connectivity、邻接边保持、有限感知半径内的连通维护等。

## 检索策略与覆盖范围

- 原始查询：`medical image segmentation prediction guided decoder multi scale`
- 年份范围：2020 到 2026
- 去重后文献数：33
- 各源返回数量：openalex=20, semanticscholar=0, crossref=20, arxiv=0
- 检索错误：semanticscholar: HTTPError: HTTP Error 429: ; arxiv: HTTPError: HTTP Error 429: Unknown Error
- 说明：该结果依赖 API 覆盖、查询式质量和元数据完整度；最终用于论文写作前仍需人工筛选全文。

## 技术路线分类

### 连通度保持与代数连通度控制

该类包含 19 条候选文献，覆盖 2020-2026。排名靠前的代表工作是《TransAttUnet: Multi-Level Attention-Guided U-Net With Transformer for Medical Image Segmentation》，可作为进入该分支的入口。

- TransAttUnet: Multi-Level Attention-Guided U-Net With Transformer for Medical Image Segmentation (2023), Bingzhi Chen, Yishu Liu, Zheng Zhang, et al., IEEE Transactions on Emerging Topics in Computational Intelligence; citations=544, DOI/ID=10.1109/tetci.2023.3309626.
- Multi-Scale Self-Guided Attention for Medical Image Segmentation (2020), Ashish Sinha, José Dolz, IEEE Journal of Biomedical and Health Informatics; citations=596, DOI/ID=10.1109/jbhi.2020.2986926.
- MISSFormer: An Effective Transformer for 2D Medical Image Segmentation (2022), Xiaohong Huang, Zhifang Deng, Dandan Li, et al., IEEE Transactions on Medical Imaging; citations=974, DOI/ID=10.1109/tmi.2022.3230943.
- MDA-Unet: A Multi-Scale Dilated Attention U-Net for Medical Image Segmentation (2022), Alyaa Amer, Tryphon Lambrou, Xujiong Ye, Applied Sciences; citations=63, DOI/ID=10.3390/app12073676.
- MSA$^2$Net: Multi-scale Adaptive Attention-guided Network for Medical Image Segmentation (2024), Sina Ghorbani Kolahi, S. Kamal Chaharsooghi, Toktam Khatibi, et al., arXiv (Cornell University); citations=29, DOI/ID=10.48550/arxiv.2407.21640.

### 编队、队形与一致性控制

该类包含 14 条候选文献，覆盖 2020-2026。排名靠前的代表工作是《TransAttUnet: Multi-Level Attention-Guided U-Net With Transformer for Medical Image Segmentation》，可作为进入该分支的入口。

- TransAttUnet: Multi-Level Attention-Guided U-Net With Transformer for Medical Image Segmentation (2023), Bingzhi Chen, Yishu Liu, Zheng Zhang, et al., IEEE Transactions on Emerging Topics in Computational Intelligence; citations=544, DOI/ID=10.1109/tetci.2023.3309626.
- Multi-Scale Self-Guided Attention for Medical Image Segmentation (2020), Ashish Sinha, José Dolz, IEEE Journal of Biomedical and Health Informatics; citations=596, DOI/ID=10.1109/jbhi.2020.2986926.
- MDA-Unet: A Multi-Scale Dilated Attention U-Net for Medical Image Segmentation (2022), Alyaa Amer, Tryphon Lambrou, Xujiong Ye, Applied Sciences; citations=63, DOI/ID=10.3390/app12073676.
- MSA$^2$Net: Multi-scale Adaptive Attention-guided Network for Medical Image Segmentation (2024), Sina Ghorbani Kolahi, S. Kamal Chaharsooghi, Toktam Khatibi, et al., arXiv (Cornell University); citations=29, DOI/ID=10.48550/arxiv.2407.21640.
- Edge Information-Guided 3D UNet for Medical Image Segmentation (2026), Shaolong Chen, Yicai Gong, Fei Luo, et al., International Conference on Artificial Intelligence and Electromechanical Automation (AIEA); citations=0, DOI/ID=10.1109/aiea70743.2026.11632338.

### 避障、安全约束与有限感知

该类包含 2 条候选文献，覆盖 2024-2025。排名靠前的代表工作是《Consistency-Guided Differential Decoding for Enhancing Semi-Supervised Medical Image Segmentation》，可作为进入该分支的入口。

- Consistency-Guided Differential Decoding for Enhancing Semi-Supervised Medical Image Segmentation (2024), Qingjie Zeng, Yutong Xie, Zilin Lu, et al., IEEE Transactions on Medical Imaging; citations=100, DOI/ID=10.1109/tmi.2024.3429340.
- Uncertainty-Guided Hybrid Decoder for Medical Image Segmentation (2025), Chen Lu, Jiajie Mu, IEEE International Conference on Pattern Recognition, Machine Vision and Artificial Intelligence (PRMVAI); citations=0, DOI/ID=10.1109/prmvai65741.2025.11108450.

### 事件触发、通信受限与分布式实现

该类包含 1 条候选文献，覆盖 2020-2020。排名靠前的代表工作是《Encoder-decoder with multi-scale information fusion for semantic image segmentation》，可作为进入该分支的入口。

- Encoder-decoder with multi-scale information fusion for semantic image segmentation (2020), Xinxin Ma, Kai Liu, Chongyang Ding, et al., Eleventh International Conference on Graphics and Image Processing (ICGIP 2019); citations=1, DOI/ID=10.1117/12.2557506.

### 鲁棒性、攻击与不确定系统

该类包含 5 条候选文献，覆盖 2022-2026。排名靠前的代表工作是《MISSFormer: An Effective Transformer for 2D Medical Image Segmentation》，可作为进入该分支的入口。

- MISSFormer: An Effective Transformer for 2D Medical Image Segmentation (2022), Xiaohong Huang, Zhifang Deng, Dandan Li, et al., IEEE Transactions on Medical Imaging; citations=974, DOI/ID=10.1109/tmi.2022.3230943.
- Modality-Agnostic Domain Generalizable Medical Image Segmentation by Multi-Frequency in Multi-Scale Attention (2024), Ju-Hyeon Nam, Nur Suriza Syazwany, Su Jung Kim, et al., IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=132, DOI/ID=10.1109/cvpr52733.2024.01091.
- Consistency-Guided Differential Decoding for Enhancing Semi-Supervised Medical Image Segmentation (2024), Qingjie Zeng, Yutong Xie, Zilin Lu, et al., IEEE Transactions on Medical Imaging; citations=100, DOI/ID=10.1109/tmi.2024.3429340.
- Uncertainty-Guided Hybrid Decoder for Medical Image Segmentation (2025), Chen Lu, Jiajie Mu, IEEE International Conference on Pattern Recognition, Machine Vision and Artificial Intelligence (PRMVAI); citations=0, DOI/ID=10.1109/prmvai65741.2025.11108450.
- Multi-scale Representations Using Transformers for Enhanced 3D Medical Image Segmentation with CLIP-Guided Textual Prompt (2026), Fangliang Huang, Huanqing Xu, Xian Shao, et al., crossref; citations=0, DOI/ID=10.2139/ssrn.6602382.

### 优化、MPC 与性能约束

该类包含 15 条候选文献，覆盖 2020-2026。排名靠前的代表工作是《TransAttUnet: Multi-Level Attention-Guided U-Net With Transformer for Medical Image Segmentation》，可作为进入该分支的入口。

- TransAttUnet: Multi-Level Attention-Guided U-Net With Transformer for Medical Image Segmentation (2023), Bingzhi Chen, Yishu Liu, Zheng Zhang, et al., IEEE Transactions on Emerging Topics in Computational Intelligence; citations=544, DOI/ID=10.1109/tetci.2023.3309626.
- Multi-Scale Self-Guided Attention for Medical Image Segmentation (2020), Ashish Sinha, José Dolz, IEEE Journal of Biomedical and Health Informatics; citations=596, DOI/ID=10.1109/jbhi.2020.2986926.
- EMCAD: Efficient Multi-Scale Convolutional Attention Decoding for Medical Image Segmentation (2024), Md Mostafijur Rahman, Mustafa Munir, Radu Marculescu, IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=615, DOI/ID=10.1109/cvpr52733.2024.01118.
- MISSFormer: An Effective Transformer for 2D Medical Image Segmentation (2022), Xiaohong Huang, Zhifang Deng, Dandan Li, et al., IEEE Transactions on Medical Imaging; citations=974, DOI/ID=10.1109/tmi.2022.3230943.
- Medical Image Segmentation Review: The Success of U-Net (2024), Reza Azad, Ehsan Khodapanah Aghdam, Amelie Rauland, et al., IEEE Transactions on Pattern Analysis and Machine Intelligence; citations=1304, DOI/ID=10.1109/tpami.2024.3435571.

### 综述、基础理论与背景材料

该类包含 4 条候选文献，覆盖 2020-2024。排名靠前的代表工作是《Medical Image Segmentation Review: The Success of U-Net》，可作为进入该分支的入口。

- Medical Image Segmentation Review: The Success of U-Net (2024), Reza Azad, Ehsan Khodapanah Aghdam, Amelie Rauland, et al., IEEE Transactions on Pattern Analysis and Machine Intelligence; citations=1304, DOI/ID=10.1109/tpami.2024.3435571.
- Medical image segmentation using deep learning: A survey (2022), Risheng Wang, Tao Lei, Ruixia Cui, et al., IET Image Processing; citations=1020, DOI/ID=10.1049/ipr2.12419.
- Deep semantic segmentation of natural and medical images: a review (2020), Saeid Asgari Taghanaki, Kumar Abhishek, Joseph Cohen, et al., Artificial Intelligence Review; citations=935, DOI/ID=10.1007/s10462-020-09854-1.
- Review for "Medical image segmentation method based on multi‐scale feature and U‐net network" (2023), Unknown authors, crossref; citations=0, DOI/ID=10.1002/itl2.451/v2/review1.

### 其他相关工作

该类包含 5 条候选文献，覆盖 2022-2026。排名靠前的代表工作是《Customized Segment Anything Model for Medical Image Segmentation》，可作为进入该分支的入口。

- Customized Segment Anything Model for Medical Image Segmentation (2023), Kaidong Zhang, Dong Liu, arXiv (Cornell University); citations=454, DOI/ID=10.48550/arxiv.2304.13785.
- MSDNet: Multi-scale decoder for few-shot semantic segmentation via transformer-guided prototyping (2025), Amirreza Fateh, Mohammad Reza Mohammadi, Mohammad Reza Jahed-Motlagh, Image and Vision Computing; citations=27, DOI/ID=10.1016/j.imavis.2025.105672.
- CSAFD: Cross-scale attention fusion decoder for medical image segmentation (2026), Xiaoshan Pan, Bin Yao, Yuzhao Liu, et al., Signal, Image and Video Processing; citations=0, DOI/ID=10.1007/s11760-026-05269-7.
- A Semi-Supervised Medical Image Segmentation Framework Based on Pseudo-Label Discrepancy and Multi-Decoder Collaboration (2025), Dongen Guo, Xin Lu, Xufan Chen, et al., crossref; citations=0, DOI/ID=10.2139/ssrn.5397011.
- Semi-supervised multi-scale dense based encoder-decoder architecture for lung computed tomography image segmentation (2022), Shang Gao, International Conference on Optics and Machine Vision (ICOMV 2022); citations=0, DOI/ID=10.1117/12.2634417.

## 代表性论文表

| 角色 | 论文 | 年份 | 来源/期刊会议 | 为什么重要 |
| --- | --- | --- | --- | --- |
| 高被引基础 | TransAttUnet: Multi-Level Attention-Guided U-Net With Transformer for Medical Image Segmentation | 2023 | IEEE Transactions on Emerging Topics in Computational Intelligence | 引用量较高，适合作为理论背景或问题定义入口。 |
| 高被引基础 | Multi-Scale Self-Guided Attention for Medical Image Segmentation | 2020 | IEEE Journal of Biomedical and Health Informatics | 引用量较高，适合作为理论背景或问题定义入口。 |
| 高被引基础 | EMCAD: Efficient Multi-Scale Convolutional Attention Decoding for Medical Image Segmentation | 2024 | IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings | 引用量较高，适合作为理论背景或问题定义入口。 |
| 高被引基础 | MISSFormer: An Effective Transformer for 2D Medical Image Segmentation | 2022 | IEEE Transactions on Medical Imaging | 引用量较高，适合作为理论背景或问题定义入口。 |
| 高被引基础 | Customized Segment Anything Model for Medical Image Segmentation | 2023 | arXiv (Cornell University) | 引用量较高，适合作为理论背景或问题定义入口。 |
| 综述入口 | Medical Image Segmentation Review: The Success of U-Net | 2024 | IEEE Transactions on Pattern Analysis and Machine Intelligence | 覆盖面广，适合快速建立术语和分支地图。 |
| 代表工作 | MDA-Unet: A Multi-Scale Dilated Attention U-Net for Medical Image Segmentation | 2022 | Applied Sciences | 与主题词和核心技术路线高度相关，可作为分支阅读材料。 |
| 近期进展 | MSA$^2$Net: Multi-scale Adaptive Attention-guided Network for Medical Image Segmentation | 2024 | arXiv (Cornell University) | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 近期进展 | Edge Information-Guided 3D UNet for Medical Image Segmentation | 2026 | International Conference on Artificial Intelligence and Electromechanical Automation (AIEA) | 反映近年约束控制、鲁棒性或应用场景的新变化。 |
| 近期进展 | DMAGNet: Dual‐path multi‐scale attention guided network for medical image segmentation | 2023 | IET Image Processing | 反映近年约束控制、鲁棒性或应用场景的新变化。 |

## 研究脉络

- 2019-2022：约 9 条；代表工作为《Medical Transformer: Gated Axial-Attention for Medical Image Segmentation》，该阶段重点可从 高被引基础 切入。
- 2023-2026：约 24 条；代表工作为《Medical Image Segmentation Review: The Success of U-Net》，该阶段重点可从 综述入口 切入。

## 主要方法对比

| 方法族 | 典型关注点 | 优势 | 常见限制 |
| --- | --- | --- | --- |
| Potential/barrier/navigation functions | 保持已有边、避免距离约束破坏 | 直观，适合编队与避障 | 参数调节和局部极值问题明显 |
| Algebraic connectivity / Fiedler value | 直接优化或估计图连通度 | 能刻画全局连通裕度 | 分布式估计和可扩展性是难点 |
| Event-triggered connectivity control | 减少通信和控制更新 | 适合资源受限网络 | 需要证明无 Zeno 行为和鲁棒性 |
| Prescribed performance / constrained control | 把连通、碰撞、误差约束统一处理 | 约束表达清晰 | 对模型假设和扰动界较敏感 |
| MPC / optimization-based methods | 在预测窗口内优化连通和任务性能 | 可处理多目标权衡 | 计算量和实时性压力较高 |

## 重要论文阅读顺序

1. Medical Transformer: Gated Axial-Attention for Medical Image Segmentation (2021), Jeya Maria Jose Valanarasu, Poojan Oza, Ilker Hacihaliloglu, et al., Lecture notes in computer science; citations=1630, DOI/ID=10.1007/978-3-030-87193-2_4.
2. Medical Image Segmentation Review: The Success of U-Net (2024), Reza Azad, Ehsan Khodapanah Aghdam, Amelie Rauland, et al., IEEE Transactions on Pattern Analysis and Machine Intelligence; citations=1304, DOI/ID=10.1109/tpami.2024.3435571.
3. Medical image segmentation using deep learning: A survey (2022), Risheng Wang, Tao Lei, Ruixia Cui, et al., IET Image Processing; citations=1020, DOI/ID=10.1049/ipr2.12419.
4. MISSFormer: An Effective Transformer for 2D Medical Image Segmentation (2022), Xiaohong Huang, Zhifang Deng, Dandan Li, et al., IEEE Transactions on Medical Imaging; citations=974, DOI/ID=10.1109/tmi.2022.3230943.
5. Deep semantic segmentation of natural and medical images: a review (2020), Saeid Asgari Taghanaki, Kumar Abhishek, Joseph Cohen, et al., Artificial Intelligence Review; citations=935, DOI/ID=10.1007/s10462-020-09854-1.
6. EMCAD: Efficient Multi-Scale Convolutional Attention Decoding for Medical Image Segmentation (2024), Md Mostafijur Rahman, Mustafa Munir, Radu Marculescu, IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings; citations=615, DOI/ID=10.1109/cvpr52733.2024.01118.
7. Multi-Scale Self-Guided Attention for Medical Image Segmentation (2020), Ashish Sinha, José Dolz, IEEE Journal of Biomedical and Health Informatics; citations=596, DOI/ID=10.1109/jbhi.2020.2986926.
8. TransAttUnet: Multi-Level Attention-Guided U-Net With Transformer for Medical Image Segmentation (2023), Bingzhi Chen, Yishu Liu, Zheng Zhang, et al., IEEE Transactions on Emerging Topics in Computational Intelligence; citations=544, DOI/ID=10.1109/tetci.2023.3309626.
9. Customized Segment Anything Model for Medical Image Segmentation (2023), Kaidong Zhang, Dong Liu, arXiv (Cornell University); citations=454, DOI/ID=10.48550/arxiv.2304.13785.
10. MDA-Unet: A Multi-Scale Dilated Attention U-Net for Medical Image Segmentation (2022), Alyaa Amer, Tryphon Lambrou, Xujiong Ye, Applied Sciences; citations=63, DOI/ID=10.3390/app12073676.

## 局限与开放问题

- 从理论保证走向真实机器人/无人系统实验仍是关键缺口，尤其是有限感知、通信丢包和动态障碍同时存在时。
- 代数连通度类方法需要更可扩展的分布式估计与控制实现，避免全局谱信息成为隐含假设。
- 连通保持、碰撞避免、队形精度和能耗之间存在多目标冲突，需要更清晰的权衡机制。
- 事件触发和通信受限控制需要同时说明无 Zeno、鲁棒性和实际网络延迟下的性能。
- 面向攻击或故障的连通度控制还需要把安全检测、拓扑重构和控制律设计更紧密地结合。

## 网页补充建议

- `"medical image segmentation prediction guided decoder multi scale" PDF`
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
