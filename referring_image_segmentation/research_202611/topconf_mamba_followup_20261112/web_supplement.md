# 一手网页补充与人工筛选
核查日期2026-11-12。本文件与API元数据分开。以下链接用于确认正式出处、方法与作者代码，不能把有仓库链接等同于完整开源。
核心查询包括 SegMAN CVPR 2025；MambaIR ECCV 2024；Spatial-Mamba ICLR 2025 structure aware state fusion github；MICCAI 2025 Mamba 2D skin lesion segmentation github；site:papers.miccai.org/miccai-2026 Mamba skin；site:openaccess.thecvf.com/content/CVPR2026 medical image segmentation Mamba。

|来源|链接|用途|
|---|---|---|
|CENet正式论文|https://papers.miccai.org/miccai-2025/paper/2882_paper.pdf|已读方法段，DSEB/CFAM定义|
|CENet作者代码|https://github.com/xmindflow/cenet|实际模块及皮肤训练入口|
|MambaIR ECCV|https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2740_ECCV_2024_paper.php|正式出处、局部信息/通道动机|
|MambaIR作者代码|https://github.com/csguoh/MambaIR|CAB/VSSBlock/ASSM实际实现|
|MambaIRv2 CVPR|https://openaccess.thecvf.com/content/CVPR2025/html/Guo_MambaIRv2_Attentive_State_Space_Restoration_CVPR_2025_paper.html|正式出处，后续访问403但前次检索可读；以作者代码补充|
|SegMAN CVPR|https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_SegMAN_Omni-scale_Context_Modeling_with_State_Space_Models_and_Local_CVPR_2025_paper.pdf|encoder/decoder机制|
|SegMAN作者代码|https://github.com/yunxiangfu2001/SegMAN|浅层未使用、实际MMSCoPE和安装依赖|
|Spatial-Mamba ICLR|https://openreview.net/forum?id=iDe1mtxqK5|正式ICLR2025，不误标年份|
|Spatial-Mamba论文|https://proceedings.iclr.cc/paper_files/paper/2025/file/b7216f4a324864e1f592c18de4d83d10-Paper-Conference.pdf|状态融合方法|
|Spatial-Mamba代码|https://github.com/EdwardChasel/Spatial-Mamba|实际算子/缓存/参数接口|
|2DMamba CVPR|https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_2DMamba_Efficient_State_Space_Model_for_Image_Representation_with_Applications_CVPR_2025_paper.html|二维状态空间及WSI/ADE20K任务|
|2DMamba代码|https://github.com/AtlasAnalyticsLab/2DMamba|CMake CUDA依赖；部分原始README请求501已记录|
|Mamba-Adaptor CVPR|https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Mamba-Adaptor_State_Space_Model_Adaptor_for_Visual_Recognition_CVPR_2025_paper.html|适配器思路；源码未核实|
|MambaOut CVPR|https://openaccess.thecvf.com/content/CVPR2025/html/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.html|分类与密集任务结论边界|
|MambaOut代码|https://github.com/yuweihao/MambaOut|GatedCNNBlock|
|GeoSemba CVPR2026|https://openaccess.thecvf.com/content/CVPR2026/papers/Sun_GeoSemba_Reconstructing_State_Space_Model_for_Cross_Paradigm_Representation_in_CVPR_2026_paper.pdf|医学Mamba SSR/CAR|
|GeoSemba作者仓库|https://github.com/Mrliujunwen/GeoSemba|代码缺失状态：仅README与图片|
|HEDS-Net MICCAI2026|https://papers.miccai.org/miccai-2026/0452-Paper4167.html|正式出处、ISIC任务、三组件|
|HEDS-Net代码|https://github.com/Momentisbeused/HEDS-Net|模块存在而整网依赖不齐|
|STDDNet ICCV2025|https://openaccess.thecvf.com/content/ICCV2025/html/Chen_STDDNet_Harnessing_Mamba_for_Video_Polyp_Segmentation_via_Spatial-aligned_Temporal_ICCV_2025_paper.html|单图数据缺少视频信息，不优先|
|LoG-VMamba ACCV2024|https://openaccess.thecvf.com/content/ACCV2024/html/Dang_LoG-VMamba_Local-Global_Vision_Mamba_for_Medical_Image_Segmentation_ACCV_2024_paper.html|近邻方法，会议分类区分|
|CAMS WACV2025|https://openaccess.thecvf.com/content/WACV2025/html/Khan_CAMS_Convolution_and_Attention-Free_Mamba-Based_Cardiac_Image_Segmentation_WACV_2025_paper.html|心脏任务近邻|
|TIFCMamba MICCAI2025|https://papers.miccai.org/miccai-2025/0523-Paper0648.html|文本弱监督，当前任务额外信息条件不同|
|MHMamba CVPR Findings2026|https://openaccess.thecvf.com/content/CVPR2026F/html/Tao_MHMamba_Multi-Head_Mamba_for_3D_Brain_Tumor_Segmentation_CVPRF_2026_paper.html|不是CVPR主会，3D任务|

可靠性说明：只依赖官方会议页/论文和作者仓库进行技术结论；搜索中的第三方列表仅作发现线索，不作为性能事实依据。本轮没有逐篇复现，也没有独立核实作者所有训练权重或论文分数。部分页面403、arXiv特定版本404、HEDS PDF网页解析长度超限均不解读为论文不存在。
