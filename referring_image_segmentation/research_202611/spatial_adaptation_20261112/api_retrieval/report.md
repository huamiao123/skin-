# Scholarly Deep Research Retrieval Report

- Query: Mamba medical image segmentation local spatial enhancement
- Year range: 2024 to 2026
- Records after deduplication: 33
- Ranking: score = 0.40 * lexical_relevance + 0.20 * citation_score + 0.15 * recency_score + 0.10 * venue_source_confidence + 0.10 * open_access_score + 0.05 * metadata_completeness

## Ranked Papers

### 1. Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation

- Score: 0.8709
- Authors: Ziyang Wang, Jian-Qing Zheng, Yichi Zhang, Cui, Ge, et al.
- Year / Venue: 2024 / arXiv (Cornell University)
- DOI: 10.48550/arxiv.2402.05079
- Sources: openalex
- Citations: 350
- URL: https://doi.org/10.48550/arxiv.2402.05079
- Abstract: In recent advancements in medical image analysis, Convolutional Neural Networks (CNN) and Vision Transformers (ViT) have set significant benchmarks. While the former excels in capturing local features through its convolution operations, the latter achieves remarkable global context understanding by leveraging self-attention mechanisms. However, both architectures exhibit limitations in efficiently modeling long-range dependencies within medical images, which is a critical aspect for precise segmentation. Inspired by the Mamba architecture, known for its proficiency in handling long sequences and global contextual information with enhanced computational efficiency as a State Space Model (SSM)

### 2. FMTNet: A Fourier‐Mamba–Transformer Enhanced Network for Medical Image Segmentation

- Score: 0.8156
- Authors: Shaoqiang Wang, Guiling Shi, Yuanyuan Zhang, Sibo Qiao, et al.
- Year / Venue: 2026 / CAAI Transactions on Intelligence Technology
- DOI: 10.1049/cit2.70141
- Sources: openalex
- Citations: 2
- URL: https://doi.org/10.1049/cit2.70141
- Abstract: ABSTRACT Models based on U‐shaped networks have achieved widespread success in the field of medical image segmentation, but their performance is generally limited by structural bottlenecks in the network. At this stage, feature maps experience a sharp decline in spatial resolution due to continuous downsampling, resulting in significant loss of critical boundaries and structural details. Additionally, the local receptive fields of convolutions limit the effective modelling of global context. To address this core issue, we propose a novel enhanced segmentation network called FMTNet. FMTNet fundamentally enhances the expressive power of deep features by integrating an innovative composite enha

### 3. VM-UNet: Vision Mamba UNet for Medical Image Segmentation

- Score: 0.8036
- Authors: Jiacheng Ruan, Jincheng Li, Suncheng Xiang
- Year / Venue: 2025 / ACM Transactions on Multimedia Computing Communications and Applications
- DOI: 10.1145/3767748
- Sources: openalex
- Citations: 1323
- URL: https://doi.org/10.1145/3767748
- Abstract: In the realm of medical image segmentation, both CNN-based and Transformer-based models have been extensively explored. However, CNNs exhibit limitations in long-range modeling capabilities, whereas Transformers are hampered by their quadratic computational complexity. Recently, State Space Models (SSMs), exemplified by Mamba, have emerged as a promising approach. They not only excel in modeling long-range interactions but also maintain a linear computational complexity. In this paper, leveraging state space models, we propose a U-shape architecture model for medical image segmentation, named Vision Mamba UNet (VM-UNet). Specifically, the Visual State Space (VSS) block is introduced as the f

### 4. LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation

- Score: 0.8031
- Authors: Weibin Liao, Yinghao Zhu, Xinyuan Wang, Chengwei Pan, et al.
- Year / Venue: 2024 / arXiv (Cornell University)
- DOI: 10.48550/arxiv.2403.05246
- Sources: openalex
- Citations: 238
- URL: https://doi.org/10.48550/arxiv.2403.05246
- Abstract: UNet and its variants have been widely used in medical image segmentation. However, these models, especially those based on Transformer architectures, pose challenges due to their large number of parameters and computational loads, making them unsuitable for mobile health applications. Recently, State Space Models (SSMs), exemplified by Mamba, have emerged as competitive alternatives to CNN and Transformer architectures. Building upon this, we employ Mamba as a lightweight substitute for CNN and Transformer within UNet, aiming at tackling challenges stemming from computational resource limitations in real medical settings. To this end, we introduce the Lightweight Mamba UNet (LightM-UNet) th

### 5. U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation

- Score: 0.7924
- Authors: Jun Ma, Li, Feifei, Bo Wang
- Year / Venue: 2024 / arXiv (Cornell University)
- DOI: 10.48550/arxiv.2401.04722
- Sources: openalex
- Citations: 1267
- URL: https://doi.org/10.48550/arxiv.2401.04722
- Abstract: Convolutional Neural Networks (CNNs) and Transformers have been the most popular architectures for biomedical image segmentation, but both of them have limited ability to handle long-range dependencies because of inherent locality or computational complexity. To address this challenge, we introduce U-Mamba, a general-purpose network for biomedical image segmentation. Inspired by the State Space Sequence Models (SSMs), a new family of deep sequence models known for their strong capability in handling long sequences, we design a hybrid CNN-SSM block that integrates the local feature extraction power of convolutional layers with the abilities of SSMs for capturing the long-range dependency. Mor

### 6. DPM-UNet: A Mamba-Based Network with Dynamic Perception Feature Enhancement for Medical Image Segmentation

- Score: 0.7893
- Authors: Shangyu Xu, Xiaohang Liu, Hongsheng Lei, Bin Hui
- Year / Venue: 2025 / Sensors
- DOI: 10.3390/s25227053
- Sources: crossref
- Citations: 1
- URL: https://doi.org/10.3390/s25227053
- Abstract: In medical image segmentation, effective integration of global and local features is crucial. Current methods struggle to simultaneously model long-range dependencies and fine local details. Convolutional Neural Networks (CNNs) excel at extracting local features but are limited by their local receptive fields for capturing long-range dependencies. While global self-attention mechanisms (e.g., in Transformers) can capture long-range spatial relationships, their quadratic computational complexity incurs high costs for high-resolution medical images. To address these limitations, State Space Models (SSMs), which maintain linear complexity while effectively establishing long-range dependencies, 

### 7. Mamba-Sea: A Mamba-Based Framework With Global-to-Local Sequence Augmentation for Generalizable Medical Image Segmentation

- Score: 0.7766
- Authors: Zihan Cheng, Jintao Guo, Jian Zhang, Lei Qi, et al.
- Year / Venue: 2025 / IEEE Transactions on Medical Imaging
- DOI: 10.1109/tmi.2025.3564765
- Sources: openalex, arxiv
- Citations: 44
- URL: https://doi.org/10.1109/tmi.2025.3564765
- Abstract: To segment medical images with distribution shifts, domain generalization (DG) has emerged as a promising setting to train models on source domains that can generalize to unseen target domains. Existing DG methods are mainly based on CNN or ViT architectures. Recently, advanced state space models, represented by Mamba, have shown promising results in various supervised medical image segmentation. The success of Mamba is primarily owing to its ability to capture long-range dependencies while keeping linear complexity with input sequence length, making it a promising alternative to CNNs and ViTs. Inspired by the success, in the paper, we explore the potential of the Mamba architecture to addre

### 8. LKM-UNet: Large Kernel Vision Mamba UNet for Medical Image Segmentation

- Score: 0.728
- Authors: Jinhong Wang, Jintai Chen, Danny Z. Chen, Jian Wu
- Year / Venue: 2024 / Lecture notes in computer science
- DOI: 10.1007/978-3-031-72111-3_34
- Sources: openalex
- Citations: 161
- URL: https://doi.org/10.1007/978-3-031-72111-3_34

### 9. HLSM-UNet:Hybrid Local Spatial Mamba UNet for Medical Image Segmentation

- Score: 0.7263
- Authors: Xude Zhang, Xiaoping Wu, Changzhen Zhang, Yumin Tang, et al.
- Year / Venue: 2024 / 2024 International Conference on Neuromorphic Computing (ICNC)
- DOI: 10.1109/icnc64304.2024.10987734
- Sources: crossref
- Citations: 2
- URL: https://doi.org/10.1109/icnc64304.2024.10987734

### 10. PF-Mamba-UNet: An Anatomical-Prior-Guided Medical Image Segmentation Network Based on Mamba and Spatial-Channel Collaborative Attention Fusion

- Score: 0.7157
- Authors: Leyi Xiao, Kangshun Liao, Baoxian Zhou, Chaodong Fan
- Year / Venue: 2026 / Unknown
- DOI: 10.2139/ssrn.7553347
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.7553347
- Abstract: Medical image segmentation plays a pivotal role in clinical diagnosis assistance, lesion detection, and computer-aided medicine. However, factors such as complex organ boundary morphologies, multi-scale structural variations, and low contrast continue to pose challenges for accurate segmentation. To address this, this paper proposes an efficient segmentation network based on the fusion of anatomical prior and features—Anatomical-prior-guided Fusion Mamba-UNet (PF-Mamba-UNet). Firstly, the CR-VSS module is designed. Within the VSS module, a convolutional refinement branch is constructed to enhance local detail modeling. An integrated CSS2D module enables multi-directional scanning along horiz

### 11. HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation

- Score: 0.7137
- Authors: Jiashu Xu
- Year / Venue: 2024 / arXiv (Cornell University)
- DOI: 10.48550/arxiv.2405.05007
- Sources: openalex
- Citations: 74
- URL: https://doi.org/10.48550/arxiv.2405.05007
- Abstract: Automatic medical image segmentation technology has the potential to expedite pathological diagnoses, thereby enhancing the efficiency of patient care. However, medical images often have complex textures and structures, and the models often face the problem of reduced image resolution and information loss due to downsampling. To address this issue, we propose HC-Mamba, a new medical image segmentation model based on the modern state space model Mamba. Specifically, we introduce the technique of dilated convolution in the HC-Mamba model to capture a more extensive range of contextual information without increasing the computational cost by extending the perceptual field of the convolution ker

### 12. Vision Mamba and xLSTM-UNet for medical image segmentation

- Score: 0.6887
- Authors: Xin Zhong, Gehao Lu, Hao Li
- Year / Venue: 2025 / Scientific Reports
- DOI: 10.1038/s41598-025-88967-5
- Sources: crossref
- Citations: 32
- URL: https://doi.org/10.1038/s41598-025-88967-5

### 13. SFma-Unet: A Mamba-Based Spatial-Frequency Fusion Network for Medical Image Segmentation

- Score: 0.6626
- Authors: Zhanpeng Liu, Yuqiang Zhang, Bin Wang, Yang Yang, et al.
- Year / Venue: 2025 / IEEE International Conference on Acoustics Speech and Signal Processing
- DOI: 10.1109/icassp49660.2025.10889117
- Sources: openalex
- Citations: 4
- URL: https://doi.org/10.1109/icassp49660.2025.10889117
- Abstract: Recently, Mamba-based methods have gained popularity in medical image segmentation due to their ability to model long-range dependencies with linear computational complexity. However, current segmentation methods often face challenges such as low contrast, blurred boundaries, and unclear backgrounds in medical images. Considering that perceived objects and features exhibit greater discriminative power in the frequency domain, we propose a novel mamba-based spatial and frequency domain feature fusion network, SFMa-Unet, to address these challenges. Specifically, we designed the Spatial-Frequency Interaction (SFI) module, which leverages the powerful modeling capabilities of Mamba to fuse spat

### 14. SFMamba3D: A Spatial-Frequency Dual-Branch Mamba Network for 3D Medical Image Segmentation

- Score: 0.6586
- Authors: Chenyu Li, Xingchun Ru
- Year / Venue: 2026 / 2026 Asia Conference on Computer Vision, Image Processing and Pattern Recognition (CVIPPR)
- DOI: 10.1109/cvippr69607.2026.11604503
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/cvippr69607.2026.11604503

### 15. GL-Mamba UNet: Global-to-Local Context-Guided Vision Mamba for Medical Image Segmentation

- Score: 0.6586
- Authors: Longhui Wang, Jie Li, Yue Ruan
- Year / Venue: 2026 / Proceedings of the 2026 6th International Conference on Bioinformatics and Intelligent Computing
- DOI: 10.1145/3809986.3810131
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1145/3809986.3810131

### 16. ASRM-Net: Multi-Scale Feature Enhancement with Adaptive Spatial Recalibration for Medical Image Segmentation

- Score: 0.6586
- Authors: Xinyang Chen
- Year / Venue: 2026 / 2026 7th International Conference on Computer Vision, Image and Deep Learning (CVIDL)
- DOI: 10.1109/cvidl70130.2026.11637809
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/cvidl70130.2026.11637809

### 17. SICP-Mamba: Support-Information-Conditioned Progressive State-Space Modeling for Few-Shot Medical Image Segmentation

- Score: 0.6586
- Authors: Wenxi Wang, Wenhui Huang
- Year / Venue: 2026 / Unknown
- DOI: 10.2139/ssrn.7227878
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.7227878
- Abstract: Few-shot medical image segmentation aims to delineate target organs or lesions from query images using only a few annotated support samples, which is important for reducing annotation burden in biomedical image analysis. However, reliable segmentation under limited supervision remains challenging because medical images often exhibit complex anatomical structures, large inter-patient variations, low contrast, ambiguous boundaries, and substantial intra-class morphological differences. Existing methods usually establish support--query correlations through prototype matching, feature fusion, or cross-attention. These strategies may focus on local similarities or pairwise token relations, leadin

### 18. Gl-MambaNet: A global-local hybrid Mamba network for medical image segmentation

- Score: 0.65
- Authors: Xiaoyan Kui, Shen Jiang, Qinsong Li, Yifei Peng, et al.
- Year / Venue: 2025 / Neurocomputing
- DOI: 10.1016/j.neucom.2025.129580
- Sources: openalex
- Citations: 31
- URL: https://doi.org/10.1016/j.neucom.2025.129580

### 19. PolyMamba: Spatial-Prior Guided Mamba for Polyp Segmentation with High-Frequency Enhancement

- Score: 0.6462
- Authors: Renyu Fu, Shurui Hu, Xiao Zheng, Chang Tang, et al.
- Year / Venue: 2026 / Lecture Notes in Computer Science
- DOI: 10.1007/978-3-032-05141-7_44
- Sources: crossref
- Citations: 4
- URL: https://doi.org/10.1007/978-3-032-05141-7_44

### 20. A diffusion model based on multi-scale spatial Mamba for medical image segmentation

- Score: 0.6307
- Authors: Chun Li, Qiule Sun, Muqing Zhang, Jianxin Zhang
- Year / Venue: 2025 / Engineering Applications of Artificial Intelligence
- DOI: 10.1016/j.engappai.2025.111028
- Sources: openalex
- Citations: 15
- URL: https://doi.org/10.1016/j.engappai.2025.111028

### 21. A Local-Global Fusion Vision Mamba UNet Framework for medical image segmentation

- Score: 0.6177
- Authors: Yanbo Li, Zihan Mao, Feiwei Qin, Yong Kang Peng, et al.
- Year / Venue: 2026 / Engineering Applications of Artificial Intelligence
- DOI: 10.1016/j.engappai.2026.113987
- Sources: openalex
- Citations: 6
- URL: https://doi.org/10.1016/j.engappai.2026.113987

### 22. SCM-UNet: Spatial-channel Mamba UNet for medical image segmentation

- Score: 0.6147
- Authors: Haijie Yan, Qiuhong Hong, Shoulin Wei, Xiangliang Zhang, et al.
- Year / Venue: 2025 / Digital Signal Processing
- DOI: 10.1016/j.dsp.2025.105550
- Sources: openalex
- Citations: 8
- URL: https://doi.org/10.1016/j.dsp.2025.105550

### 23. Rethinking Boundary Detection in Deep Learning-Based Medical Image Segmentation

- Score: 0.5864
- Authors: Yi Lin, Dong Zhang, Xiao Fang, Yufan Chen, et al.
- Year / Venue: 2025 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2505.04652v1
- Abstract: Medical image segmentation is a pivotal task within the realms of medical image analysis and computer vision. While current methods have shown promise in accurately segmenting major regions of interest, the precise segmentation of boundary areas remains challenging. In this study, we propose a novel network architecture named CTO, which combines Convolutional Neural Networks (CNNs), Vision Transformer (ViT) models, and explicit edge detection operators to tackle this challenge. CTO surpasses existing methods in terms of segmentation accuracy and strikes a better balance between accuracy and efficiency, without the need for additional data inputs or label injections. Specifically, CTO adheres

### 24. MambaMIM: Pre-training Mamba with State Space Token Interpolation and its Application to Medical Image Segmentation

- Score: 0.5836
- Authors: Fenghe Tang, Bingkun Nian, Yingtai Li, Zihang Jiang, et al.
- Year / Venue: 2024 / arXiv
- DOI: 10.1016/j.media.2025.103606
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2408.08070v2
- Abstract: Recently, the state space model Mamba has demonstrated efficient long-sequence modeling capabilities, particularly for addressing long-sequence visual tasks in 3D medical imaging. However, existing generative self-supervised learning methods have not yet fully unleashed Mamba's potential for handling long-range dependencies because they overlook the inherent causal properties of state space sequences in masked modeling. To address this challenge, we propose a general-purpose pre-training framework called MambaMIM, a masked image modeling method based on a novel TOKen-Interpolation strategy (TOKI) for the selective structure state space sequence, which learns causal relationships of state spa

### 25. MAFUNet: Mamba with adaptive fusion UNet for medical image segmentation

- Score: 0.5605
- Authors: Minchen Yang, Ziyi Yang, Nur Intan Raihana Ruhaiyem
- Year / Venue: 2025 / Image and Vision Computing
- DOI: 10.1016/j.imavis.2025.105655
- Sources: openalex
- Citations: 9
- URL: https://doi.org/10.1016/j.imavis.2025.105655

### 26. Synergistic Mamba: Mastering Global Frequency and Local Spatial Contexts for Low-light image Enhancement

- Score: 0.5414
- Authors: Shijun Fu, Pingping Liu, Tongshun Zhang, hongwei zhao, et al.
- Year / Venue: 2025 / Unknown
- DOI: 10.2139/ssrn.5736831
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.5736831

### 27. Fréchet Radiomic Distance (FRD): A Versatile Metric for Comparing Medical Imaging Datasets

- Score: 0.5264
- Authors: Nicholas Konz, Richard Osuala, Preeti Verma, Yuwen Chen, et al.
- Year / Venue: 2024 / arXiv
- DOI: 10.1016/j.media.2026.103943
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2412.01496v2
- Abstract: Determining whether two sets of images belong to the same or different distributions or domains is a crucial task in modern medical image analysis and deep learning; for example, to evaluate the output quality of image generative models. Currently, metrics used for this task either rely on the (potentially biased) choice of some downstream task, such as segmentation, or adopt task-independent perceptual metrics (e.g., Fréchet Inception Distance/FID) from natural imaging, which we show insufficiently capture anatomical features. To this end, we introduce a new perceptual metric tailored for medical images, FRD (Fréchet Radiomic Distance), which utilizes standardized, clinically meaningful, an

### 28. Test-time generative augmentation for medical image segmentation

- Score: 0.5264
- Authors: Xiao Ma, Yuhui Tao, Zetian Zhang, Yuhan Zhang, et al.
- Year / Venue: 2024 / arXiv
- DOI: 10.1016/j.media.2025.103902
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2406.17608v2
- Abstract: Medical image segmentation is critical for clinical diagnosis, treatment planning, and monitoring, yet segmentation models often struggle with uncertainties stemming from occlusions, ambiguous boundaries, and variations in imaging devices. Traditional test-time augmentation (TTA) techniques typically rely on predefined geometric and photometric transformations, limiting their adaptability and effectiveness in complex medical scenarios. In this study, we introduced Test-Time Generative Augmentation (TTGA), a novel augmentation strategy specifically tailored for medical image segmentation at inference time. Different from conventional augmentation strategies that suffer from excessive randomne

### 29. A Mamba-based channel attention-enhanced model for medical image segmentation

- Score: 0.5064
- Authors: Ying Liu, Jiamin Li, Yating Zhao, Zhigang Liu, et al.
- Year / Venue: 2026 / Signal Image and Video Processing
- DOI: 10.1007/s11760-026-05555-4
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.1007/s11760-026-05555-4

### 30. Large-Kernel Spatial-Frequency Fusion Deep Mamba Network for Remote Sensing Image Segmentation

- Score: 0.5014
- Authors: ChenZhe Xia, Eziz Tursun
- Year / Venue: 2026 / Unknown
- DOI: 10.2139/ssrn.6190562
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.6190562
- Abstract: Semantic segmentation of remote sensing images is a fundamental task in geospatial informationanalysis. Unlike natural images, the complex background in high-resolution remote sensing imagesposes significant challenges for object segmentation, such as occlusion and intra-class variations, oftenleading to fragmented and mis-segmented results. To address these challenges, we proposed LargeKernel Spatial-Frequency Fusion Deep Mamba Network (Mamba-LSFDNet), a remote sensingsemantic segmentation network that combines large-kernel attention with the state-space modelMamba. Specifically, we first designed a Large-Kernel Spatial-Frequency Hybrid Module (LSFModule) , by expanding the receptive field 

### 31. DEVA Mamba: dual encoder visual attention mamba for medical image segmentation

- Score: 0.5014
- Authors: Jingwen Zhang, Dongxu Cheng, Qiwei Dong
- Year / Venue: 2026 / International Conference on Computational Intelligence and Image Analysis (ICCIIA 2026)
- DOI: 10.1117/12.3113089
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1117/12.3113089

### 32. Hcmunet: A Hybrid Cnn and Mamba Network for Medical Ultrasound Image Segmentation

- Score: 0.4743
- Authors: Chong Zhang, Guohui Wei, Min Qiu
- Year / Venue: 2024 / Unknown
- DOI: 10.2139/ssrn.4920609
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.4920609

### 33. Selective and Multi-Scale Fusion Mamba for Medical Image Segmentation

- Score: 0.4743
- Authors: Guangju Li, Qinghua Huang, Wei Wang, Longzhong Liu
- Year / Venue: 2024 / Unknown
- DOI: 10.2139/ssrn.4895634
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.4895634

## Coverage Note

This report supports screening and does not guarantee exhaustive coverage. Final scholarly use requires human review.
