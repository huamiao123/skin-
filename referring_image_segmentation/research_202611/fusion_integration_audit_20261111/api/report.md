# Scholarly Deep Research Retrieval Report

- Query: medical image segmentation skip connections feature fusion
- Year range: 2015 to 2026
- Records after deduplication: 55
- Ranking: score = 0.40 * lexical_relevance + 0.20 * citation_score + 0.15 * recency_score + 0.10 * venue_source_confidence + 0.10 * open_access_score + 0.05 * metadata_completeness

## Ranked Papers

### 1. UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation

- Score: 0.925
- Authors: Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang
- Year / Venue: 2019 / IEEE Transactions on Medical Imaging
- DOI: 10.1109/tmi.2019.2959609
- Sources: openalex, arxiv
- Citations: 4666
- URL: https://doi.org/10.1109/tmi.2019.2959609
- Abstract: The state-of-the-art models for medical image segmentation are variants of U-Net and fully convolutional networks (FCN). Despite their success, these models have two limitations: (1) their optimal depth is apriori unknown, requiring extensive architecture search or inefficient ensemble of models of varying depths; and (2) their skip connections impose an unnecessarily restrictive fusion scheme, forcing aggregation only at the same-scale feature maps of the encoder and decoder sub-networks. To overcome these two limitations, we propose UNet++, a new neural architecture for semantic and instance segmentation, by (1) alleviating the unknown network depth with an efficient ensemble of U-Nets of 

### 2. UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-Wise Perspective with Transformer

- Score: 0.9138
- Authors: Haonan Wang, Peng Cao, Jiaqi Wang, Osmar R. Zai͏̈ane
- Year / Venue: 2022 / Proceedings of the AAAI Conference on Artificial Intelligence
- DOI: 10.1609/aaai.v36i3.20144
- Sources: openalex
- Citations: 1248
- URL: https://doi.org/10.1609/aaai.v36i3.20144
- Abstract: Most recent semantic segmentation methods adopt a U-Net framework with an encoder-decoder architecture. It is still challenging for U-Net with a simple skip connection scheme to model the global multi-scale context: 1) Not each skip connection setting is effective due to the issue of incompatible feature sets of encoder and decoder stage, even some skip connection negatively influence the segmentation performance; 2) The original U-Net is worse than the one without any skip connection on some datasets. Based on our findings, we propose a new segmentation framework, named UCTransNet (with a proposed CTrans module in U-Net), from the channel perspective with attention mechanism. Specifically, 

### 3. CMUNEXT: An Efficient Medical Image Segmentation Network Based on Large Kernel and Skip Fusion

- Score: 0.8918
- Authors: Fenghe Tang, Jianrui Ding, Quan Quan, Lingtao Wang, et al.
- Year / Venue: 2024 / IEEE International Symposium on Biomedical Imaging (ISBI)
- DOI: 10.1109/isbi56570.2024.10635609
- Sources: openalex
- Citations: 211
- URL: https://doi.org/10.1109/isbi56570.2024.10635609
- Abstract: The u-shaped architecture has emerged as a crucial paradigm in the design of medical image segmentation networks. However, due to the inherent local limitations of convolution, a fully convolutional segmentation network with u-shaped architecture struggles to effectively extract global context information, which is vital for the precise localization of lesions. While hybrid architectures combining CNN and Transformer can address these issues, their applications are limited due to the computational resource. In addition, the inductive bias of convolution in lightweight networks adeptly fits the scarce medical data, which is lacking in the Transformer based network. To extract global context i

### 4. SMAFormer: Synergistic Multi-Attention Transformer for Medical Image Segmentation

- Score: 0.8681
- Authors: Fuchen Zheng, Xuhang Chen, Weihuang Liu, Haolun Li, et al.
- Year / Venue: 2024 / IEEE International Conference on Bioinformatics and Biomedicine (BIBM)
- DOI: 10.1109/bibm62325.2024.10822736
- Sources: openalex
- Citations: 77
- URL: https://doi.org/10.1109/bibm62325.2024.10822736
- Abstract: In medical image segmentation, specialized computer vision techniques, notably transformers grounded in attention mechanisms and residual networks employing skip connections, have been instrumental in advancing performance. Nonetheless, previous models often falter when segmenting small, irregularly shaped tumors. To this end, we introduce SMAFormer, an efficient, Transformer-based architecture that fuses multiple attention mechanisms for enhanced segmentation of small tumors and organs. SMAFormer can capture both local and global features for medical image segmentation. The architecture comprises two pivotal components. First, a Synergistic Multi-Attention (SMA) Transformer block is propose

### 5. U-Net V2: Rethinking the Skip Connections of U-Net for Medical Image Segmentation

- Score: 0.8588
- Authors: Yaopeng Peng, Danny Z. Chen, Milan Sonka A
- Year / Venue: 2025 / IEEE International Symposium on Biomedical Imaging (ISBI)
- DOI: 10.1109/isbi60581.2025.10980742
- Sources: openalex, crossref
- Citations: 251
- URL: https://doi.org/10.1109/isbi60581.2025.10980742
- Abstract: In this paper, we introduce U-Net v2, a new robust and efficient U-Net variant for medical image segmentation. It aims to augment the infusion of semantic information into low-level features while simultaneously refining high-level features with finer details. For an input image, we begin by extracting multilevel features with a deep neural network encoder. Next, we enhance the feature map of each level by infusing semantic information from higher-level features and integrating finer details from lower-level features through Hadamard product. Our novel skip connections empower features of all the levels with enriched semantic characteristics and intricate details. The improved features are s

### 6. HiFormer: Hierarchical Multi-scale Representations Using Transformers for Medical Image Segmentation

- Score: 0.8496
- Authors: Moein Heidari, Amirhossein Kazerouni, Milad Soltany, Reza Azad, et al.
- Year / Venue: 2023 / IEEE Winter Conference on Applications of Computer Vision
- DOI: 10.1109/wacv56688.2023.00614
- Sources: openalex
- Citations: 608
- URL: https://doi.org/10.1109/wacv56688.2023.00614
- Abstract: Convolutional neural networks (CNNs) have been the consensus for medical image segmentation tasks. However, they suffer from the limitation in modeling long-range dependencies and spatial correlations due to the nature of convolution operation. Although transformers were first developed to address this issue, they fail to capture low-level features. In contrast, it is demonstrated that both local and global features are crucial for dense prediction, such as segmenting in challenging contexts. In this paper, we propose HiFormer, a novel method that efficiently bridges a CNN and a transformer for medical image segmentation. Specifically, we design two multi-scale feature representations using 

### 7. Hybrid attention mechanism of feature fusion for medical image segmentation

- Score: 0.8259
- Authors: Shanshan Tong, Zhentao Zuo, Zuxiang Liu, Dengdi Sun, et al.
- Year / Venue: 2023 / IET Image Processing
- DOI: 10.1049/ipr2.12934
- Sources: openalex
- Citations: 19
- URL: https://doi.org/10.1049/ipr2.12934
- Abstract: Abstract Traditional convolution neural networks (CNN) have achieved good performance in multi‐organ segmentation of medical images. Due to the lack of ability to model long‐range dependencies and correlations between image pixels, CNN usually ignores the information of channel dimension. To further improve the performance of multi‐organ segmentation, a hybrid attention mechanism model is proposed. First, a CNN was used to extract multi‐scale feature maps and fed into the Channel Attention Enhancement Module (CAEM) to selectively pay attention to target organs in medical images, and the Transformer encoded tokenized image patches from CNN feature maps as the input sequence to model long‐rang

### 8. MAFUNet: Multi-Attention Fusion Network for Medical Image Segmentation

- Score: 0.7878
- Authors: Lili Wang, Jiayu Zhao, Hailu Yang
- Year / Venue: 2023 / IEEE Access
- DOI: 10.1109/access.2023.3320685
- Sources: openalex
- Citations: 3
- URL: https://doi.org/10.1109/access.2023.3320685
- Abstract: The purpose of medical image segmentation is to identify target organs, tissues or lesion areas at the pixel level to help doctors evaluate and prevent diseases. Therefore, the model is required to have higher accuracy and robust representation. At present, the proposed models focus on the improvement of performance, and ignore the number of trainable parameters. This paper proposes a lightweight ECA-Residual module to build a model encoder, which can effectively extract features while reducing the number of parameters. The feature fusion of encoder and decoder using simple skip connections will produce semantic differences. Therefore, in this paper a spatial attention gating module is desig

### 9. ES-UNet: efficient 3D medical image segmentation with enhanced skip connections in 3D UNet

- Score: 0.7871
- Authors: Minyoung Park, Seungtaek Oh, Junyoung Park, Taikyeong Jeong, et al.
- Year / Venue: 2025 / BMC Medical Imaging
- DOI: 10.1186/s12880-025-01857-0
- Sources: crossref
- Citations: 22
- URL: https://doi.org/10.1186/s12880-025-01857-0
- Abstract: Abstract Background Deep learning has significantly advanced medical image analysis, particularly in semantic segmentation, which is essential for clinical decisions. However, existing 3D segmentation models, like the traditional 3D UNet, face challenges in balancing computational efficiency and accuracy when processing volumetric medical data. This study aims to develop an improved architecture for 3D medical image segmentation with enhanced learning strategies to improve accuracy and address challenges related to limited training data. Methods We propose ES-UNet, a 3D segmentation architecture that achieves superior segmentation performance while offering competitive efficiency across mult

### 10. FFUNet: A novel feature fusion makes strong decoder for medical image segmentation

- Score: 0.7631
- Authors: Junsong Xie, Renju Zhu, Zezhi Wu, Jinling Ouyang
- Year / Venue: 2022 / IET Signal Processing
- DOI: 10.1049/sil2.12114
- Sources: openalex
- Citations: 23
- URL: https://doi.org/10.1049/sil2.12114
- Abstract: Abstract Convolutional neural networks (CNNs) have strong ability to extract local features, but it is slightly lacking in extracting global contexts. In contrast, transformers are good at long‐distance modelling due to the global self‐attention mechanisms while its performance in localization is limited. On the other hand, the feature gap between an encoder and decoder is also challenging for a U‐shaped network, which adopts a plain skip connection. Inherited from convolutional networks and transformers, FFUNet, a hybrid network structure with a novel module named the Feature Fusion Module (FFM) is proposed for medical image segmentation. The proposed FFM consisting of Feature Attention Sel

### 11. Global and Local Feature Reconstruction for Medical Image Segmentation

- Score: 0.7523
- Authors: Jiahuan Song, Xinjian Chen, Qianlong Zhu, Fei Shi, et al.
- Year / Venue: 2022 / IEEE Transactions on Medical Imaging
- DOI: 10.1109/tmi.2022.3162111
- Sources: openalex
- Citations: 92
- URL: https://doi.org/10.1109/tmi.2022.3162111
- Abstract: Learning how to capture long-range dependencies and restore spatial information of down-sampled feature maps are the basis of the encoder-decoder structure networks in medical image segmentation. U-Net based methods use feature fusion to alleviate these two problems, but the global feature extraction ability and spatial information recovery ability of U-Net are still insufficient. In this paper, we propose a Global Feature Reconstruction (GFR) module to efficiently capture global context features and a Local Feature Reconstruction (LFR) module to dynamically up-sample features, respectively. For the GFR module, we first extract the global features with category representation from the featur

### 12. A Multi‐Scale Feature Interaction and Fusion Medical Image Segmentation Method

- Score: 0.7131
- Authors: Yanjin Wang, Hualing Li, Gaizhen Liu, Jiaxin Huo, et al.
- Year / Venue: 2025 / International Journal of Imaging Systems and Technology
- DOI: 10.1002/ima.70207
- Sources: openalex
- Citations: 4
- URL: https://doi.org/10.1002/ima.70207
- Abstract: ABSTRACT To address the challenges of edge loss and low segmentation accuracy in small regions in medical image segmentation, this study proposes a novel segmentation network, MSFIF‐Net, which integrates the convolutional neural networks (CNNs) and transformer. Built upon the TransUNet architecture, our approach introduces two novel modules: the multi‐group contextual attention (MDGA) module and the multi‐scale dilated aggregation (MSDAM) module. The MDGA module enhances feature extraction across different dimensions by facilitating the interaction and fusion of multiple contextual information groups. Meanwhile, the MSDAM module optimizes feature fusion in skip connections by integrating mul

### 13. An effective UNet using feature interaction and fusion for organ segmentation in medical image

- Score: 0.7017
- Authors: Xiaolin Gou, Chuanlin Liao, Jizhe Zhou, Fengshuo Ye, et al.
- Year / Venue: 2026 / Engineering Applications of Artificial Intelligence
- DOI: 10.1016/j.engappai.2026.113727
- Sources: openalex
- Citations: 4
- URL: https://doi.org/10.1016/j.engappai.2026.113727

### 14. MAGE-UNet: Progressive skip-feature refinement for medical image segmentation

- Score: 0.6967
- Authors: Yuhang Wang, Shuoyang Wang, Chengze Lv, Gongrui Wang, et al.
- Year / Venue: 2026 / Signal, Image and Video Processing
- DOI: 10.1007/s11760-026-05357-8
- Sources: crossref
- Citations: 4
- URL: https://doi.org/10.1007/s11760-026-05357-8

### 15. UNestFormer: Enhancing Decoders and Skip Connections with Nested Transformers for Medical Image Segmentation

- Score: 0.6878
- Authors: Adnan Md Tayeb, Tae-Hyong Kim
- Year / Venue: 2024 / IEEE Access
- DOI: 10.1109/access.2024.3516079
- Sources: crossref
- Citations: 7
- URL: https://doi.org/10.1109/access.2024.3516079

### 16. DAC-SaNet: improved skip connections in U-Net for medical image segmentation

- Score: 0.685
- Authors: Zili Lei, Lin Chai, YuCheng Zhang, XinYi Bao
- Year / Venue: 2025 / IET conference proceedings.
- DOI: 10.1049/icp.2025.0032
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.1049/icp.2025.0032
- Abstract: Medical image segmentation plays a pivotal role in various domains, including biomedical research, clinical diagnosis, and pathological analysis. Nevertheless, the prevalent U-shaped network architecture in current medical image segmentation suffers from limitations such as inadequate fusion of hierarchical feature information through simplistic concatenation and the ineﬀiciency of all skip connections. To address these issues, we propose a lightweight DAC-SaNet framework, designed to mitigate the unequal contribution of skip connections, semantic gap, and connectivity constraints in traditional U-nets. Specifically, our framework incorporates a Spatial Channel Attention Fusion Module with D

### 17. Rethinking U-Net: Task-Adaptive Mixture of Skip Connections for Enhanced Medical Image Segmentation

- Score: 0.6717
- Authors: Zichen Luo, Xinshan Zhu, Lan Zhang, Biao Sun
- Year / Venue: 2025 / Proceedings of the AAAI Conference on Artificial Intelligence
- DOI: 10.1609/aaai.v39i6.32627
- Sources: crossref
- Citations: 11
- URL: https://doi.org/10.1609/aaai.v39i6.32627
- Abstract: U-Net is a widely used model for medical image segmentation, renowned for its strong feature extraction capabilities and U-shaped design, which incorporates skip connections to preserve critical information. However, its decoders exhibit information-specific preferences for the supplementary content provided by skip connections, instead of adhering to a strict one-to-one correspondence, which limits its flexibility across diverse tasks. To address this limitation, we propose the Task-Adaptive Mixture of Skip Connections (TA-MoSC) module, inspired by the Mixture of Experts (MoE) framework. TA-MoSC innovatively reinterprets skip connections as a task allocation problem, employing a routing mec

### 18. SAttisUNet: UNet-like Swin Transformer with Attentive Skip Connections for Enhanced Medical Image Segmentation

- Score: 0.6614
- Authors: Maryam Tavakol Elahi, WonSook Lee, Philippe Phan
- Year / Venue: 2023 / 2023 International Conference on Machine Learning and Applications (ICMLA)
- DOI: 10.1109/icmla58977.2023.00301
- Sources: crossref
- Citations: 3
- URL: https://doi.org/10.1109/icmla58977.2023.00301

### 19. KGS-UNet: KAN-Gated Skip Connections for Small-Structure Medical Image Segmentation with Limited Data

- Score: 0.6586
- Authors: Chaolin Wu, Shun Long
- Year / Venue: 2026 / 2026 International Joint Conference on Neural Networks (IJCNN)
- DOI: 10.1109/ijcnn69854.2026.11728422
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/ijcnn69854.2026.11728422

### 20. Dual Enhanced Swin-UNet with Adaptive Attention and Refined Skip Connections for Medical Image Segmentation

- Score: 0.6586
- Authors: Wang Haoyu, Lan Gang
- Year / Venue: 2026 / 2026 IEEE 2nd International Conference on Electronics, Energy Systems and Power Engineering (EESPE)
- DOI: 10.1109/eespe68405.2026.11648225
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/eespe68405.2026.11648225

### 21. Dynamic feature fusion and weighted consistency learning for semi-supervised medical image segmentation

- Score: 0.6586
- Authors: Longwei Wang
- Year / Venue: 2026 / 2026 International Joint Conference on Neural Networks (IJCNN)
- DOI: 10.1109/ijcnn69854.2026.11729256
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/ijcnn69854.2026.11729256

### 22. A New Multilevel Feature Fusion Network for Medical Image Segmentation

- Score: 0.651
- Authors: Xiaojing Qiu
- Year / Venue: 2021 / Sensing and Imaging
- DOI: 10.1007/s11220-021-00346-2
- Sources: crossref
- Citations: 5
- URL: https://doi.org/10.1007/s11220-021-00346-2

### 23. Dual-Branch UNet: Hybrid Transformer-CNN Architecture with Enhanced Skip Connections for Medical Image Segmentation

- Score: 0.6486
- Authors: Junren Shao
- Year / Venue: 2025 / 2025 IEEE 7th International Conference on Power, Intelligent Computing and Systems (ICPICS)
- DOI: 10.1109/icpics66386.2025.11347130
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/icpics66386.2025.11347130

### 24. Efficient Boundary-Aware Multi-Scale Feature Fusion for Medical Image Segmentation

- Score: 0.6486
- Authors: Youssef Ahmedm, Ruotong Luan
- Year / Venue: 2025 / Unknown
- DOI: 10.20944/preprints202511.0918.v1
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.20944/preprints202511.0918.v1
- Abstract: Accurate medical image segmentation plays a crucial role in clinical diagnosis and treatment planning. Yet, existing U-Net variants often fail to effectively capture both fine anatomical details and global contextual relationships while maintaining efficiency. To overcome these challenges, we propose EB-MSF-UNet, an Efficient Boundary-aware Multi-Scale Feature Fusion U-Net that integrates global reasoning with local detail preservation. The framework employs a dual-pathway encoder combining lightweight convolutional layers for local feature extraction and Mamba-inspired modules for longrange dependency modeling, regulated by an adaptive feature gating mechanism. A cross-attentionbased multi-

### 25. Enhancing feature fusion of U-like networks with dynamic skip connections

- Score: 0.6438
- Authors: Yue Cao, Quansong He, Kaishen Wang, Jianlong Xiong, et al.
- Year / Venue: 2026 / Medical Image Analysis
- DOI: 10.1016/j.media.2026.104010
- Sources: crossref
- Citations: 5
- URL: https://doi.org/10.1016/j.media.2026.104010

### 26. Deep Feature-Level Sensor Fusion Using Skip Connections for Real-Time Object Detection in Autonomous Driving

- Score: 0.6399
- Authors: Vijay John, Seiichi Mita
- Year / Venue: 2021 / Electronics
- DOI: 10.3390/electronics10040424
- Sources: crossref
- Citations: 30
- URL: https://doi.org/10.3390/electronics10040424
- Abstract: Object detection is an important perception task in autonomous driving and advanced driver assistance systems. The visible camera is widely used for perception, but its performance is limited by illumination and environmental variations. For robust vision-based perception, we propose a deep learning framework for effective sensor fusion of the visible camera with complementary sensors. A feature-level sensor fusion technique, using skip connection, is proposed for the sensor fusion of the visible camera with the millimeter-wave radar and the thermal camera. The two networks are called the RV-Net and the TV-Net, respectively. These networks have two input branches and one output branch. The i

### 27. SK-VM++: Mamba assists skip-connections for medical image segmentation

- Score: 0.6384
- Authors: Renkai Wu, Liuyue Pan, Pengchen Liang, Qing Chang, et al.
- Year / Venue: 2025 / Biomedical Signal Processing and Control
- DOI: 10.1016/j.bspc.2025.107646
- Sources: openalex
- Citations: 35
- URL: https://doi.org/10.1016/j.bspc.2025.107646

### 28. Multi-scale feature pyramid fusion network for medical image segmentation

- Score: 0.6176
- Authors: Bing Zhang, Yang Wang, Caifu Ding, Ziqing Deng, et al.
- Year / Venue: 2022 / International Journal of Computer Assisted Radiology and Surgery
- DOI: 10.1007/s11548-022-02738-5
- Sources: openalex
- Citations: 52
- URL: https://doi.org/10.1007/s11548-022-02738-5

### 29. MF2-Net: A multipath feature fusion network for medical image segmentation

- Score: 0.6056
- Authors: Nagaraj Yamanakkanavar, Bumshik Lee
- Year / Venue: 2022 / Engineering Applications of Artificial Intelligence
- DOI: 10.1016/j.engappai.2022.105004
- Sources: openalex
- Citations: 31
- URL: https://doi.org/10.1016/j.engappai.2022.105004

### 30. EFFM: Rethinking feature fusion mechanisms for medical image segmentation

- Score: 0.6028
- Authors: Xiaoyan Zhang, Zheng Zhao, Yongqin Zhang, Chunlin Yu, et al.
- Year / Venue: 2025 / Biomedical Signal Processing and Control
- DOI: 10.1016/j.bspc.2025.109334
- Sources: openalex
- Citations: 7
- URL: https://doi.org/10.1016/j.bspc.2025.109334

### 31. Tree-NET: Enhancing 2D Medical Image Segmentation Through Efficient Low-Level Feature Training

- Score: 0.5936
- Authors: Orhan Demirci, Bulent Yilmaz
- Year / Venue: 2025 / arXiv
- DOI: 10.1007/s00521-026-12420-5
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2501.02140v2
- Abstract: This paper introduces Tree-NET, a novel framework for medical image segmentation that leverages bottleneck supervision to enhance both segmentation accuracy and computational efficiency. While previous studies have applied bottleneck feature supervision to segmentation tasks, it has typically been limited to the training phase, offering no computational benefits during inference. To the best of our knowledge, this is the first framework to employ dual bottleneck supervision for segmentation, leveraging latent space features at both the input and output stages. This approach reduces input and label dimensions with minimal parameter overhead while preserving accuracy. Tree-NET features a three

### 32. Fréchet Radiomic Distance (FRD): A Versatile Metric for Comparing Medical Imaging Datasets

- Score: 0.5836
- Authors: Nicholas Konz, Richard Osuala, Preeti Verma, Yuwen Chen, et al.
- Year / Venue: 2024 / arXiv
- DOI: 10.1016/j.media.2026.103943
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2412.01496v2
- Abstract: Determining whether two sets of images belong to the same or different distributions or domains is a crucial task in modern medical image analysis and deep learning; for example, to evaluate the output quality of image generative models. Currently, metrics used for this task either rely on the (potentially biased) choice of some downstream task, such as segmentation, or adopt task-independent perceptual metrics (e.g., Fréchet Inception Distance/FID) from natural imaging, which we show insufficiently capture anatomical features. To this end, we introduce a new perceptual metric tailored for medical images, FRD (Fréchet Radiomic Distance), which utilizes standardized, clinically meaningful, an

### 33. FDiff-Fusion:Denoising diffusion fusion network based on fuzzy learning for 3D medical image segmentation

- Score: 0.5836
- Authors: Weiping Ding, Sheng Geng, Haipeng Wang, Jiashuang Huang, et al.
- Year / Venue: 2024 / arXiv
- DOI: 10.1016/j.inffus.2024.102540
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2408.02075v1
- Abstract: In recent years, the denoising diffusion model has achieved remarkable success in image segmentation modeling. With its powerful nonlinear modeling capabilities and superior generalization performance, denoising diffusion models have gradually been applied to medical image segmentation tasks, bringing new perspectives and methods to this field. However, existing methods overlook the uncertainty of segmentation boundaries and the fuzziness of regions, resulting in the instability and inaccuracy of the segmentation results. To solve this problem, a denoising diffusion fusion network based on fuzzy learning for 3D medical image segmentation (FDiff-Fusion) is proposed in this paper. By integrati

### 34. Renal Medical Image Segmentation using Cross-Perceptron Deliberation and Multi- Scale Feature Fusion

- Score: 0.5817
- Authors: Swapna J., Roselin Kiruba R.
- Year / Venue: 2025 / Journal of Innovative Image Processing
- DOI: 10.36548/jiip.2025.3.025
- Sources: crossref
- Citations: 2
- URL: https://doi.org/10.36548/jiip.2025.3.025
- Abstract: Architectures based on Convolutional Neural Networks (CNNs), like U-Net, have demonstrated notable efficiency in the segmentation of renal medical images. However, because convolution processes are limited and have restricted accessible fields, they frequently have trouble capturing long-range dependencies. Recent developments have improved global context modeling by incorporating transformer modules into U-Net variations to address this issue. However, during the global fusion process, these transformers based methods run the risk of losing important local spatial information. This research introduces Multi-Scale MCPA, a unique architecture designed specifically for the segmentation of 2D r

### 35. Enhanced medical image segmentation using U-Net with residual connections and dual attention mechanism

- Score: 0.5792
- Authors: Leyi Xiao, Jiaojiao Song, Xia Xie, Chaodong Fan
- Year / Venue: 2025 / Engineering Applications of Artificial Intelligence
- DOI: 10.1016/j.engappai.2025.110794
- Sources: openalex
- Citations: 32
- URL: https://doi.org/10.1016/j.engappai.2025.110794

### 36. UNet++: A Nested U-Net Architecture for Medical Image Segmentation

- Score: 0.5736
- Authors: Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang
- Year / Venue: 2018 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/1807.10165v1
- Abstract: In this paper, we present UNet++, a new, more powerful architecture for medical image segmentation. Our architecture is essentially a deeply-supervised encoder-decoder network where the encoder and decoder sub-networks are connected through a series of nested, dense skip pathways. The re-designed skip pathways aim at reducing the semantic gap between the feature maps of the encoder and decoder sub-networks. We argue that the optimizer would deal with an easier learning task when the feature maps from the decoder and encoder networks are semantically similar. We have evaluated UNet++ in comparison with U-Net and wide U-Net architectures across multiple medical image segmentation tasks: nodule

### 37. An effective CNN and Transformer complementary network for medical image segmentation

- Score: 0.5595
- Authors: Feiniu Yuan, Zhengxiao Zhang, Zhijun Fang
- Year / Venue: 2022 / Pattern Recognition
- DOI: 10.1016/j.patcog.2022.109228
- Sources: openalex
- Citations: 569
- URL: https://doi.org/10.1016/j.patcog.2022.109228

### 38. Segmentation Ability Map: Interpret deep features for medical image segmentation

- Score: 0.5564
- Authors: Sheng He, Yanfang Feng, P. Ellen Grant, Yangming Ou
- Year / Venue: 2022 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2212.09206v1
- Abstract: Deep convolutional neural networks (CNNs) have been widely used for medical image segmentation. In most studies, only the output layer is exploited to compute the final segmentation results and the hidden representations of the deep learned features have not been well understood. In this paper, we propose a prototype segmentation (ProtoSeg) method to compute a binary segmentation map based on deep features. We measure the segmentation abilities of the features by computing the Dice between the feature segmentation map and ground-truth, named as the segmentation ability score (SA score for short). The corresponding SA score can quantify the segmentation abilities of deep features in different

### 39. Learning With Context Feedback Loop for Robust Medical Image Segmentation

- Score: 0.5536
- Authors: Kibrom Berihu Girum, Gilles Créhange, Alain Lalande
- Year / Venue: 2021 / arXiv
- DOI: 10.1109/tmi.2021.3060497
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2103.02844v1
- Abstract: Deep learning has successfully been leveraged for medical image segmentation. It employs convolutional neural networks (CNN) to learn distinctive image features from a defined pixel-wise objective function. However, this approach can lead to less output pixel interdependence producing incomplete and unrealistic segmentation results. In this paper, we present a fully automatic deep learning method for robust medical image segmentation by formulating the segmentation problem as a recurrent framework using two systems. The first one is a forward system of an encoder-decoder CNN that predicts the segmentation result from the input image. The predicted probabilistic output of the forward system i

### 40. Hybrid-scale contextual fusion network for medical image segmentation

- Score: 0.5499
- Authors: Hua Bao, Yuqing Zhu, Qing Li
- Year / Venue: 2022 / Computers in Biology and Medicine
- DOI: 10.1016/j.compbiomed.2022.106439
- Sources: openalex
- Citations: 33
- URL: https://doi.org/10.1016/j.compbiomed.2022.106439

### 41. FSC-UNet: a lightweight medical image segmentation algorithm fused with skip connections

- Score: 0.5446
- Authors: Yixin Chen, Jianjun Zhang, Xulin Zong, Zhipeng Zhao, et al.
- Year / Venue: 2022 / Fourteenth International Conference on Digital Image Processing (ICDIP 2022)
- DOI: 10.1117/12.2644360
- Sources: crossref
- Citations: 2
- URL: https://doi.org/10.1117/12.2644360

### 42. Adaptive Multi-Feature Fusion Transformer for Cardiac Medical Image Segmentation

- Score: 0.5414
- Authors: Xiaoning Zhang, Yanjun Peng, Sijia Xing, Zengmin Zhang
- Year / Venue: 2025 / Unknown
- DOI: 10.2139/ssrn.5087203
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.5087203

### 43. Test-time generative augmentation for medical image segmentation

- Score: 0.5264
- Authors: Xiao Ma, Yuhui Tao, Zetian Zhang, Yuhan Zhang, et al.
- Year / Venue: 2024 / arXiv
- DOI: 10.1016/j.media.2025.103902
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2406.17608v2
- Abstract: Medical image segmentation is critical for clinical diagnosis, treatment planning, and monitoring, yet segmentation models often struggle with uncertainties stemming from occlusions, ambiguous boundaries, and variations in imaging devices. Traditional test-time augmentation (TTA) techniques typically rely on predefined geometric and photometric transformations, limiting their adaptability and effectiveness in complex medical scenarios. In this study, we introduced Test-Time Generative Augmentation (TTGA), a novel augmentation strategy specifically tailored for medical image segmentation at inference time. Different from conventional augmentation strategies that suffer from excessive randomne

### 44. MambaMIM: Pre-training Mamba with State Space Token Interpolation and its Application to Medical Image Segmentation

- Score: 0.5264
- Authors: Fenghe Tang, Bingkun Nian, Yingtai Li, Zihang Jiang, et al.
- Year / Venue: 2024 / arXiv
- DOI: 10.1016/j.media.2025.103606
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2408.08070v2
- Abstract: Recently, the state space model Mamba has demonstrated efficient long-sequence modeling capabilities, particularly for addressing long-sequence visual tasks in 3D medical imaging. However, existing generative self-supervised learning methods have not yet fully unleashed Mamba's potential for handling long-range dependencies because they overlook the inherent causal properties of state space sequences in masked modeling. To address this challenge, we propose a general-purpose pre-training framework called MambaMIM, a masked image modeling method based on a novel TOKen-Interpolation strategy (TOKI) for the selective structure state space sequence, which learns causal relationships of state spa

### 45. Cross-dimensional transfer learning in medical image segmentation with deep learning

- Score: 0.5164
- Authors: Hicham Messaoudi, Ahror Belaid, Douraied Ben Salem, Pierre-Henri Conze
- Year / Venue: 2023 / arXiv
- DOI: 10.1016/j.media.2023.102868
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2307.15872v1
- Abstract: Over the last decade, convolutional neural networks have emerged and advanced the state-of-the-art in various image analysis and computer vision applications. The performance of 2D image classification networks is constantly improving and being trained on databases made of millions of natural images. However, progress in medical image analysis has been hindered by limited annotated data and acquisition constraints. These limitations are even more pronounced given the volumetry of medical imaging data. In this paper, we introduce an efficient way to transfer the efficiency of a 2D classification network trained on natural images to 2D, 3D uni- and multi-modal medical image segmentation applic

### 46. Weakly-Supervised 3D Medical Image Segmentation using Geometric Prior and Contrastive Similarity

- Score: 0.5164
- Authors: Hao Du, Qihua Dong, Yan Xu, Jing Liao
- Year / Venue: 2023 / arXiv
- DOI: 10.1109/tmi.2023.3269523
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2302.02125v1
- Abstract: Medical image segmentation is almost the most important pre-processing procedure in computer-aided diagnosis but is also a very challenging task due to the complex shapes of segments and various artifacts caused by medical imaging, (i.e., low-contrast tissues, and non-homogenous textures). In this paper, we propose a simple yet effective segmentation framework that incorporates the geometric prior and contrastive similarity into the weakly-supervised segmentation framework in a loss-based fashion. The proposed geometric prior built on point cloud provides meticulous geometry to the weakly-supervised segmentation proposal, which serves as better supervision than the inherent property of the b

### 47. Segment Anything Model for Medical Image Analysis: an Experimental Study

- Score: 0.5164
- Authors: Maciej A. Mazurowski, Haoyu Dong, Hanxue Gu, Jichen Yang, et al.
- Year / Venue: 2023 / arXiv
- DOI: 10.1016/j.media.2023.102918
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2304.10517v3
- Abstract: Training segmentation models for medical images continues to be challenging due to the limited availability of data annotations. Segment Anything Model (SAM) is a foundation model that is intended to segment user-defined objects of interest in an interactive manner. While the performance on natural images is impressive, medical image domains pose their own set of challenges. Here, we perform an extensive evaluation of SAM's ability to segment medical images on a collection of 19 medical imaging datasets from various modalities and anatomies. We report the following findings: (1) SAM's performance based on single prompts highly varies depending on the dataset and the task, from IoU=0.1135 for

### 48. CapsNet for Medical Image Segmentation

- Score: 0.4993
- Authors: Minh Tran, Viet-Khoa Vo-Ho, Kyle Quinn, Hien Nguyen, et al.
- Year / Venue: 2022 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2203.08948v1
- Abstract: Convolutional Neural Networks (CNNs) have been successful in solving tasks in computer vision including medical image segmentation due to their ability to automatically extract features from unstructured data. However, CNNs are sensitive to rotation and affine transformation and their success relies on huge-scale labeled datasets capturing various input variations. This network paradigm has posed challenges at scale because acquiring annotated data for medical segmentation is expensive, and strict privacy regulations. Furthermore, visual representation learning with CNNs has its own flaws, e.g., it is arguable that the pooling layer in traditional CNNs tends to discard positional information

### 49. Test-Time Adaptable Neural Networks for Robust Medical Image Segmentation

- Score: 0.4864
- Authors: Neerav Karani, Ertunc Erdil, Krishna Chaitanya, Ender Konukoglu
- Year / Venue: 2020 / arXiv
- DOI: 10.1016/j.media.2020.101907
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2004.04668v4
- Abstract: Convolutional Neural Networks (CNNs) work very well for supervised learning problems when the training dataset is representative of the variations expected to be encountered at test time. In medical image segmentation, this premise is violated when there is a mismatch between training and test images in terms of their acquisition details, such as the scanner model or the protocol. Remarkable performance degradation of CNNs in this scenario is well documented in the literature. To address this problem, we design the segmentation CNN as a concatenation of two sub-networks: a relatively shallow image normalization CNN, followed by a deep CNN that segments the normalized image. We train both the

### 50. Uncertainty-aware multi-view co-training for semi-supervised medical image segmentation and domain adaptation

- Score: 0.4864
- Authors: Yingda Xia, Dong Yang, Zhiding Yu, Fengze Liu, et al.
- Year / Venue: 2020 / arXiv
- DOI: 10.1016/j.media.2020.101766
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2006.16806v1
- Abstract: Although having achieved great success in medical image segmentation, deep learning-based approaches usually require large amounts of well-annotated data, which can be extremely expensive in the field of medical image analysis. Unlabeled data, on the other hand, is much easier to acquire. Semi-supervised learning and unsupervised domain adaptation both take the advantage of unlabeled data, and they are closely related to each other. In this paper, we propose uncertainty-aware multi-view co-training (UMCT), a unified framework that addresses these two tasks for volumetric medical image segmentation. Our framework is capable of efficiently utilizing unlabeled data for better performance. We fi

### 51. Building medical image classifiers with very limited data using segmentation networks

- Score: 0.4664
- Authors: Ken C. L. Wong, Tanveer Syeda-Mahmood, Mehdi Moradi
- Year / Venue: 2018 / arXiv
- DOI: 10.1016/j.media.2018.07.010
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/1808.05205v1
- Abstract: Deep learning has shown promising results in medical image analysis, however, the lack of very large annotated datasets confines its full potential. Although transfer learning with ImageNet pre-trained classification models can alleviate the problem, constrained image sizes and model complexities can lead to unnecessary increase in computational cost and decrease in performance. As many common morphological features are usually shared by different classification tasks of an organ, it is greatly beneficial if we can extract such features to improve classification with limited samples. Therefore, inspired by the idea of curriculum learning, we propose a strategy for building medical image clas

### 52. GAN-GA: A Generative Model based on Genetic Algorithm for Medical Image Generation

- Score: 0.4593
- Authors: M. AbdulRazek, G. Khoriba, M. Belal
- Year / Venue: 2023 / arXiv
- DOI: 10.3389/978-2-8325-1231-9
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2401.00314v1
- Abstract: Medical imaging is an essential tool for diagnosing and treating diseases. However, lacking medical images can lead to inaccurate diagnoses and ineffective treatments. Generative models offer a promising solution for addressing medical image shortage problems due to their ability to generate new data from existing datasets and detect anomalies in this data. Data augmentation with position augmentation methods like scaling, cropping, flipping, padding, rotation, and translation could lead to more overfitting in domains with little data, such as medical image data. This paper proposes the GAN-GA, a generative model optimized by embedding a genetic algorithm. The proposed model enhances image f

### 53. TransMorph: Transformer for unsupervised medical image registration

- Score: 0.4393
- Authors: Junyu Chen, Eric C. Frey, Yufan He, William P. Segars, et al.
- Year / Venue: 2021 / arXiv
- DOI: 10.1016/j.media.2022.102615
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2111.10480v6
- Abstract: In the last decade, convolutional neural networks (ConvNets) have been a major focus of research in medical image analysis. However, the performances of ConvNets may be limited by a lack of explicit consideration of the long-range spatial relationships in an image. Recently Vision Transformer architectures have been proposed to address the shortcomings of ConvNets and have produced state-of-the-art performances in many medical imaging applications. Transformers may be a strong candidate for image registration because their substantially larger receptive field enables a more precise comprehension of the spatial correspondence between moving and fixed images. Here, we present TransMorph, a hyb

### 54. Multiple Skip Connections of Dilated Convolution Network for Semantic Segmentation

- Score: 0.4231
- Authors: Takayoshi Yamashita, Hironori Furukawa, Hironobu Fujiyoshi
- Year / Venue: 2018 / 2018 25th IEEE International Conference on Image Processing (ICIP)
- DOI: 10.1109/icip.2018.8451033
- Sources: crossref
- Citations: 11
- URL: https://doi.org/10.1109/icip.2018.8451033

### 55. Distribution Matching Losses Can Hallucinate Features in Medical Image Translation

- Score: 0.4093
- Authors: Joseph Paul Cohen, Margaux Luck, Sina Honari
- Year / Venue: 2018 / arXiv
- DOI: 10.1007/978-3-030-00928-1_60
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/1805.08841v3
- Abstract: This paper discusses how distribution matching losses, such as those used in CycleGAN, when used to synthesize medical images can lead to mis-diagnosis of medical conditions. It seems appealing to use these new image synthesis methods for translating images from a source to a target domain because they can produce high quality images and some even do not require paired data. However, the basis of how these image translation models work is through matching the translation output to the distribution of the target domain. This can cause an issue when the data provided in the target domain has an over or under representation of some classes (e.g. healthy or sick). When the output of an algorithm

## Coverage Note

This report supports screening and does not guarantee exhaustive coverage. Final scholarly use requires human review.
