# Scholarly Deep Research Retrieval Report

- Query: skin lesion segmentation transformer mamba
- Year range: 2020 to 2026
- Records after deduplication: 102
- Ranking: score = 0.40 * lexical_relevance + 0.20 * citation_score + 0.15 * recency_score + 0.10 * venue_source_confidence + 0.10 * open_access_score + 0.05 * metadata_completeness

## Ranked Papers

### 1. UltraLight VM-UNet: Parallel Vision Mamba significantly reduces parameters for skin lesion segmentation

- Score: 0.8634
- Authors: Renkai Wu, Yinghao Liu, Guochen Ning, Pengchen Liang, et al.
- Year / Venue: 2025 / Patterns
- DOI: 10.1016/j.patter.2025.101298
- Sources: openalex, crossref, arxiv
- Citations: 293
- URL: https://doi.org/10.1016/j.patter.2025.101298
- Abstract: Previously, medical image segmentation models were often constrained by computational resources and memory, limiting their application in mobile medical devices or resource-constrained environments. In contrast, the UltraLight VM-UNet model proposed in this study, with its low parameter count and computational complexity, is able to reduce the requirements on device resources without sacrificing performance. This means that it can conveniently assist doctors in diagnosing diseases in more scenarios, such as on medical devices in remote areas or in emergency medical rescue for fast processing of medical image data. In the long run, the ambition of this research is to promote the popularizatio

### 2. SkinMamba: A Precision Skin Lesion Segmentation Architecture with Cross-Scale Global State Modeling and Frequency Boundary Guidance

- Score: 0.8624
- Authors: Shun Zou, Zhang, Mingya, Fan, Bingjian, Zhou, Zhengyi, et al.
- Year / Venue: 2024 / arXiv (Cornell University)
- DOI: 10.48550/arxiv.2409.10890
- Sources: openalex, arxiv
- Citations: 22
- URL: https://doi.org/10.48550/arxiv.2409.10890
- Abstract: Skin lesion segmentation is a crucial method for identifying early skin cancer. In recent years, both convolutional neural network (CNN) and Transformer-based methods have been widely applied. Moreover, combining CNN and Transformer effectively integrates global and local relationships, but remains limited by the quadratic complexity of Transformer. To address this, we propose a hybrid architecture based on Mamba and CNN, called SkinMamba. It maintains linear complexity while offering powerful long-range dependency modeling and local feature extraction capabilities. Specifically, we introduce the Scale Residual State Space Block (SRSSB), which captures global contextual relationships and cro

### 3. SGNet: A Structure-Guided Network with Dual-Domain Boundary Enhancement and Semantic Fusion for Skin Lesion Segmentation

- Score: 0.8329
- Authors: Haijiao Yun, Qingyu Du, Ziqing Han, Mingjing Li, et al.
- Year / Venue: 2025 / Sensors
- DOI: 10.3390/s25154652
- Sources: openalex
- Citations: 7
- URL: https://doi.org/10.3390/s25154652
- Abstract: Segmentation of skin lesions in dermoscopic images is critical for the accurate diagnosis of skin cancers, particularly malignant melanoma, yet it is hindered by irregular lesion shapes, blurred boundaries, low contrast, and artifacts, such as hair interference. Conventional deep learning methods, typically based on UNet or Transformer architectures, often face limitations in regard to fully exploiting lesion features and incur high computational costs, compromising precise lesion delineation. To overcome these challenges, we propose SGNet, a structure-guided network, integrating a hybrid CNN-Mamba framework for robust skin lesion segmentation. The SGNet employs the Visual Mamba (VMamba) enc

### 4. AttmNet: a hybrid Transformer integrating self-attention, Mamba, and multi-layer convolution for enhanced lesion segmentation

- Score: 0.8292
- Authors: Hancan Zhu, Yibing Huang, Kelin Yao, Jin Shang, et al.
- Year / Venue: 2025 / Quantitative Imaging in Medicine and Surgery
- DOI: 10.21037/qims-2024-2561
- Sources: openalex
- Citations: 6
- URL: https://doi.org/10.21037/qims-2024-2561
- Abstract: Background: Accurate lesion segmentation is critical for cancer diagnosis and treatment. Convolutional neural networks (CNNs) are widely used for medical image segmentation but struggle to capture long-range dependencies. Transformers mitigate this limitation but come with high computational costs. Mamba, a state-space model (SSM), efficiently models long-range dependencies but lacks precision in fine details. To address these challenges, this study aimed to develop a novel segmentation approach that combines the strengths of CNNs, Transformers, and Mamba, enhancing both global context understanding and local feature extraction in medical image segmentation. Methods: We propose AttmNet, a U-

### 5. VM-UNet: Vision Mamba UNet for Medical Image Segmentation

- Score: 0.815
- Authors: Jiacheng Ruan, Jincheng Li, Suncheng Xiang
- Year / Venue: 2025 / ACM Transactions on Multimedia Computing Communications and Applications
- DOI: 10.1145/3767748
- Sources: openalex
- Citations: 1308
- URL: https://doi.org/10.1145/3767748
- Abstract: In the realm of medical image segmentation, both CNN-based and Transformer-based models have been extensively explored. However, CNNs exhibit limitations in long-range modeling capabilities, whereas Transformers are hampered by their quadratic computational complexity. Recently, State Space Models (SSMs), exemplified by Mamba, have emerged as a promising approach. They not only excel in modeling long-range interactions but also maintain a linear computational complexity. In this paper, leveraging state space models, we propose a U-shape architecture model for medical image segmentation, named Vision Mamba UNet (VM-UNet). Specifically, the Visual State Space (VSS) block is introduced as the f

### 6. VML-UNet: Fusing Vision Mamba and Lightweight Attention Mechanism for Skin Lesion Segmentation

- Score: 0.8056
- Authors: Tang Tang, Haihui Wang, Qiang Rao, Ke Zuo, et al.
- Year / Venue: 2025 / Electronics
- DOI: 10.3390/electronics14142866
- Sources: openalex
- Citations: 2
- URL: https://doi.org/10.3390/electronics14142866
- Abstract: Deep learning has advanced medical image segmentation, yet existing methods struggle with complex anatomical structures. Mainstream models, such as CNN, Transformer, and hybrid architectures, face challenges including insufficient information representation and redundant complexity, which limit their clinical deployment. Developing efficient and lightweight networks is crucial for accurate lesion localization and optimized clinical workflows. We propose the VML-UNet, a lightweight segmentation network with core innovations including the CPMamba module and the multi-scale local supervision module (MLSM). The CPMamba module integrates the visual state space (VSS) block and a channel prior atte

### 7. HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation

- Score: 0.8046
- Authors: Jiashu Xu
- Year / Venue: 2024 / arXiv (Cornell University)
- DOI: 10.48550/arxiv.2405.05007
- Sources: openalex
- Citations: 72
- URL: https://doi.org/10.48550/arxiv.2405.05007
- Abstract: Automatic medical image segmentation technology has the potential to expedite pathological diagnoses, thereby enhancing the efficiency of patient care. However, medical images often have complex textures and structures, and the models often face the problem of reduced image resolution and information loss due to downsampling. To address this issue, we propose HC-Mamba, a new medical image segmentation model based on the modern state space model Mamba. Specifically, we introduce the technique of dilated convolution in the HC-Mamba model to capture a more extensive range of contextual information without increasing the computational cost by extending the perceptual field of the convolution ker

### 8. HBTMNet: A Tri-Hybrid Boundary-Aware Transformer Mamba Network for Precise Skin Lesion Segmentation

- Score: 0.795
- Authors: Sanjyoti Kumari Tarai, Renuka Arora
- Year / Venue: 2026 / International Journal of Drug Delivery Technology
- DOI: 10.25258/ijddt.16.74s.27
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.25258/ijddt.16.74s.27
- Abstract: Accurate automated segmentation of melanoma from dermoscopic images remains a critical challenge in computer-aided diagnosis pipelines due to high intra-class diversity, fuzzy lesion boundaries, and structural artifacts. Conventional deep learning frameworks, including convolutional neural networks, vision transformers, and standard hybrid models, struggle to reconcile localized textural feature extraction, long-range global context modelling, and sharp edge delineation. We propose HBTMNet, a Tri-Hybrid Boundary-Aware Transformer Mamba Network unified within an end-to-end encoderdecoder paradigm. The architecture integrates a linear-complexity Swin-UMamba backbone using 2D selective-scan vis

### 9. A Deep Feature Fusion Visual Mamba for Skin Lesion Segmentation

- Score: 0.795
- Authors: Sisi Yan, Hao Zheng, Yaru Fu, Yuhui Ren, et al.
- Year / Venue: 2026 / International Conference on Image, Signal Processing and Pattern Recognition (ISPP)
- DOI: 10.1109/ispp69262.2026.11542987
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.1109/ispp69262.2026.11542987
- Abstract: Skin lesion segmentation is a critical task in medical image analysis, particularly for early detection of skin cancer. Existing deep learning methods, including Convolutional Neural Networks and Transformer-based architectures, face challenges in balancing computational efficiency and long-range dependency modeling. To address these issues, we propose a Deep Feature Fusion Visual Mamba for accurate and efficient skin lesion segmentation. Our core contribution is a cooperative design between the bottleneck Visual Mamba and the proposed Dual Feature Fusion Module (DFM), where Mamba efficiently models longrange dependencies and DFM integrates attention-guided multiscale feature fusion to joint

### 10. An Adaptive Multi-Scale Fusion Model Based on Mamba for Skin Lesion Image Segmentation

- Score: 0.795
- Authors: Li Yan, Bo Peng, Qiang Li, Bo Li
- Year / Venue: 2026 / International Conference on Computer Supported Cooperative Work in Design (CSCWD)
- DOI: 10.1109/cscwd68734.2026.11582186
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.1109/cscwd68734.2026.11582186
- Abstract: Accurately capturing lesion regions is crucial in skin lesion image segmentation. While existing CNN and Transformer segmentation models have made significant progress in this field, achieving a balance between segmentation accuracy and computational efficiency remains challenging. To address this issue, we propose a lightweight Mamba-based Adaptive Multi-Scale Fusion UNet (AMF-UNet) model for skin lesion image segmentation. We design a Local Enhanced Residual Mamba (LERM) Block, which combines depthwise separable convolutions with a Mamba-based parallel processing module, effectively reducing the number of model parameters while maintaining the ability to model both global and local feature

### 11. AC-MambaSeg: An Adaptive Convolution and Mamba-Based Architecture for Enhanced Skin Lesion Segmentation

- Score: 0.7941
- Authors: Nguyen Viet Thanh, Van-Truong Pham, Thi-Thao Tran
- Year / Venue: 2024 / Lecture notes in networks and systems
- DOI: 10.1007/978-3-031-76197-3_2
- Sources: openalex, arxiv
- Citations: 34
- URL: https://doi.org/10.1007/978-3-031-76197-3_2
- Abstract: Skin lesion segmentation is a critical task in computer-aided diagnosis systems for dermatological diseases. Accurate segmentation of skin lesions from medical images is essential for early detection, diagnosis, and treatment planning. In this paper, we propose a new model for skin lesion segmentation namely AC-MambaSeg, an enhanced model that has the hybrid CNN-Mamba backbone, and integrates advanced components such as Convolutional Block Attention Module (CBAM), Attention Gate, and Selective Kernel Bottleneck. AC-MambaSeg leverages the Vision Mamba framework for efficient feature extraction, while CBAM and Selective Kernel Bottleneck enhance its ability to focus on informative regions and 

### 12. MUCM-Net: a Mamba powered UCM-Net for skin lesion segmentation

- Score: 0.7907
- Authors: Chunyu Yuan, Dongfang Zhao, Sos С. Agaian
- Year / Venue: 2024 / Exploration of Medicine
- DOI: 10.37349/emed.2024.00250
- Sources: openalex, crossref, arxiv
- Citations: 30
- URL: https://doi.org/10.37349/emed.2024.00250
- Abstract: Aim: Skin lesion segmentation is critical for early skin cancer detection. Challenges in automatic segmentation from dermoscopic images include variations in color, texture, and artifacts of indistinct lesion boundaries. This study aims to develop and evaluate MUCM-Net, a lightweight and efficient model for skin lesion segmentation, leveraging Mamba state-space models integrated with UCM-Net architecture optimized for mobile deployment and early skin cancer detection. Methods: MUCM-Net combines Convolutional Neural Networks (CNNs), multi-layer perceptions (MLPs), and Mamba elements into a hybrid feature learning module. Results: The model was trained and tested on the International Skin Imag

### 13. ScaleFusionNet: transformer-guided multi-scale feature fusion for skin lesion segmentation

- Score: 0.7871
- Authors: Saqib Qamar, Syed Furqan Qadri, Roobaea Alroobaea, Goram Mufarah Alshmrani, et al.
- Year / Venue: 2025 / Scientific Reports
- DOI: 10.1038/s41598-025-17300-x
- Sources: openalex, arxiv
- Citations: 18
- URL: https://doi.org/10.1038/s41598-025-17300-x
- Abstract: Abstract Melanoma is a malignant tumor that originates from skin cell lesions. Accurate and efficient segmentation of skin lesions is essential for quantitative analysis but remains a challenge owing to blurred lesion boundaries, gradual color changes, and irregular shapes. To address this, we propose ScaleFusionNet, a hybrid model that integrates a Cross-Attention Transformer Module (CATM) and adaptive fusion block (AFB) to enhance feature extraction and fusion by capturing both local and global features. We introduce CATM, which utilizes Swin transformer blocks and Cross Attention Fusion (CAF) to adaptively refine feature fusion and reduce semantic gaps in the encoder-decoder to improve se

### 14. Seg-MambaDiff: Skin Lesion Segmentation Based on Mamba and Diffusion Model

- Score: 0.785
- Authors: Jiacheng Zheng, Xinghong Chen, Weilin Wu, Guannan Chen
- Year / Venue: 2025 / International Conference on Computer Graphics, Image and Virtualization (ICCGIV)
- DOI: 10.1109/iccgiv65419.2025.11085387
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.1109/iccgiv65419.2025.11085387
- Abstract: With the advancement of medical imaging, skin lesion segmentation has become essential in clinical diagnosis and treatment, especially for early detection of dermatological conditions. Although CNNs and Transformer architectures have been widely adopted for medical image segmentation tasks, they still struggle to accurately capture subtle details. Recently, both diffusion models and Mamba models have each shown notable advancements in image segmentation tasks, however, an efficient hybrid model is still lacking, and the diffusion model continues to face the challenge of high complexity. To address these challenges, we propose Seg-MambaDiff, a novel framework for skin lesions segmenting that 

### 15. MambaU-Lite: A Lightweight Model Based on Mamba and Integrated Channel-Spatial Attention for Skin Lesion Segmentation

- Score: 0.7823
- Authors: Thi-Nhu-Quynh Nguyen, Quang-Huy Ho, Duy-Thai Nguyen, Hoang-Minh-Quang Le, et al.
- Year / Venue: 2025 / Lecture notes in networks and systems
- DOI: 10.1007/978-981-95-1746-6_6
- Sources: openalex, arxiv
- Citations: 15
- URL: https://doi.org/10.1007/978-981-95-1746-6_6
- Abstract: Early detection of skin abnormalities plays a crucial role in diagnosing and treating skin cancer. Segmentation of affected skin regions using AI-powered devices is relatively common and supports the diagnostic process. However, achieving high performance remains a significant challenge due to the need for high-resolution images and the often unclear boundaries of individual lesions. At the same time, medical devices require segmentation models to have a small memory foot-print and low computational cost. Based on these requirements, we introduce a novel lightweight model called MambaU-Lite, which combines the strengths of Mamba and CNN architectures, featuring just over 400K parameters and 

### 16. Skin-lesion segmentation using boundary-aware segmentation network and classification based on a mixture of convolutional and transformer neural networks

- Score: 0.7808
- Authors: Javaria Amin, Marium Azhar, Habiba Arshad, Amad Zafar, et al.
- Year / Venue: 2025 / Frontiers in Medicine
- DOI: 10.3389/fmed.2025.1524146
- Sources: crossref
- Citations: 25
- URL: https://doi.org/10.3389/fmed.2025.1524146
- Abstract: Background Skin cancer is one of the most prevalent cancers worldwide. In the clinical domain, skin lesions such as melanoma detection are still a challenge due to occlusions, poor contrast, poor image quality, and similarities between skin lesions. Deep-/machine-learning methods are used for the early, accurate, and efficient detection of skin lesions. Therefore, we propose a boundary-aware segmentation network (BASNet) model comprising prediction and residual refinement modules. Materials and methods The prediction module works like a U-Net and is densely supervised by an encoder and decoder. A hybrid loss function is used, which has the potential to help in the clinical domain of dermatol

### 17. Transformer-based multi-attention hybrid networks for skin lesion segmentation

- Score: 0.7795
- Authors: Zhiwei Dong, Jinjiang Li, Zhen Hua
- Year / Venue: 2024 / Expert Systems with Applications
- DOI: 10.1016/j.eswa.2023.123016
- Sources: crossref
- Citations: 45
- URL: https://doi.org/10.1016/j.eswa.2023.123016

### 18. KM-UNet: medical image segmentation with selective-scan Mamba and Kolmogorov-Arnold networks

- Score: 0.7714
- Authors: Yibo Zhang, Jingwen Zhao, Xiang Liu, Xian Tang, et al.
- Year / Venue: 2026 / PeerJ Computer Science
- DOI: 10.7717/peerj-cs.3727
- Sources: openalex
- Citations: 190
- URL: https://doi.org/10.7717/peerj-cs.3727
- Abstract: Biomedical image segmentation plays a crucial role in aiding diagnosis and treatment planning. However, constructing effective frameworks remains challenging due to the variable size and irregular shape of target structures. U-Net has become a cornerstone in this field, but integrating it with Transformer or Multilayer Perceptron (MLP) models faces limitations such as quadratic computational complexity and insufficient interpretability. To address these challenges, we propose KM-UNet, a novel structure inspired by state-space models (SSMs) ( e.g ., Mamba) and the Kolmogorov-Arnold network (KAN). KM-UNet leverages nonlinear, learnable activation functions, rooted in the Kolmogorov-Arnold repr

### 19. SUTrans-NET: a hybrid transformer approach to skin lesion segmentation

- Score: 0.7708
- Authors: Yaqin Li, Tonghe Tian, Jing Hu, Cao Yuan
- Year / Venue: 2024 / PeerJ Computer Science
- DOI: 10.7717/peerj-cs.1935
- Sources: crossref
- Citations: 25
- URL: https://doi.org/10.7717/peerj-cs.1935
- Abstract: Melanoma is a malignant skin tumor that threatens human life and health. Early detection is essential for effective treatment. However, the low contrast between melanoma lesions and normal skin and the irregularity in size and shape make skin lesions difficult to detect with the naked eye in the early stages, making the task of skin lesion segmentation challenging. Traditional encoder-decoder built with U-shaped networks using convolutional neural network (CNN) networks have limitations in establishing long-term dependencies and global contextual connections, while the Transformer architecture is limited in its application to small medical datasets. To address these issues, we propose a new 

### 20. Intelligent skin lesion segmentation using deformable attention Transformer U‐Net with bidirectional attention mechanism in skin cancer images

- Score: 0.7635
- Authors: Lili Cai, Keke Hou, Su Zhou
- Year / Venue: 2024 / Skin Research and Technology
- DOI: 10.1111/srt.13783
- Sources: crossref
- Citations: 19
- URL: https://doi.org/10.1111/srt.13783
- Abstract: Abstract Background In recent years, the increasing prevalence of skin cancers, particularly malignant melanoma, has become a major concern for public health. The development of accurate automated segmentation techniques for skin lesions holds immense potential in alleviating the burden on medical professionals. It is of substantial clinical importance for the early identification and intervention of skin cancer. Nevertheless, the irregular shape, uneven color, and noise interference of the skin lesions have presented significant challenges to the precise segmentation. Therefore, it is crucial to develop a high‐precision and intelligent skin lesion segmentation framework for clinical treatme

### 21. UltraLBM-UNet: Ultralight bidirectional mamba-based model for skin lesion segmentation

- Score: 0.7598
- Authors: 范琳煊, Juntao Jiang, Weixuan Liu, Zhucun Xue, et al.
- Year / Venue: 2026 / Neurocomputing
- DOI: 10.1016/j.neucom.2026.134608
- Sources: openalex, crossref, arxiv
- Citations: 4
- URL: https://doi.org/10.1016/j.neucom.2026.134608
- Abstract: Skin lesion segmentation is a crucial step in dermatology for guiding clinical decision-making. However, existing methods for accurate, robust, and resource-efficient lesion analysis have limitations, including low performance and high computational complexity. To address these limitations, we propose UltraLBM-UNet, a lightweight U-Net variant that integrates a bidirectional Mamba-based global modeling mechanism with multi-branch local feature perception. The proposed architecture integrates efficient local feature injection with bidirectional state-space modeling, enabling richer contextual interaction across spatial dimensions while maintaining computational compactness suitable for point-

### 22. Deep Skin Lesion Segmentation with Transformer-CNN Fusion: Toward Intelligent Skin Cancer Analysis

- Score: 0.7579
- Authors: Xin Wang, Xiaopei Zhang, Xingang Wang
- Year / Venue: 2025 / 2025 6th International Conference on Internet of Things, Artificial Intelligence and Mechanical Automation (IoTAIMA)
- DOI: 10.1109/iotaima66468.2025.11212688
- Sources: crossref, arxiv
- Citations: 7
- URL: https://doi.org/10.1109/iotaima66468.2025.11212688
- Abstract: This paper proposes a high-precision semantic segmentation method based on an improved TransUNet architecture to address the challenges of complex lesion structures, blurred boundaries, and significant scale variations in skin lesion images. The method integrates a transformer module into the traditional encoder-decoder framework to model global semantic information, while retaining a convolutional branch to preserve local texture and edge features. This enhances the model's ability to perceive fine-grained structures. A boundary-guided attention mechanism and multi-scale upsampling path are also designed to improve lesion boundary localization and segmentation consistency. To verify the eff

### 23. Semi-Supervised Skin Lesion Segmentation With Coupling CNN and Transformer Features

- Score: 0.7577
- Authors: Mohammad D. Alahmadi, Wajdi Alghamdi
- Year / Venue: 2022 / IEEE Access
- DOI: 10.1109/access.2022.3224005
- Sources: crossref
- Citations: 42
- URL: https://doi.org/10.1109/access.2022.3224005

### 24. LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation

- Score: 0.757
- Authors: Weibin Liao, Yinghao Zhu, Xinyuan Wang, Chengwei Pan, et al.
- Year / Venue: 2024 / arXiv (Cornell University)
- DOI: 10.48550/arxiv.2403.05246
- Sources: openalex
- Citations: 233
- URL: https://doi.org/10.48550/arxiv.2403.05246
- Abstract: UNet and its variants have been widely used in medical image segmentation. However, these models, especially those based on Transformer architectures, pose challenges due to their large number of parameters and computational loads, making them unsuitable for mobile health applications. Recently, State Space Models (SSMs), exemplified by Mamba, have emerged as competitive alternatives to CNN and Transformer architectures. Building upon this, we employ Mamba as a lightweight substitute for CNN and Transformer within UNet, aiming at tackling challenges stemming from computational resource limitations in real medical settings. To this end, we introduce the Lightweight Mamba UNet (LightM-UNet) th

### 25. U-SwinTrans: Automated Skin Lesion Segmentation Using Swin Transformer

- Score: 0.747
- Authors: Aaditya Lochan Sharma, Kalpana Sharma, Utkarsh Prakash Srivastava, Palash Ghosal
- Year / Venue: 2025 / 2025 3rd International Conference on Intelligent Systems, Advanced Computing and Communication (ISACC)
- DOI: 10.1109/isacc65211.2025.10969397
- Sources: crossref
- Citations: 9
- URL: https://doi.org/10.1109/isacc65211.2025.10969397

### 26. MCM-UNet: Mamba convolutional mixing network for skin lesion image segmentation

- Score: 0.7428
- Authors: Minchen Yang, Nur Intan Raihana Ruhaiyem
- Year / Venue: 2026 / Biomedical Signal Processing and Control
- DOI: 10.1016/j.bspc.2025.108791
- Sources: crossref
- Citations: 5
- URL: https://doi.org/10.1016/j.bspc.2025.108791

### 27. FFM-Net: Fusing Frequency Selection Information with Mamba for Skin Lesion Segmentation

- Score: 0.7356
- Authors: Lifang Chen, Entao Yu, Qi Ping Cao, Ke Hu
- Year / Venue: 2025 / Information
- DOI: 10.3390/info16121102
- Sources: openalex, crossref
- Citations: 2
- URL: https://doi.org/10.3390/info16121102
- Abstract: Accurate segmentation of lesion regions is essential for skin cancer diagnosis. As dermoscopic images of skin lesions demonstrate different sizes, diverse shapes, fuzzy boundaries, and so on, accurate segmentation still faces great challenges. To address these issues, we propose a new dermatologic image segmentation network, FFM-Net. In FFM-Net, we design a new FM block encoder based on state space models (SSMs), which integrates a low-frequency information extraction module (LEM) and an edge detail extraction module (EEM) to extract broader overall structural information and more accurate edge detail information, respectively. At the same time, we dynamically adjust the input channel ratios

### 28. TrUNet: Dual-Branch Network by Fusing CNN and Transformer for Skin Lesion Segmentation

- Score: 0.7341
- Authors: Wei Chen, Qian Mu, Jie Qi
- Year / Venue: 2024 / IEEE Access
- DOI: 10.1109/access.2024.3463713
- Sources: crossref
- Citations: 8
- URL: https://doi.org/10.1109/access.2024.3463713

### 29. BEFNet: A Hybrid CNN-Mamba Architecture for Accurate Skin Lesion Image Segmentation

- Score: 0.7228
- Authors: Wei Wang, Yuewen Luo, Xin Wang
- Year / Venue: 2024 / 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)
- DOI: 10.1109/bibm62325.2024.10822106
- Sources: crossref
- Citations: 5
- URL: https://doi.org/10.1109/bibm62325.2024.10822106

### 30. VMPANet: Vision Mamba Skin Lesion Image Segmentation Model Based on Prompt and Attention Mechanism Fusion

- Score: 0.7206
- Authors: Zinuo Peng, Shuxian Liu, Chenhao Li
- Year / Venue: 2025 / Journal of Imaging
- DOI: 10.3390/jimaging11120443
- Sources: crossref
- Citations: 2
- URL: https://doi.org/10.3390/jimaging11120443
- Abstract: In the realm of medical image processing, the segmentation of dermatological lesions is a pivotal technique for the early detection of skin cancer. However, existing methods for segmenting images of skin lesions often encounter limitations when dealing with intricate boundaries and diverse lesion shapes. To address these challenges, we propose VMPANet, designed to accurately localize critical targets and capture edge structures. VMPANet employs an inverted pyramid convolution to extract multi-scale features while utilizing the visual Mamba module to capture long-range dependencies among image features. Additionally, we leverage previously extracted masks as cues to facilitate efficient featu

### 31. When Mamba Meets xLSTM: An Efficient and Precise Method with the xLSTM-VMUNet Model for Skin lesion Segmentation

- Score: 0.7156
- Authors: Fang, Zhuoyi, Liu, Jiajia, Shi, Kexuan, Qiang Han
- Year / Venue: 2024 / arXiv (Cornell University)
- DOI: 10.48550/arxiv.2411.09363
- Sources: openalex
- Citations: 2
- URL: https://doi.org/10.48550/arxiv.2411.09363
- Abstract: Automatic melanoma segmentation is essential for early skin cancer detection, yet challenges arise from the heterogeneity of melanoma, as well as interfering factors like blurred boundaries, low contrast, and imaging artifacts. While numerous algorithms have been developed to address these issues, previous approaches have often overlooked the need to jointly capture spatial and sequential features within dermatological images. This limitation hampers segmentation accuracy, especially in cases with indistinct borders or structurally similar lesions. Additionally, previous models lacked both a global receptive field and high computational efficiency. In this work, we present the xLSTM-VMUNet M

### 32. Lesion-Aware Vision Transformer and Mamba Fusion for Lightweight Dermoscopic Skin Cancer Classification

- Score: 0.715
- Authors: Tanzila Saba
- Year / Venue: 2026 / Journal of Intelligent Decision Making and Information Science
- DOI: 10.17762/jidmis.v3.5754
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.17762/jidmis.v3.5754
- Abstract: Skin cancer is one of the most destructive malignancies of the current decade and the fifth most common cancer. Automated dermoscopic skin lesion diagnosis is a foundational task in computer-aided diagnosis (CAD). Early clinical detection of melanoma drastically increases the 5-year patient survival rate from below 15% in advanced Stage IV to over 98% in localized Stage IA. To address these challenges while maintaining parameter efficiency suitable for edge-device deployment, this work proposes the Lesion-Aware Fusion MambaFormer (LAF-MambaFormer), a streamlined, three-stage hybrid architecture engineered for robust binary classification (Benign vs. Malignant). With a compact budget of exact

### 33. An Ultra-Lightweight Cross-scale Attention Mamba Network for Accurate Skin Lesion Segmentation

- Score: 0.7122
- Authors: Razan Alharith, Jiashu Zhang, Chengqiang Zhao
- Year / Venue: 2026 / IEEE Journal of Biomedical and Health Informatics
- DOI: 10.1109/jbhi.2026.3708901
- Sources: crossref
- Citations: 1
- URL: https://doi.org/10.1109/jbhi.2026.3708901

### 34. FSM-UNet: An ultra-lightweight frequency-spatial Mamba network for skin lesion segmentation

- Score: 0.7079
- Authors: Heshi Zheng, Xiaohong Huang, Yanlei Ge, Zhao Zhang, et al.
- Year / Venue: 2026 / Image and Vision Computing
- DOI: 10.1016/j.imavis.2026.106179
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.1016/j.imavis.2026.106179

### 35. Mamba-EdgeNet: learnable edge-guided state space model for skin lesion segmentation

- Score: 0.705
- Authors: JiaMing Xu, Qi Mao, Lei Qiu, Yu Chen, et al.
- Year / Venue: 2026 / Biomedical Physics & Engineering Express
- DOI: 10.1088/2057-1976/aeada8
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.1088/2057-1976/aeada8
- Abstract: Skin lesion segmentation is one of the key tasks in computer-aided diagnosis. However, accurate boundary delineation remains challenging because of low contrast, irregular boundaries, and visual artifacts in skin lesion areas. To address these problems, we developed Mamba-EdgeNet. Mamba-EdgeNet is a hybrid encoder-decoder framework that selectively integrates ResNet50 with EfficientMamba2D state-space blocks. A resolution-adaptive scanning strategy dynamically alternates between multi-directional and sequential spatial scanning, enabling long-range contextual modeling with linear complexity relative to sequence length. To delineate fuzzy borders, a learnable edge-guidance branch is construct

### 36. Transformer and CNN-Based Deep Learning Models for Skin Lesion Segmentation

- Score: 0.7022
- Authors: Abdelkader Alrabai, Amira Echtioui, Fathi Kallel
- Year / Venue: 2025 / 2025 IEEE International Conference on Advanced Systems and Emergent Technologies (IC_ASET)
- DOI: 10.1109/ic_aset65966.2025.11232106
- Sources: crossref
- Citations: 1
- URL: https://doi.org/10.1109/ic_aset65966.2025.11232106

### 37. S³-Mamba: Small-Size-Sensitive Mamba for Lesion Segmentation

- Score: 0.7011
- Authors: Gui Wang, Yuexiang Li, Wenting Chen, Meidan Ding, et al.
- Year / Venue: 2025 / Proceedings of the AAAI Conference on Artificial Intelligence
- DOI: 10.1609/aaai.v39i7.32824
- Sources: openalex
- Citations: 21
- URL: https://doi.org/10.1609/aaai.v39i7.32824
- Abstract: Small lesions play a critical role in early disease diagnosis and intervention of severe infections. Popular models often face challenges in segmenting small lesions, as it occupies only a minor portion of an image, while down-sampling operations may inevitably lose focus on local features of small lesions. To tackle the challenges, we propose a Small-Size-Sensitive Mamba (S³-Mamba), which promotes the sensitivity to small lesions across three dimensions: channel, spatial, and training strategy. Specifically, an Enhanced Visual State Space block is designed to focus on small lesions through multiple residual connections to preserve local features, and selectively amplify important details wh

### 38. LSF-Mamba: A Lightweight Skin Lesion Segmentation Network with Local Structural Compensation and Frequency-Domain Residual Supplementation

- Score: 0.7
- Authors: Jinsong Zhang, Wei Liang, Guanqiu Qi, Zhiqin Zhu
- Year / Venue: 2026 / Algorithms
- DOI: 10.3390/a19060444
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.3390/a19060444
- Abstract: Accurate skin lesion segmentation remains a significant challenge for lightweight models, as they often struggle to effectively balance global context understanding with stable local structural representation. Shallow skip connections can introduce redundant background noise into the decoder, while lightweight Mamba-based backbones have limited ability to preserve two-dimensional local continuity during sequential scanning. Additionally, direct enhancement in the frequency domain may amplify high-frequency components that are irrelevant to lesion discrimination, thereby compromising representation stability. To address these issues, this paper proposes LSF-Mamba, a lightweight skin lesion se

### 39. Mscmamba-Unet: A Lightweight Multi-Scale Mamba Network for Skin Lesion Segmentation

- Score: 0.6979
- Authors: Shouming Hou, Jianchao Hou, Yuteng Pang, Aoyu Xia, et al.
- Year / Venue: 2025 / SSRN Electronic Journal
- DOI: 10.2139/ssrn.5197807
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.5197807

### 40. FCMamba: frequency-aware CNN-Mamba hybrid model for skin lesion segmentation

- Score: 0.695
- Authors: Wenxuan Liu, Zexuan Ji
- Year / Venue: 2026 / International Conference on Graphics and Image Processing (ICGIP)
- DOI: 10.1117/12.3094276
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.1117/12.3094276
- Abstract: To address the challenge of balancing global context modeling and local detail capture in skin lesion segmentation, this paper proposes Frequency-Aware CNN-Mamba (FCMamba), a novel hybrid architecture that integrates spatial-frequency collaborative modeling for high-precision medical image segmentation. Existing methods, such as CNNs, lack global dependency modeling, while Transformers and pure Mamba models suffer from high computational complexity or local detail loss. FCMamba introduces a dual-branch encoder: a CNN branch extracts multi-scale local texture features, and a Mamba branch captures global structural information through selective state space modeling. Innovatively, discrete wave

### 41. DPFF-net: dual-path feature fusion network with vision Mamba for skin lesion segmentation

- Score: 0.6943
- Authors: Yongquan Xiao, Xiaomei Tian, Hongwei Deng, Yong Mei Deng, et al.
- Year / Venue: 2025 / Physica Scripta
- DOI: 10.1088/1402-4896/ae10e9
- Sources: openalex
- Citations: 1
- URL: https://doi.org/10.1088/1402-4896/ae10e9
- Abstract: Abstract Accurate segmentation of skin lesions is crucial for the early screening and treatment of skin cancer. In recent years, Convolutional Neural Networks (CNNs) and Transformer-based approaches have achieved remarkable progress in image segmentation tasks. However, convolutional neural networks are limited in their ability to model long-range dependencies, which limits their effectiveness in capturing fine-grained details within medical images, while Transformers are constrained by their quadratic computational complexity. Recent studies have demonstrated that State Space Models (SSMs), particularly those represented by the Mamba architecture, can efficiently model long-range interactio

### 42. FITrans: Skin Lesion Segmentation Based on Feature Integration and Transformer

- Score: 0.6935
- Authors: Likun Zhang
- Year / Venue: 2023 / 2023 3rd International Conference on Neural Networks, Information and Communication Engineering (NNICE)
- DOI: 10.1109/nnice58320.2023.10105777
- Sources: crossref
- Citations: 2
- URL: https://doi.org/10.1109/nnice58320.2023.10105777

### 43. MaxCAT-Net: Hybrid transformer-based skin lesion classification and lightweight segmentation with calibration and cross-dataset validation

- Score: 0.6929
- Authors: Sadek Al Prince, Israt Fatema Shorna
- Year / Venue: 2026 / Unknown
- DOI: 10.64898/2026.11.01.763389
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.64898/2026.11.01.763389
- Abstract: Abstract Class imbalance, poor calibration of confidence and poor generalisation under different imaging conditions are restrictions of automated analysis of dermoscopic skin lesions. We propose a hybrid classifier MaxCAT-Net (MaxViT + CBAM + Attention Transformer), which consists of a pre-trained MaxViT-Tiny backbone and the combination of CBAM and two-layer lightweight Transformer encoder and evaluate four encoder-decoder networks for lesion segmentation as a separate task on ISIC 2018 Task 1. All the images of exact/near duplicate were removed and the remaining 10,012 images were distributed lesion-wise (72/14/14) without two or more images having the same type of lesion in HAM10000. On t

### 44. MambaSegNet: Global Context Modeling with Mamba for Skin Lesion Segmentation

- Score: 0.6929
- Authors: Anitha Setti, Saladi Saritha, Yepuganti Karuna, Sankati Ramakrishna
- Year / Venue: 2026 / 2026 International Conference on Cognitive Computing and Networking Systems (ICC-CNS)
- DOI: 10.1109/icc-cns70518.2026.11606064
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/icc-cns70518.2026.11606064

### 45. Attn-UNTER: Attention-Guided Hierarchical Transformer for Skin Lesion Segmentation

- Score: 0.6929
- Authors: M.A.Rafi
- Year / Venue: 2026 / 2026 5th International Conference on Electrical, Computer &amp;amp; Telecommunication Engineering (ICECTE)
- DOI: 10.1109/icecte69292.2026.11429363
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/icecte69292.2026.11429363

### 46. HPFT-Net: Hierarchical Perception-Fusion Transformer for Skin Lesion Segmentation

- Score: 0.6929
- Authors: Guodong Zhang, Jiayi Zhu, Wei Guo, Ronghui Ju
- Year / Venue: 2026 / 2026 International Conference on Image, Signal Processing and Pattern Recognition (ISPP)
- DOI: 10.1109/ispp69262.2026.11543133
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/ispp69262.2026.11543133

### 47. Skin Lesion Segmentation and Cancer Classification using U -Net and Vision Transformer Models

- Score: 0.6929
- Authors: Shantini K, Shalini V, E Ahila Devi
- Year / Venue: 2026 / 2026 6th International Conference on Pervasive Computing and Social Networking (ICPCSN)
- DOI: 10.1109/icpcsn68523.2026.11543712
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/icpcsn68523.2026.11543712

### 48. Skin Lesion Segmentation using Vision Transformer and UNet

- Score: 0.69
- Authors: Bibat Thokar
- Year / Venue: 2025 / International Journal on Engineering Technology and Infrastructure Development
- DOI: 10.3126/injet-indev.v2i1.82195
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.3126/injet-indev.v2i1.82195
- Abstract: Medical images play a crucial role in diagnosing and analyzing serious illnesses. To make the typically lengthy process of reviewing these images more efficient, an automated approach for segmenting abnormal features is essential. Due to the limited availability of medical image data, deep learning frameworks for multi-class image segmentation have been developed. However, many current deep learning frameworks lack flexibility. To address the problem, advanced architectures have been added to improve segmentation performance. In particular, the UNet model has been enhanced with a Vision Transformer (ViT), enabling it to better capture structural features in medical images. The ISIC dataset h

### 49. Research Review of Deep Learning in Skin Lesions Image Segmentation

- Score: 0.685
- Authors: MENG Xiangfu, LI Jiaxun, YU Chunlin, LU Yunxuan
- Year / Venue: 2026 / DOAJ (DOAJ: Directory of Open Access Journals)
- DOI: 10.3778/j.issn.1673-9418.2504019
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.3778/j.issn.1673-9418.2504019
- Abstract: Skin lesions exhibit a wide variety of types and complex clinical manifestations, ranging from benign conditions to malignant melanomas. Early detection and accurate segmentation of these lesions are critical for the diagnosis and treatment of skin cancer, particularly in the early identification and localization of high-risk lesions such as malignant melanoma, which can significantly improve patient survival rates. In recent years, deep learning techniques have achieved remarkable progress in skin lesion image segmentation, greatly enhancing both accuracy and efficiency. This paper presents a comprehensive review of deep learning research in the field of skin lesion image segmentation. Firs

### 50. Algorithm design for skin lesion image segmentation based on improved VM-Unet

- Score: 0.685
- Authors: Mingyuan Gao, Yuchao Wang, Zhengzhu Wen
- Year / Venue: 2026 / International Conference on Image Processing and Intelligent Control (IPIC)
- DOI: 10.1117/12.3119214
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.1117/12.3119214
- Abstract: CNNs struggle with long-range dependencies, while Transformers face quadratic complexity in high-resolution skin lesion segmentation. Recently, State Space Models (SSMs) like Mamba have offered efficient linear-complexity alternatives; however, they often lack local inductive bias, leading to background noise retention and semantic-detail misalignment. To address these issues, we propose ACS-VMUNet, incorporating a "calibration-then-fusion" strategy. A Convolutional Block Attention Module (CBAM) is embedded in shallow layers to suppress artifacts such as hair textures, while an Adaptive Fusion of Semantic and Detail Integration (AF-SDI) module aligns multi-scale features via cross-scale inte

### 51. H-vmunet: High-order Vision Mamba UNet for medical image segmentation

- Score: 0.6846
- Authors: Renkai Wu, Yinghao Liu, Pengchen Liang, Qing Chang
- Year / Venue: 2025 / Neurocomputing
- DOI: 10.1016/j.neucom.2025.129447
- Sources: openalex
- Citations: 276
- URL: https://doi.org/10.1016/j.neucom.2025.129447

### 52. UCMViT: Enhancing Skin Lesion Segmentation with Efficient Transformer-Based Feature Fusion and Small Lesion Cropping

- Score: 0.6829
- Authors: Zahra Ghorbani, Farzaneh Koohestani, Nader Karimi, Shahram Shirani, et al.
- Year / Venue: 2025 / 2025 IEEE World AI IoT Congress (AIIoT)
- DOI: 10.1109/aiiot65859.2025.11105228
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/aiiot65859.2025.11105228

### 53. Multiple Hyperparameter Optimization in Image Processing Employing Swin Transformer for Enhanced Skin Lesion Segmentation

- Score: 0.6829
- Authors: Praveena S, Veeramakali T
- Year / Venue: 2025 / 2025 International Conference on Computational Robotics, Testing and Engineering Evaluation (ICCRTEE)
- DOI: 10.1109/iccrtee64519.2025.11053082
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/iccrtee64519.2025.11053082

### 54. Emerging Semi-Supervised Skin Lesion Segmentation using Unlabelled Samples with GAN and Transformer Representations

- Score: 0.6829
- Authors: M. Megala, T.R. Nisha Dayana
- Year / Venue: 2025 / 2025 9th International Conference on Electronics, Communication and Aerospace Technology (ICECA)
- DOI: 10.1109/iceca66444.2025.11382863
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/iceca66444.2025.11382863

### 55. HMT-UNet: A hybird Mamba-Transformer Vision UNet for Medical Image Segmentation

- Score: 0.6805
- Authors: Zhang, Mingya, Chen, Zhihao, Yiyuan Ge, Xianping Tao
- Year / Venue: 2024 / arXiv (Cornell University)
- DOI: 10.48550/arxiv.2408.11289
- Sources: openalex
- Citations: 14
- URL: https://doi.org/10.48550/arxiv.2408.11289
- Abstract: In the field of medical image segmentation, models based on both CNN and Transformer have been thoroughly investigated. However, CNNs have limited modeling capabilities for long-range dependencies, making it challenging to exploit the semantic information within images fully. On the other hand, the quadratic computational complexity poses a challenge for Transformers. State Space Models (SSMs), such as Mamba, have been recognized as a promising method. They not only demonstrate superior performance in modeling long-range interactions, but also preserve a linear computational complexity. The hybrid mechanism of SSM (State Space Model) and Transformer, after meticulous design, can enhance its 

### 56. ASP-VMUNet: Atrous Shifted Parallel Vision Mamba U-Net for Skin Lesion Segmentation

- Score: 0.6779
- Authors: Muyi Bao, Shuchang Lyu, Zhaoyang Xu, Qi Zhao, et al.
- Year / Venue: 2025 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2503.19427v1
- Abstract: Skin lesion segmentation is a critical challenge in computer vision, and it is essential to separate pathological features from healthy skin for diagnostics accurately. Traditional Convolutional Neural Networks (CNNs) are limited by narrow receptive fields, and Transformers face significant computational burdens. This paper presents a novel skin lesion segmentation framework, the Atrous Shifted Parallel Vision Mamba UNet (ASP-VMUNet), which integrates the efficient and scalable Mamba architecture to overcome limitations in traditional CNNs and computationally demanding Transformers. The framework introduces an atrous scan technique that minimizes background interference and expands the recep

### 57. LiteMamba-Bound: A lightweight Mamba-based model with boundary-aware and normalized active contour loss for skin lesion segmentation

- Score: 0.6764
- Authors: Quang-Huy Ho, Thi-Nhu-Quynh Nguyen, Thi-Thao Tran, Van-Truong Pham
- Year / Venue: 2025 / Methods
- DOI: 10.1016/j.ymeth.2025.01.008
- Sources: openalex
- Citations: 23
- URL: https://doi.org/10.1016/j.ymeth.2025.01.008

### 58. MF-Mamba: An Efficient Multi-Feature Fusion Mamba Model for Medical Image Segmentation

- Score: 0.675
- Authors: Weitong Xing, Jun Geng, Xiaojuan Yao, Mengran Liu, et al.
- Year / Venue: 2025 / IEEE International Conference on High Performance Computing and Communications; IEEE International Conference on Smart City; IEEE International Conference on Data Science and Systems (HPCC/SmartCity/DSS)
- DOI: 10.1109/hpcc67675.2025.00061
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.1109/hpcc67675.2025.00061
- Abstract: Traditional CNN and Transformer show good performance in medical image segmentation, but both have problems. CNN capture local features with convolution operations but have trouble modeling long-range dependencies. Transformers use self-attention to understand global context but are very computationally expensive. In medical images, lesions have big changes in shape and size, requiring models to include both general shapes and specific local boundaries. This makes it important to combine local and global features well. To solve these problems, we suggest the Multi-Feature Fusion Mamba model (MF-Mamba). The model uses a Multi-Scale Channel Fusion Network (MCFN) to get object features at diffe

### 59. SkinFormer: Learning Statistical Texture Representation with Transformer for Skin Lesion Segmentation

- Score: 0.6679
- Authors: Rongtao Xu, Changwei Wang, Jiguang Zhang, Shibiao Xu, et al.
- Year / Venue: 2024 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2409.08652v1
- Abstract: Accurate skin lesion segmentation from dermoscopic images is of great importance for skin cancer diagnosis. However, automatic segmentation of melanoma remains a challenging task because it is difficult to incorporate useful texture representations into the learning process. Texture representations are not only related to the local structural information learned by CNN, but also include the global statistical texture information of the input image. In this paper, we propose a trans\textbf{Former} network (\textbf{SkinFormer}) that efficiently extracts and fuses statistical texture representation for \textbf{Skin} lesion segmentation. Specifically, to quantify the statistical texture of input

### 60. TESL-Net: A Transformer-Enhanced CNN for Accurate Skin Lesion Segmentation

- Score: 0.6679
- Authors: Shahzaib Iqbal, Muhammad Zeeshan, Mehwish Mehmood, Tariq M. Khan, et al.
- Year / Venue: 2024 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2408.09687v1
- Abstract: Early detection of skin cancer relies on precise segmentation of dermoscopic images of skin lesions. However, this task is challenging due to the irregular shape of the lesion, the lack of sharp borders, and the presence of artefacts such as marker colours and hair follicles. Recent methods for melanoma segmentation are U-Nets and fully connected networks (FCNs). As the depth of these neural network models increases, they can face issues like the vanishing gradient problem and parameter redundancy, potentially leading to a decrease in the Jaccard index of the segmentation model. In this study, we introduced a novel network named TESL-Net for the segmentation of skin lesions. The proposed TES

### 61. VM-UNET-V2: Rethinking Vision Mamba UNet for Medical Image Segmentation

- Score: 0.6674
- Authors: Mingya Zhang, Yue Yu, Jin Sun, Limei Gu, et al.
- Year / Venue: 2024 / Lecture notes in computer science
- DOI: 10.1007/978-981-97-5128-0_27
- Sources: openalex
- Citations: 213
- URL: https://doi.org/10.1007/978-981-97-5128-0_27

### 62. Skin Lesion Segmentation Improved by Transformer-based Networks with Inter-scale Dependency Modeling

- Score: 0.6579
- Authors: Sania Eskandari, Janet Lumpp, Luis Sanchez Giraldo
- Year / Venue: 2023 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2310.13604v1
- Abstract: Melanoma, a dangerous type of skin cancer resulting from abnormal skin cell growth, can be treated if detected early. Various approaches using Fully Convolutional Networks (FCNs) have been proposed, with the U-Net architecture being prominent To aid in its diagnosis through automatic skin lesion segmentation. However, the symmetrical U-Net model's reliance on convolutional operations hinders its ability to capture long-range dependencies crucial for accurate medical image segmentation. Several Transformer-based U-Net topologies have recently been created to overcome this limitation by replacing CNN blocks with different Transformer modules to capture local and global representations. Further

### 63. Inter-Scale Dependency Modeling for Skin Lesion Segmentation with Transformer-based Networks

- Score: 0.6579
- Authors: Sania Eskandari, Janet Lumpp
- Year / Venue: 2023 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2310.13727v1
- Abstract: Melanoma is a dangerous form of skin cancer caused by the abnormal growth of skin cells. Fully Convolutional Network (FCN) approaches, including the U-Net architecture, can automatically segment skin lesions to aid diagnosis. The symmetrical U-Net model has shown outstanding results, but its use of a convolutional operation limits its ability to capture long-range dependencies, which are essential for accurate medical image segmentation. In addition, the U-shaped structure suffers from the semantic gaps between the encoder and decoder. In this study, we developed and evaluated a U-shaped hierarchical Transformer-based structure for skin lesion segmentation while we proposed an Inter-scale Co

### 64. AViT: Adapting Vision Transformers for Small Skin Lesion Segmentation Datasets

- Score: 0.6579
- Authors: Siyi Du, Nourhan Bayasi, Ghassan Hamarneh, Rafeef Garbi
- Year / Venue: 2023 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2307.13897v2
- Abstract: Skin lesion segmentation (SLS) plays an important role in skin lesion analysis. Vision transformers (ViTs) are considered an auspicious solution for SLS, but they require more training data compared to convolutional neural networks (CNNs) due to their inherent parameter-heavy structure and lack of some inductive biases. To alleviate this issue, current approaches fine-tune pre-trained ViT backbones on SLS datasets, aiming to leverage the knowledge learned from a larger set of natural images to lower the amount of skin training data needed. However, fully fine-tuning all parameters of large backbones is computationally expensive and memory intensive. In this paper, we propose AViT, a novel ef

### 65. DermoMamba: a cross-scale Mamba-based model with guide fusion loss for skin lesion segmentation in dermoscopy images

- Score: 0.6458
- Authors: Ngoc-Khai Hoang, Dinh-Hieu Nguyen, Thi-Thao Tran, Van-Truong Pham
- Year / Venue: 2025 / Pattern Analysis and Applications
- DOI: 10.1007/s10044-025-01506-w
- Sources: openalex
- Citations: 7
- URL: https://doi.org/10.1007/s10044-025-01506-w

### 66. Boundary-aware Transformers for Skin Lesion Segmentation

- Score: 0.645
- Authors: Jiacheng Wang, Lan Wei, Liansheng Wang, Qichao Zhou, et al.
- Year / Venue: 2021 / arXiv
- DOI: 10.1007/978-3-030-87193-2_20
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2110.03864v1
- Abstract: Skin lesion segmentation from dermoscopy images is of great importance for improving the quantitative analysis of skin cancer. However, the automatic segmentation of melanoma is a very challenging task owing to the large variation of melanoma and ambiguous boundaries of lesion areas. While convolutional neutral networks (CNNs) have achieved remarkable progress in this task, most of existing solutions are still incapable of effectively capturing global dependencies to counteract the inductive bias caused by limited receptive fields. Recently, transformers have been proposed as a promising tool for global context modeling by employing a powerful global attention mechanism, but one of their mai

### 67. WVM-UNet: A Wavelet–Vision Mamba Framework for Enhanced Medical Image Segmentation

- Score: 0.6443
- Authors: Yulong Yang, Wen Gao, Zhengguo Wu, Chuanghua Yang
- Year / Venue: 2026 / Journal of Imaging
- DOI: 10.3390/jimaging12080375
- Sources: openalex
- Citations: 1
- URL: https://doi.org/10.3390/jimaging12080375
- Abstract: Accurate segmentation of skin lesions and gastrointestinal polyps is essential for early diagnosis and treatment planning. Currently, Convolutional Neural Networks (CNNs) are limited by local receptive fields, missing small lesions. While Transformers model global context, their quadratic computational complexity incurs high costs. To address these limitations, we propose the Wavelet-Vision Mamba UNet (WVM-UNet), integrating State Space Models (SSMs) for linear-complexity long-range dependencies and wavelet transforms for fine-grained feature extraction. The network employs a Wavelet-based Residual State Space (WRSS) block, combining the multi-scale decomposition of discrete wavelet transfor

### 68. SegUMamba: Integrating Mamba with U_net for Medical Image Segmentation

- Score: 0.6156
- Authors: Yue Zhao, Chenxuan Liu, Xuemeng Zhou, Xinyu Zhang
- Year / Venue: 2024 / International Conference on Image Processing, Computer Vision and Machine Learning (ICICML)
- DOI: 10.1109/icicml63543.2024.10958038
- Sources: openalex
- Citations: 2
- URL: https://doi.org/10.1109/icicml63543.2024.10958038
- Abstract: In medical image segmentation, capturing long-range dependencies is critical for precise segmentation. Convolutional Neural Networks(CNNs)excel at local features, while Transformers comprehend global context, but both struggle with long-range dependencies. Mamba addresses this with its selection mechanism and dynamic matrix. By integrating Mamba with U-Net, we developed the SegUMamba model, which effectively captures local lesion details as well as distant interactions between lesions and surrounding skin. The encoder-decoder architecture, built on VMamba and utilizing skip connections, merges shallow and deep-level information, enhancing segmentation accuracy. We designed a flexible integra

### 69. DMPB-Net: A Mamba-Driven Semi-Supervised Network for Precise Breast Ultrasound Lesion Segmentation

- Score: 0.6129
- Authors: Haiyang Sun, Xuanbo Xu, Li Wang
- Year / Venue: 2026 / Unknown
- DOI: 10.21203/rs.3.rs-10095241/v1
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.21203/rs.3.rs-10095241/v1
- Abstract: Abstract Accurate segmentation of breast lesions in ultrasound images is critical for early cancer diagnosis, yet it remains challenging due to speckle noise, acoustic artifacts, and blurred boundaries. In this paper, we propose DMPB-Net, a novel semi-supervised framework designed for precise breast lesion segmentation under limited supervision. The framework utilizes a Mamba-driven encoder to capture long-range global dependencies, effectively overcoming the receptive field limitations of conventional CNNs. To mitigate ultrasound-specific interference, we introduce a Semantic Decoupling (SD) module that separates lesion-related features from task-irrelevant background noise. Furthermore, a 

### 70. MSG-Mamba: Multi-Scale State Gated Mamba Network for 3D Breast MRI Lesion Segmentation

- Score: 0.6129
- Authors: Jiayi Wu, Xiyang Zhao, Chunli Kong
- Year / Venue: 2026 / 2026 41st Youth Academic Annual Conference of Chinese Association of Automation (YAC)
- DOI: 10.1109/yac71005.2026.11615919
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/yac71005.2026.11615919

### 71. Privileged Lesion-Context Relational Distillation for Mask-Free Skin Lesion Classification

- Score: 0.6079
- Authors: Abu Mukaddim Rahi, Md Mithun Hossain, Md Zulficar Hasan Joy, M. F. Mridha, et al.
- Year / Venue: 2026 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2607.18773v1
- Abstract: Accurate skin lesion classification can benefit from lesion segmentation masks, but requiring masks or an auxiliary segmentation model during inference reduces clinical practicality and increases computational complexity. This work introduces Privileged Lesion-Context Relational Distillation (PLCRD), a teacher-student framework that exploits lesion masks exclusively during training while preserving image-only inference. The privileged teacher jointly analyzes the original dermoscopic image and its mask-guided lesion region to learn lesion-specific and contextual diagnostic representations. An image-only student is then trained through complementary knowledge-transfer mechanisms that convey t

### 72. Automated Detection of Multiple Sclerosis Lesions on 7-tesla MRI Using U-net and Transformer-based Segmentation

- Score: 0.6079
- Authors: Michael Maynord, Minghui Liu, Cornelia Fermüller, Seongjin Choi, et al.
- Year / Venue: 2026 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2604.00469v1
- Abstract: Ultra-high field 7-tesla (7T) MRI improves visualization of multiple sclerosis (MS) white matter lesions (WML) but differs sufficiently in contrast and artifacts from 1.5-3T imaging - suggesting that widely used automated segmentation tools may not translate directly. We analyzed 7T FLAIR scans and generated reference WML masks from Lesion Segmentation Tool (LST) outputs followed by expert manual revision. As external comparators, we applied LST-LPA and the more recent LST-AI ensemble, both originally developed on lower-field data. We then trained 3D UNETR and SegFormer transformer-based models on 7T FLAIR at multiple resolutions (0.5x0.5x0.5^3, 1.0x1.0x1.0^3, and 1.5x1.5x2.0^3) and evaluate

### 73. Exploring the Impact of Skin Color on Skin Lesion Segmentation

- Score: 0.6079
- Authors: Kuniko Paxton, Medina Kapo, Amila Akagić, Koorosh Aslansefat, et al.
- Year / Venue: 2026 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2603.29694v1
- Abstract: Skin cancer, particularly melanoma, remains a major cause of morbidity and mortality, making early detection critical. AI-driven dermatology systems often rely on skin lesion segmentation as a preprocessing step to delineate the lesion from surrounding skin and support downstream analysis. While fairness concerns regarding skin tone have been widely studied for lesion classification, the influence of skin tone on the segmentation stage remains under-quantified and is frequently assessed using coarse, discrete skin tone categories. In this work, we evaluate three strong segmentation architectures (UNet, DeepLabV3 with a ResNet50 backbone, and DINOv2) on two public dermoscopic datasets (HAM100

### 74. ELM-UNet: An efficient and lightweight Vision Mamba UNet for skin lesion segmentation

- Score: 0.6072
- Authors: Songling Xia, Hongwei Deng, Feng Chen
- Year / Venue: 2025 / Journal of Computational Science
- DOI: 10.1016/j.jocs.2025.102743
- Sources: openalex
- Citations: 1
- URL: https://doi.org/10.1016/j.jocs.2025.102743

### 75. Mamba-Constrained Inter-Slice Consistency Network for Stroke Lesion Segmentation

- Score: 0.6029
- Authors: Lukas Schneider, Hannah Weber, Jonas Richter
- Year / Venue: 2025 / Unknown
- DOI: 10.20944/preprints202512.2518.v1
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.20944/preprints202512.2518.v1
- Abstract: Conventional 2D or pseudo-3D models often produce anatomically inconsistent stroke lesion masks across slices due to the lack of explicit inter-slice dependency modeling. MSC-Mamba addresses this problem by introducing a bidirectional Mamba backbone tailored to capture long-range interactions along the slice axis. The model encourages coherent lesion boundaries across adjacent slices and reduces abrupt fluctuations caused by noise or low contrast. A topology-aware decoder further improves structural continuity and prevents fragmentation. Experiments on ATLAS v2 (655 subjects; 520 for training and 135 for testing) show that MSC-Mamba achieves a Dice score of 0.856, outperforming nnUNet (0.806

### 76. Hybrid Transformer Encoder-Decoder Framework with Collective Deep Learning for Semantic Skin-Lesion Segmentation

- Score: 0.6
- Authors: Bushra Al-Saadi
- Year / Venue: 2026 / Journal of Al-Farabi for Engineering Sciences
- DOI: 10.59746/ftn10q41
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.59746/ftn10q41
- Abstract: Accurate skin-lesion segmentation remains difficult because dermoscopic images often contain low contrast, irregular borders, hair, illumination variation, and heterogeneous lesion appearance. This study proposes a hybrid Transformer encoder-decoder framework with collective deep learning to combine global contextual modeling with local boundary recovery and complementary prediction fusion. The pipeline uses Wiener filtering and dynamic histogram equalization, patch embedding, hierarchical Swin/Transformer encoding, skip-connected decoding, attention refinement, and normalized validation-based fusion of complementary prediction branches. The framework is evaluated on an internal 70%/15%/15% 

### 77. SMC-UNet: Spiral-Scan Mamba and CNN Fusing UNet for Skin Lesion Segmentation

- Score: 0.5979
- Authors: Jiahao Chen, Weihang Liu, Jiake Lv
- Year / Venue: 2026 / Communications in computer and information science
- DOI: 10.1007/978-981-92-3531-5_14
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.1007/978-981-92-3531-5_14

### 78. What Can We Learn from Inter-Annotator Variability in Skin Lesion Segmentation?

- Score: 0.5979
- Authors: Kumar Abhishek, Jeremy Kawahara, Ghassan Hamarneh
- Year / Venue: 2025 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2508.09381v1
- Abstract: Medical image segmentation exhibits intra- and inter-annotator variability due to ambiguous object boundaries, annotator preferences, expertise, and tools, among other factors. Lesions with ambiguous boundaries, e.g., spiculated or infiltrative nodules, or irregular borders per the ABCD rule, are particularly prone to disagreement and are often associated with malignancy. In this work, we curate IMA++, the largest multi-annotator skin lesion segmentation dataset, on which we conduct an in-depth study of variability due to annotator, malignancy, tool, and skill factors. We find a statistically significant (p<0.001) association between inter-annotator agreement (IAA), measured using Dice, and 

### 79. DiffMamba: Leveraging Mamba for Effective Fusion of Noise and Conditional Features in Diffusion Models for Skin Lesion Segmentation

- Score: 0.5879
- Authors: Amit Shakya, Shruti S. Phutke, Chetan Gupta, Rupesh Kumar, et al.
- Year / Venue: 2025 / Communications in computer and information science
- DOI: 10.1007/978-3-031-93703-3_5
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.1007/978-3-031-93703-3_5

### 80. Segmentation Style Discovery: Application to Skin Lesion Images

- Score: 0.5879
- Authors: Kumar Abhishek, Jeremy Kawahara, Ghassan Hamarneh
- Year / Venue: 2024 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2408.02787v1
- Abstract: Variability in medical image segmentation, arising from annotator preferences, expertise, and their choice of tools, has been well documented. While the majority of multi-annotator segmentation approaches focus on modeling annotator-specific preferences, they require annotator-segmentation correspondence. In this work, we introduce the problem of segmentation style discovery, and propose StyleSeg, a segmentation method that learns plausible, diverse, and semantically consistent segmentation styles from a corpus of image-mask pairs without any knowledge of annotator correspondence. StyleSeg consistently outperforms competing methods on four publicly available skin lesion segmentation (SLS) da

### 81. A Region of Interest Focused Triple UNet Architecture for Skin Lesion Segmentation

- Score: 0.585
- Authors: Guoqing Liu, Yu Guo, Caiying Wu, Guoqing Chen, et al.
- Year / Venue: 2023 / arXiv
- DOI: 10.1002/ima.23090
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2311.12581v1
- Abstract: Skin lesion segmentation is of great significance for skin lesion analysis and subsequent treatment. It is still a challenging task due to the irregular and fuzzy lesion borders, and diversity of skin lesions. In this paper, we propose Triple-UNet to automatically segment skin lesions. It is an organic combination of three UNet architectures with suitable modules. In order to concatenate the first and second sub-networks more effectively, we design a region of interest enhancement module (ROIE). The ROIE enhances the target object region of the image by using the predicted score map of the first UNet. The features learned by the first UNet and the enhanced image help the second UNet obtain a

### 82. Advancements in Semantic Skin Lesion Segmentation: A Comparative Study with YOLOv8 for Accurate Skin Lesion Segmentation

- Score: 0.5829
- Authors: Vishnudeep M R, Prateeth Rao, Lakshmeesha
- Year / Venue: 2023 / 2023 Global Conference on Information Technologies and Communications (GCITC)
- DOI: 10.1109/gcitc60406.2023.10425934
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/gcitc60406.2023.10425934

### 83. Dpms-Unet: Dual-Path Multi-Scale Unet with Parallel Cnn-Mamba Aggregation for Skin Lesion Segmentation

- Score: 0.5757
- Authors: 焱 刘, XunLi Fan, YingFan Li
- Year / Venue: 2025 / Unknown
- DOI: 10.2139/ssrn.5312894
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.5312894

### 84. Cross-Axis Transformer and Multi-Level Adaptive Feature Fusion Guided U-Net for Skin Lesion Segmentation

- Score: 0.5757
- Authors: zhihui liu, Jianwen Yang, Jie Hu
- Year / Venue: 2025 / Unknown
- DOI: 10.2139/ssrn.5885567
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.5885567

### 85. A Survey on Deep Learning for Skin Lesion Segmentation

- Score: 0.575
- Authors: Zahra Mirikharaji, Kumar Abhishek, Alceu Bissoto, Catarina Barata, et al.
- Year / Venue: 2022 / arXiv
- DOI: 10.1016/j.media.2023.102863
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2206.00356v3
- Abstract: Skin cancer is a major public health problem that could benefit from computer-aided diagnosis to reduce the burden of this common disease. Skin lesion segmentation from images is an important step toward achieving this goal. However, the presence of natural and artificial artifacts (e.g., hair and air bubbles), intrinsic factors (e.g., lesion shape and contrast), and variations in image acquisition conditions make skin lesion segmentation a challenging task. Recently, various researchers have explored the applicability of deep learning models to skin lesion segmentation. In this survey, we cross-examine 177 research papers that deal with deep learning-based segmentation of skin lesions. We a

### 86. Accurate and Robust Lesion RECIST Diameter Prediction and Segmentation with Transformers

- Score: 0.5679
- Authors: Youbao Tang, Ning Zhang, Yirui Wang, Shenghua He, et al.
- Year / Venue: 2022 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2208.13113v1
- Abstract: Automatically measuring lesion/tumor size with RECIST (Response Evaluation Criteria In Solid Tumors) diameters and segmentation is important for computer-aided diagnosis. Although it has been studied in recent years, there is still space to improve its accuracy and robustness, such as (1) enhancing features by incorporating rich contextual information while keeping a high spatial resolution and (2) involving new tasks and losses for joint optimization. To reach this goal, this paper proposes a transformer-based network (MeaFormer, Measurement transFormer) for lesion RECIST diameter prediction and segmentation (LRDPS). It is formulated as three correlative and complementary tasks: lesion segm

### 87. Attmnet: A Composite Attention Mamba Network for Skin Lesion Segmentation

- Score: 0.5657
- Authors: Pengchong Ma, Guanglei Wang, Yelin Zhang, Yan Li
- Year / Venue: 2024 / Unknown
- DOI: 10.2139/ssrn.4953297
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.4953297

### 88. Lesion Boundary Detection for Skin Lesion Segmentation Based on Boundary Sensing and Cnn-Transformer Fusion Networks

- Score: 0.5657
- Authors: XuZhen Huang, Yuliang Ma, Xiajin Mei, Zizhuo Wu, et al.
- Year / Venue: 2024 / Unknown
- DOI: 10.2139/ssrn.4970410
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.4970410

### 89. Skin lesion segmentation by fusing local and global features using axial shift and spatial state model

- Score: 0.5621
- Authors: Guangju Li, Qinghua Huang, Wei Wang, Longzhong Liu
- Year / Venue: 2025 / Applied Soft Computing
- DOI: 10.1016/j.asoc.2025.113261
- Sources: openalex
- Citations: 6
- URL: https://doi.org/10.1016/j.asoc.2025.113261

### 90. MT-TransUNet: Mediating Multi-Task Tokens in Transformers for Skin Lesion Segmentation and Classification

- Score: 0.5579
- Authors: Jingye Chen, Jieneng Chen, Zongwei Zhou, Bin Li, et al.
- Year / Venue: 2021 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2112.01767v1
- Abstract: Recent advances in automated skin cancer diagnosis have yielded performance on par with board-certified dermatologists. However, these approaches formulated skin cancer diagnosis as a simple classification task, dismissing the potential benefit from lesion segmentation. We argue that an accurate lesion segmentation can supplement the classification task with additive lesion information, such as asymmetry, border, intensity, and physical size; in turn, a faithful lesion classification can support the segmentation task with discriminant lesion features. To this end, this paper proposes a new multi-task framework, named MT-TransUNet, which is capable of segmenting and classifying skin lesions c

### 91. Less is More: Sample Selection and Label Conditioning Improve Skin Lesion Segmentation

- Score: 0.5479
- Authors: Vinicius Ribeiro, Sandra Avila, Eduardo Valle
- Year / Venue: 2020 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2004.13856v1
- Abstract: Segmenting skin lesions images is relevant both for itself and for assisting in lesion classification, but suffers from the challenge in obtaining annotated data. In this work, we show that segmentation may improve with less data, by selecting the training samples with best inter-annotator agreement, and conditioning the ground-truth masks to remove excessive detail. We perform an exhaustive experimental design considering several sources of variation, including three different test sets, two different deep-learning architectures, and several replications, for a total of 540 experimental runs. We found that sample selection and detail removal may have impacts corresponding, respectively, to 

### 92. MVCA-UNet: A Multi-scale Visual Convolutional Attention Architecture for Skin Lesion Segmentation

- Score: 0.5272
- Authors: Zhaohui Wang, Runzhi Xu, Jiyong Xu, Changfang Chen, et al.
- Year / Venue: 2025 / Lecture notes in computer science
- DOI: 10.1007/978-981-95-0036-9_18
- Sources: openalex
- Citations: 1
- URL: https://doi.org/10.1007/978-981-95-0036-9_18

### 93. Lesion Elevation Prediction from Skin Images Improves Diagnosis

- Score: 0.5079
- Authors: Kumar Abhishek, Ghassan Hamarneh
- Year / Venue: 2024 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2408.02792v1
- Abstract: While deep learning-based computer-aided diagnosis for skin lesion image analysis is approaching dermatologists' performance levels, there are several works showing that incorporating additional features such as shape priors, texture, color constancy, and illumination further improves the lesion diagnosis performance. In this work, we look at another clinically useful feature, skin lesion elevation, and investigate the feasibility of predicting and leveraging skin lesion elevation labels. Specifically, we use a deep learning model to predict image-level lesion elevation labels from 2D skin lesion images. We test the elevation prediction accuracy on the derm7pt dataset, and use the elevation 

### 94. Exploring the Limitations of Mamba in COPY and CoT Reasoning

- Score: 0.5079
- Authors: Ruifeng Ren, Zhicong Li, Yong Liu
- Year / Venue: 2024 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2410.03810v3
- Abstract: Transformers have become the backbone of modern Large Language Models (LLMs); however, their inference overhead grows linearly with the sequence length, posing challenges for modeling long sequences. In light of this, Mamba has attracted attention for maintaining a constant inference size, with empirical evidence demonstrating that it can match Transformer performance in sequence modeling while significantly reducing computational costs. However, an open question remains: can Mamba always bring savings while achieving performance comparable to Transformers? In this paper, we focus on analyzing the expressive ability of Mamba to perform our defined COPY operation and Chain of Thought (CoT) re

### 95. Fuzzy Information Seeded Region Growing for Automated Lesions After Stroke Segmentation in MR Brain Images

- Score: 0.4979
- Authors: Mario Pascual González
- Year / Venue: 2023 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2311.11742v1
- Abstract: In the realm of medical imaging, precise segmentation of stroke lesions from brain MRI images stands as a critical challenge with significant implications for patient diagnosis and treatment. Addressing this, our study introduces an innovative approach using a Fuzzy Information Seeded Region Growing (FISRG) algorithm. Designed to effectively delineate the complex and irregular boundaries of stroke lesions, the FISRG algorithm combines fuzzy logic with Seeded Region Growing (SRG) techniques, aiming to enhance segmentation accuracy. The research involved three experiments to optimize the FISRG algorithm's performance, each focusing on different parameters to improve the accuracy of stroke lesi

### 96. SCM-UNet: Spatial-channel Mamba UNet for medical image segmentation

- Score: 0.4891
- Authors: Haijie Yan, Qiuhong Hong, Shoulin Wei, Xiangliang Zhang, et al.
- Year / Venue: 2025 / Digital Signal Processing
- DOI: 10.1016/j.dsp.2025.105550
- Sources: openalex
- Citations: 8
- URL: https://doi.org/10.1016/j.dsp.2025.105550

### 97. Artifact-Based Domain Generalization of Skin Lesion Models

- Score: 0.4879
- Authors: Alceu Bissoto, Catarina Barata, Eduardo Valle, Sandra Avila
- Year / Venue: 2022 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2208.09756v1
- Abstract: Deep Learning failure cases are abundant, particularly in the medical area. Recent studies in out-of-distribution generalization have advanced considerably on well-controlled synthetic datasets, but they do not represent medical imaging contexts. We propose a pipeline that relies on artifacts annotation to enable generalization evaluation and debiasing for the challenging skin lesion analysis context. First, we partition the data into levels of increasingly higher biased training and test sets for better generalization assessment. Then, we create environments based on skin lesion artifacts to enable domain generalization methods. Finally, after robust training, we perform a test-time debiasi

### 98. Team NeuroPoly: Description of the Pipelines for the MICCAI 2021 MS New Lesions Segmentation Challenge

- Score: 0.4779
- Authors: Uzay Macar, Enamundram Naga Karthik, Charley Gros, Andréanne Lemay, et al.
- Year / Venue: 2021 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2109.05409v2
- Abstract: This paper gives a detailed description of the pipelines used for the 2nd edition of the MICCAI 2021 Challenge on Multiple Sclerosis Lesion Segmentation. An overview of the data preprocessing steps applied is provided along with a brief description of the pipelines used, in terms of the architecture and the hyperparameters. Our code for this work can be found at: https://github.com/ivadomed/ms-challenge-2021.

### 99. Analysis of skin lesion images with deep learning

- Score: 0.4779
- Authors: Josef Steppan, Sten Hanke
- Year / Venue: 2021 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2101.03814v1
- Abstract: Skin cancer is the most common cancer worldwide, with melanoma being the deadliest form. Dermoscopy is a skin imaging modality that has shown an improvement in the diagnosis of skin cancer compared to visual examination without support. We evaluate the current state of the art in the classification of dermoscopic images based on the ISIC-2019 Challenge for the classification of skin lesions and current literature. Various deep neural network architectures pre-trained on the ImageNet data set are adapted to a combined training data set comprised of publicly available dermoscopic and clinical images of skin lesions using transfer learning and model fine-tuning. The performance and applicabilit

### 100. Improved skin lesion recognition by a Self-Supervised Curricular Deep Learning approach

- Score: 0.4779
- Authors: Kirill Sirotkin, Marcos Escudero-Viñolo, Pablo Carballeira, Juan Carlos SanMiguel
- Year / Venue: 2021 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2112.12086v1
- Abstract: State-of-the-art deep learning approaches for skin lesion recognition often require pretraining on larger and more varied datasets, to overcome the generalization limitations derived from the reduced size of the skin lesion imaging datasets. ImageNet is often used as the pretraining dataset, but its transferring potential is hindered by the domain gap between the source dataset and the target dermatoscopic scenario. In this work, we introduce a novel pretraining approach that sequentially trains a series of Self-Supervised Learning pretext tasks and only requires the unlabeled skin lesion imaging data. We present a simple methodology to establish an ordering that defines a pretext task curri

### 101. MAMBA-Based Weakly Supervised Medical Image Segmentation with Cross-Modal Textual Information

- Score: 0.4585
- Authors: Zhen Kuan Pan, Wenhui Huang, Yuanjie Zheng
- Year / Venue: 2025 / Lecture notes in computer science
- DOI: 10.1007/978-3-032-04984-1_29
- Sources: openalex
- Citations: 2
- URL: https://doi.org/10.1007/978-3-032-04984-1_29

### 102. CMambaFuse: Circular mamba feature fusion for medical image segmentation

- Score: 0.4279
- Authors: Xudong Wu, Baohua Yuan, Juxiao Zhang, Lin Xiang, et al.
- Year / Venue: 2025 / Signal Image and Video Processing
- DOI: 10.1007/s11760-025-04789-y
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.1007/s11760-025-04789-y

## Coverage Note

This report supports screening and does not guarantee exhaustive coverage. Final scholarly use requires human review.
