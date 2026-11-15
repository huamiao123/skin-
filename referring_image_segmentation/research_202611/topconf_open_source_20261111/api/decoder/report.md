# Scholarly Deep Research Retrieval Report

- Query: medical image segmentation attention decoder
- Year range: 2020 to 2026
- Records after deduplication: 111
- Ranking: score = 0.40 * lexical_relevance + 0.20 * citation_score + 0.15 * recency_score + 0.10 * venue_source_confidence + 0.10 * open_access_score + 0.05 * metadata_completeness

## Ranked Papers

### 1. TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation

- Score: 0.935
- Authors: Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, et al.
- Year / Venue: 2021 / arXiv (Cornell University)
- DOI: 10.48550/arxiv.2102.04306
- Sources: openalex
- Citations: 8005
- URL: https://doi.org/10.48550/arxiv.2102.04306
- Abstract: Medical image segmentation is an essential prerequisite for developing healthcare systems, especially for disease diagnosis and treatment planning. On various medical image segmentation tasks, the u-shaped architecture, also known as U-Net, has become the de-facto standard and achieved tremendous success. However, due to the intrinsic locality of convolution operations, U-Net generally demonstrates limitations in explicitly modeling long-range dependency. Transformers, designed for sequence-to-sequence prediction, have emerged as alternative architectures with innate global self-attention mechanisms, but can result in limited localization abilities due to insufficient low-level details. In t

### 2. TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers

- Score: 0.9297
- Authors: Jieneng Chen, Jieru Mei, Xianhang Li, Yongyi Lu, et al.
- Year / Venue: 2024 / Medical Image Analysis
- DOI: 10.1016/j.media.2024.103280
- Sources: openalex
- Citations: 1636
- URL: https://doi.org/10.1016/j.media.2024.103280
- Abstract: Medical image segmentation is crucial for healthcare, yet convolution-based methods like U-Net face limitations in modeling long-range dependencies. To address this, Transformers designed for sequence-to-sequence predictions have been integrated into medical image segmentation. However, a comprehensive understanding of Transformers' self-attention in U-Net components is lacking. TransUNet, first introduced in 2021, is widely recognized as one of the first models to integrate Transformer into medical image analysis. In this study, we present the versatile framework of TransUNet that encapsulates Transformers' self-attention into two key modules: (1) a Transformer encoder tokenizing image patc

### 3. EMCAD: Efficient Multi-Scale Convolutional Attention Decoding for Medical Image Segmentation

- Score: 0.9079
- Authors: Md Mostafijur Rahman, Mustafa Munir, Radu Marculescu
- Year / Venue: 2024 / IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings
- DOI: 10.1109/cvpr52733.2024.01118
- Sources: openalex
- Citations: 615
- URL: https://doi.org/10.1109/cvpr52733.2024.01118
- Abstract: An efficient and effective decoding mechanism is crucial in medical image segmentation, especially in scenarios with limited computational resources. However, these decoding mechanisms usually come with high computational costs. To address this concern, we introduce EMCAD, a new efficient multi-scale convolutional attention decoder, designed to optimize both performance and computational efficiency. EMCAD leverages a unique multi-scale depth-wise convolution block, significantly enhancing feature maps through multi-scale convolutions. EMCAD also employs channel, spatial, and grouped (large-kernel) gated attention mechanisms, which are highly effective at capturing intricate spatial relations

### 4. DS-TransUNet: Dual Swin Transformer U-Net for Medical Image Segmentation

- Score: 0.8972
- Authors: Ailiang Lin, Bingzhi Chen, Jiayu Xu, Zheng Zhang, et al.
- Year / Venue: 2022 / IEEE Transactions on Instrumentation and Measurement
- DOI: 10.1109/tim.2022.3178991
- Sources: openalex
- Citations: 933
- URL: https://doi.org/10.1109/tim.2022.3178991
- Abstract: Automatic medical image segmentation has made great progress owing to the powerful deep representation learning. Inspired by the success of self-attention mechanism in Transformer, considerable efforts are devoted to designing the robust variants of encoder-decoder architecture with Transformer. However, the patch division used in the existing Transformer-based models usually ignores the pixel-level intrinsic structural features inside each patch. In this paper, we propose a novel deep medical image segmentation framework called Dual Swin Transformer U-Net (DS-TransUNet), which aims to incorporate the hierarchical Swin Transformer into both encoder and decoder of the standard U-shaped archit

### 5. TransAttUnet: Multi-Level Attention-Guided U-Net With Transformer for Medical Image Segmentation

- Score: 0.8952
- Authors: Bingzhi Chen, Yishu Liu, Zheng Zhang, Guangming Lu, et al.
- Year / Venue: 2023 / IEEE Transactions on Emerging Topics in Computational Intelligence
- DOI: 10.1109/tetci.2023.3309626
- Sources: openalex
- Citations: 544
- URL: https://doi.org/10.1109/tetci.2023.3309626
- Abstract: Accurate segmentation of organs or lesions from medical images is crucial for reliable diagnosis of diseases and organ morphometry. In recent years, convolutional encoder-decoder solutions have achieved substantial progress in the field of automatic medical image segmentation. Due to the inherent bias in the convolution operations, prior models mainly focus on local visual cues formed by the neighboring pixels, but fail to fully model the long-range contextual dependencies. In this article, we propose a novel Transformer-based Attention Guided Network calledTransAttUnet, in which the multi-level guided attention and multi-scale skip connection are designed to jointly enhance the performance 

### 6. Dual Cross-Attention for medical image segmentation

- Score: 0.8865
- Authors: Gorkem Can Ates, Prasoon Poozhikunnath Mohan, Emrah Çelik
- Year / Venue: 2023 / Engineering Applications of Artificial Intelligence
- DOI: 10.1016/j.engappai.2023.107139
- Sources: openalex, arxiv
- Citations: 234
- URL: https://doi.org/10.1016/j.engappai.2023.107139
- Abstract: We propose Dual Cross-Attention (DCA), a simple yet effective attention module that is able to enhance skip-connections in U-Net-based architectures for medical image segmentation. DCA addresses the semantic gap between encoder and decoder features by sequentially capturing channel and spatial dependencies across multi-scale encoder features. First, the Channel Cross-Attention (CCA) extracts global channel-wise dependencies by utilizing cross-attention across channel tokens of multi-scale encoder features. Then, the Spatial Cross-Attention (SCA) module performs cross-attention to capture spatial dependencies across spatial tokens. Finally, these fine-grained encoder features are up-sampled a

### 7. Multi-Scale Self-Guided Attention for Medical Image Segmentation

- Score: 0.8672
- Authors: Ashish Sinha, José Dolz
- Year / Venue: 2020 / IEEE Journal of Biomedical and Health Informatics
- DOI: 10.1109/jbhi.2020.2986926
- Sources: openalex
- Citations: 596
- URL: https://doi.org/10.1109/jbhi.2020.2986926
- Abstract: Even though convolutional neural networks (CNNs) are driving progress in medical image segmentation, standard models still have some drawbacks. First, the use of multi-scale approaches, i.e., encoder-decoder architectures, leads to a redundant use of information, where similar low-level features are extracted multiple times at multiple scales. Second, long-range feature dependencies are not efficiently modeled, resulting in non-optimal discriminative feature representations associated with each semantic class. In this paper we attempt to overcome these limitations with the proposed architecture, by capturing richer contextual dependencies based on the use of guided self-attention mechanisms.

### 8. Improved UNet with Attention for Medical Image Segmentation

- Score: 0.8645
- Authors: Ahmed AL Qurri, Mohamed Khaled Almekkawy
- Year / Venue: 2023 / Sensors
- DOI: 10.3390/s23208589
- Sources: openalex
- Citations: 136
- URL: https://doi.org/10.3390/s23208589
- Abstract: Medical image segmentation is crucial for medical image processing and the development of computer-aided diagnostics. In recent years, deep Convolutional Neural Networks (CNNs) have been widely adopted for medical image segmentation and have achieved significant success. UNet, which is based on CNNs, is the mainstream method used for medical image segmentation. However, its performance suffers owing to its inability to capture long-range dependencies. Transformers were initially designed for Natural Language Processing (NLP), and sequence-to-sequence applications have demonstrated the ability to capture long-range dependencies. However, their abilities to acquire local information are limite

### 9. MIST: Medical Image Segmentation Transformer with Convolutional Attention Mixing (CAM) Decoder

- Score: 0.8633
- Authors: Md. Motiur Rahman, Shiva Shokouhmand, Smriti Bhatt, Miad Faezipour
- Year / Venue: 2024 / IEEE Winter Conference on Applications of Computer Vision
- DOI: 10.1109/wacv57701.2024.00047
- Sources: openalex, crossref
- Citations: 52
- URL: https://doi.org/10.1109/wacv57701.2024.00047
- Abstract: One of the common and promising deep learning approaches used for medical image segmentation is transformers, as they can capture long-range dependencies among the pixels by utilizing self-attention. Despite being successful in medical image segmentation, transformers face limitations in capturing local contexts of pixels in multimodal dimensions. We propose a Medical Image Segmentation Transformer (MIST) incorporating a novel Convolutional Attention Mixing (CAM) decoder to address this issue. MIST has two parts- a pre-trained multi-axis vision transformer (MaxViT) is used as an encoder, and the encoded feature representation is passed through the CAM decoder for segmenting the images. In th

### 10. Multi-scale Hierarchical Vision Transformer with Cascaded Attention Decoding for Medical Image Segmentation

- Score: 0.8475
- Authors: Md. Mostafijur Rahman, Radu Mărculescu
- Year / Venue: 2023 / arXiv (Cornell University)
- DOI: 10.48550/arxiv.2303.16892
- Sources: openalex
- Citations: 63
- URL: https://doi.org/10.48550/arxiv.2303.16892
- Abstract: Transformers have shown great success in medical image segmentation. However, transformers may exhibit a limited generalization ability due to the underlying single-scale self-attention (SA) mechanism. In this paper, we address this issue by introducing a Multi-scale hiERarchical vIsion Transformer (MERIT) backbone network, which improves the generalizability of the model by computing SA at multiple scales. We also incorporate an attention-based decoder, namely Cascaded Attention Decoding (CASCADE), for further refinement of multi-stage features generated by MERIT. Finally, we introduce an effective multi-stage feature mixing loss aggregation (MUTATION) method for better model training via i

### 11. Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation

- Score: 0.8362
- Authors: Fan Zhang, Zhiwei Gu, Hua Wang
- Year / Venue: 2026 / Proceedings of the AAAI Conference on Artificial Intelligence
- DOI: 10.1609/aaai.v40i15.38235
- Sources: openalex
- Citations: 9
- URL: https://doi.org/10.1609/aaai.v40i15.38235
- Abstract: To address the limitations of Transformer decoders in capturing edge details, recognizing local textures and modeling spatial continuity, this paper proposes a novel decoder framework specifically designed for medical image segmentation, comprising three core modules. First, the Adaptive Cross-Fusion Attention (ACFA) module integrates channel feature enhancement with spatial attention mechanisms and introduces learnable guidance in three directions (planar, horizontal, and vertical) to enhance responsiveness to key regions and structural orientations. Second, the Triple Feature Fusion Attention (TFFA) module fuses features from Spatial, Fourier and Wavelet domains, achieving joint frequency-

### 12. MEA-Net: multilayer edge attention network for medical image segmentation

- Score: 0.835
- Authors: Huilin Liu, Yue Feng, Hong Li Xu, Shufen Liang, et al.
- Year / Venue: 2022 / Scientific Reports
- DOI: 10.1038/s41598-022-11852-y
- Sources: openalex
- Citations: 56
- URL: https://doi.org/10.1038/s41598-022-11852-y
- Abstract: Medical image segmentation is a fundamental step in medical analysis and diagnosis. In recent years, deep learning networks have been used for precise segmentation. Numerous improved encoder-decoder structures have been proposed for various segmentation tasks. However, high-level features have gained more research attention than the abundant low-level features in the early stages of segmentation. Consequently, the learning of edge feature maps has been limited, which can lead to ambiguous boundaries of the predicted results. Inspired by the encoder-decoder network and attention mechanism, this study investigates a novel multilayer edge attention network (MEA-Net) to fully utilize the edge in

### 13. Medical Transformer: Gated Axial-Attention for Medical Image Segmentation

- Score: 0.8125
- Authors: Jeya Maria Jose Valanarasu, Poojan Oza, Ilker Hacihaliloglu, Vishal M. Patel
- Year / Venue: 2021 / Lecture notes in computer science
- DOI: 10.1007/978-3-030-87193-2_4
- Sources: openalex
- Citations: 1630
- URL: https://doi.org/10.1007/978-3-030-87193-2_4

### 14. RCLD-UNet: An Asymmetric U-Shaped Encoder-Decoder Based on Recurrent Criss-Cross Attention and Linear Deformable Convolution for Medical Image Segmentation

- Score: 0.8104
- Authors: Bing Su, Mengyu Zhao, Yifeng Lin, Peng Jin
- Year / Venue: 2026 / Computing and Informatics
- DOI: 10.31577/cai_2026_2_458
- Sources: openalex, crossref
- Citations: 1
- URL: https://doi.org/10.31577/cai_2026_2_458
- Abstract: With the continuous advancement of medical technology, medical image segmentation plays a pivotal role in various aspects of disease diagnosis, treatment planning, and therapeutic evaluation. In recent years, methods based on U-shaped Convolutional Neural Networks (CNNs), U-shaped Transformer architectures, and U-shaped Mamba models have achieved significant progress in this domain, greatly promoting research and application development. Although CNNs excel at local feature extraction, they are inherently limited in modeling global dependencies due to the locality of convolution operations. Transformers address this limitation through self-attention mechanisms that capture long-range depende

### 15. Advances in attention mechanisms for medical image segmentation

- Score: 0.8095
- Authors: Jianpeng Zhang, Xiaomin Chen, Bing Yang, Qingbiao Guan, et al.
- Year / Venue: 2025 / Computer Science Review
- DOI: 10.1016/j.cosrev.2024.100721
- Sources: openalex
- Citations: 171
- URL: https://doi.org/10.1016/j.cosrev.2024.100721
- Abstract: Medical image segmentation plays an important role in computer-aided diagnosis. Attention mechanisms that distinguish important parts from irrelevant parts have been widely used in medical image segmentation tasks . This paper systematically reviews the basic principles of attention mechanisms and their applications in medical image segmentation. First, we review the basic concepts of attention mechanism and formulation. Second, we surveyed about 200 articles related to medical image segmentation, and divided them into three groups based on their attention mechanisms, Pre-Transformer attention, Transformer attention and Mamba-related attention. In each group, we deeply analyze the attention 

### 16. Lightweight 2D Medical Image Segmentation Via a Decoder Using Linear Deformable Convolution and Multi-Scale Self-Attention

- Score: 0.7994
- Authors: Le Zou, Xiangxu Bu, Fengling Jiang, Zhize Wu, et al.
- Year / Venue: 2025 / IEEE Journal of Biomedical and Health Informatics
- DOI: 10.1109/jbhi.2025.3583108
- Sources: openalex
- Citations: 2
- URL: https://doi.org/10.1109/jbhi.2025.3583108
- Abstract: High-precision medical image segmentation typically demands substantial computational resources, which poses a significant limitation to its practical deployment in resource-constrained regions. Consequently, the development of decoding mechanisms that are both computationally efficient and lightweight is imperative. However, the performance of medical image segmentation is frequently limited by the simplicity of decoder designs. Balancing the optimization of decoder architectures with the reduction of computational demands while maintaining high model accuracy remains a formidable challenge. In this study, we introduce LDMSD, a novel decoder that integrates linear deformable convolution and

### 17. Seed-net: structure-enhanced encoder-decoder network via dual-attention bridge for 2D medical image segmentation

- Score: 0.795
- Authors: Zikai Wang, Biyuan Li, Jinying Ma, Jiangtao Liu, et al.
- Year / Venue: 2026 / Physica Scripta
- DOI: 10.1088/1402-4896/ae8abe
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.1088/1402-4896/ae8abe
- Abstract: Abstract Medical image segmentation (MIS) plays a crucial role in clinical diagnosis and disease prediction. However, existing encoder-decoder segmentation networks still face several structural limitations. First, conventional downsampling operations may discard high-frequency boundary details, resulting in blurred lesion edges. Second, shallow convolutional features are easily affected by local background noise and scale-variable lesion structures, while deep semantic representations often lack effective long-range dependency modeling, limiting global contextual understanding. Third, highly compressed bottleneck features often mix lesion semantics with background artifacts, which enlarges 

### 18. Medical Image Segmentation via Cascaded Attention Decoding

- Score: 0.7914
- Authors: Md. Mostafijur Rahman, Radu Mărculescu
- Year / Venue: 2023 / IEEE Winter Conference on Applications of Computer Vision
- DOI: 10.1109/wacv56688.2023.00616
- Sources: openalex
- Citations: 458
- URL: https://doi.org/10.1109/wacv56688.2023.00616
- Abstract: Transformers have shown great promise in medical image segmentation due to their ability to capture long-range dependencies through self-attention. However, they lack the ability to learn the local (contextual) relations among pixels. Previous works try to overcome this problem by embedding convolutional layers either in the encoder or decoder modules of transformers thus ending up sometimes with inconsistent features. To address this issue, we propose a novel attention-based decoder, namely CASCaded Attention DEcoder (CASCADE), which leverages the multi-scale features of hierarchical vision transformers. CASCADE consists of i) an attention gate which fuses features with skip connections and

### 19. DMD: Dual attention fusion and multi-scale feature fusion decoding for medical image segmentation

- Score: 0.7904
- Authors: Xin Wang, Yanmin Niu
- Year / Venue: 2025 / Journal of Radiation Research and Applied Sciences
- DOI: 10.1016/j.jrras.2025.101873
- Sources: openalex
- Citations: 1
- URL: https://doi.org/10.1016/j.jrras.2025.101873
- Abstract: High-precision medical image segmentation plays a decisive role in clinical diagnosis, treatment efficacy evaluation, and the development of personalized therapeutic strategies. Although models based on U-Net and its variants have achieved remarkable results across various segmentation tasks, traditional convolutional neural networks (CNNs) are inherently limited by their local receptive fields, restricting their ability to capture long-range dependencies and global contextual information. In contrast, Transformer architectures leverage self-attention mechanisms to model global information, demonstrating superior performance in segmenting complex anatomical structures. However, most existing

### 20. EPHAD: Efficient Phased Hybrid Attention Decoder for Medical Image Segmentation

- Score: 0.7879
- Authors: Zihang Guo, Ji Qiu
- Year / Venue: 2026 / Lecture notes in electrical engineering
- DOI: 10.1007/978-981-95-3312-1_8
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.1007/978-981-95-3312-1_8

### 21. CSAFD: Cross-scale attention fusion decoder for medical image segmentation

- Score: 0.7879
- Authors: Xiaoshan Pan, Bin Yao, Yuzhao Liu, Jing GUO, et al.
- Year / Venue: 2026 / Signal Image and Video Processing
- DOI: 10.1007/s11760-026-05269-7
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.1007/s11760-026-05269-7

### 22. LWD: A Lightweight Decoder Leveraging Gated Attention and Cross-Group Convolution for Medical Image Segmentation

- Score: 0.7879
- Authors: Runkai Xi, K. L. Eddie Law
- Year / Venue: 2026 / Lecture notes in computer science
- DOI: 10.1007/978-3-032-31673-8_26
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.1007/978-3-032-31673-8_26

### 23. Partial Decoder Attention Network with Contour-weighted Loss Function for Data-Imbalance Medical Image Segmentation

- Score: 0.785
- Authors: Zhengyong Huang, Ning Jiang, Sun, Xingwen, Lihua Zhang, et al.
- Year / Venue: 2026 / arXiv (Cornell University)
- DOI: 10.48550/arxiv.2601.14338
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.48550/arxiv.2601.14338
- Abstract: Image segmentation is pivotal in medical image analysis, facilitating clinical diagnosis, treatment planning, and disease evaluation. Deep learning has significantly advanced automatic segmentation methodologies by providing superior modeling capability for complex structures and fine-grained anatomical regions. However, medical images often suffer from data imbalance issues, such as large volume disparities among organs or tissues, and uneven sample distributions across different anatomical structures. This imbalance tends to bias the model toward larger organs or more frequently represented structures, while overlooking smaller or less represented structures, thereby affecting the segmenta

### 24. CSKAN-U: Enhancing U-KAN with Dual Attention and 2D Mamba Decoder for Efficient Medical Image Segmentation

- Score: 0.785
- Authors: Jinyu Ma
- Year / Venue: 2025 / International Conference on Computational Intelligence and Communication Networks (CICN)
- DOI: 10.1109/cicn67655.2025.11368053
- Sources: openalex, crossref
- Citations: 0
- URL: https://doi.org/10.1109/cicn67655.2025.11368053
- Abstract: Medical image segmentation faces significant challenges in accurately delineating boundaries and preserving finegrained details, especially in complex anatomical structures and pathological regions. While U-KAN has demonstrated strong performance by integrating Kolmogorov-Arnold Networks (KAN) into the U-Net architecture, it still has two key limitations: insufficient feature recalibration across channels and space, and the inability of its convolutional decoder to model global context effectively due to fixed receptive fields. To address these issues, we propose CSKAN-U, a novel model that augments tokenized KAN with dual-channel and spatial attention mechanisms (the CSKAN block) and replac

### 25. U-Net Transformer: Self and Cross Attention for Medical Image Segmentation

- Score: 0.7794
- Authors: Olivier Petit, Nicolas Thome, Clément Rambour, Loic Themyr, et al.
- Year / Venue: 2021 / Lecture notes in computer science
- DOI: 10.1007/978-3-030-87589-3_28
- Sources: openalex
- Citations: 368
- URL: https://doi.org/10.1007/978-3-030-87589-3_28

### 26. MACMD: Multi-dilated Contextual Attention and Channel Mixer Decoding for Medical Image Segmentation

- Score: 0.775
- Authors: Maurya, Lalit, Liu, Honghai, Reyer Zwiggelaar A
- Year / Venue: 2025 / arXiv (Cornell University)
- DOI: 10.48550/arxiv.2511.05803
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.48550/arxiv.2511.05803
- Abstract: Medical image segmentation faces challenges due to variations in anatomical structures. While convolutional neural networks (CNNs) effectively capture local features, they struggle with modeling long-range dependencies. Transformers mitigate this issue with self-attention mechanisms but lack the ability to preserve local contextual information. State-of-the-art models primarily follow an encoder-decoder architecture, achieving notable success. However, two key limitations remain: (1) Shallow layers, which are closer to the input, capture fine-grained details but suffer from information loss as data propagates through deeper layers. (2) Inefficient integration of local details and global cont

### 27. Fine-Tuning SAM2 for Generalizable Polyp Segmentation with a Channel Attention-Enhanced Decoder

- Score: 0.77
- Authors: Yixiao Liu
- Year / Venue: 2025 / Advanced Medical Research
- DOI: 10.62836/amr.v4i1.311
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.62836/amr.v4i1.311
- Abstract: Polyp segmentation is a critical task in medical image analysis, particularly in colonoscopy, where it plays a vital role in the early detection and treatment of colorectal cancer. In recent years, advancements in deep learning, especially the application of Convolutional Neural Networks (CNNs) and Transformer models, have significantly improved segmentation performance. Despite these advancements, the generalizability of these models across different datasets is often limited. Recently, Meta released the Segment Anything Model 2 (SAM2), which has demonstrated exceptional performance in both video and image segmentation tasks. This paper aims to develop a universal polyp segmentation model b

### 28. LAEDNet: A Lightweight Attention Encoder–Decoder Network for ultrasound medical image segmentation

- Score: 0.7311
- Authors: Quan Zhou, Qianwen Wang, Yunchao Bao, Lingjun Kong, et al.
- Year / Venue: 2022 / Computers & Electrical Engineering
- DOI: 10.1016/j.compeleceng.2022.107777
- Sources: openalex
- Citations: 65
- URL: https://doi.org/10.1016/j.compeleceng.2022.107777

### 29. Multi-Scale Group Agent Attention-Based Graph Convolutional Decoding Networks for 2D Medical Image Segmentation

- Score: 0.7305
- Authors: Zhichao Wang, Lin Guo, Shuchang Zhao, Shiqing Zhang, et al.
- Year / Venue: 2024 / IEEE Journal of Biomedical and Health Informatics
- DOI: 10.1109/jbhi.2024.3523112
- Sources: openalex
- Citations: 18
- URL: https://doi.org/10.1109/jbhi.2024.3523112
- Abstract: Automated medical image segmentation plays a crucial role in assisting doctors in diagnosing diseases. Feature decoding is a critical yet challenging issue for medical image segmentation. To address this issue, this work proposes a novel feature decoding network, called multi-scale group agent attention-based graph convolutional decoding networks (MSGAA-GCDN), to learn local-global features in graph structures for 2D medical image segmentation. The proposed MSGAA-GCDN combines graph convolutional network (GCN) and a lightweight multi-scale group agent attention (MSGAA) mechanism to represent features globally and locally within a graph structure. Moreover, in skip connections a simple yet ef

### 30. Triple-residual enhanced convolution- and transformer-based hybrid encoder-decoder network for medical image segmentation

- Score: 0.7291
- Authors: Minghan Lin, Ziyang Wang
- Year / Venue: 2025 / Neurocomputing
- DOI: 10.1016/j.neucom.2025.131125
- Sources: crossref
- Citations: 7
- URL: https://doi.org/10.1016/j.neucom.2025.131125

### 31. Boundary-aware Context Neural Network for Medical Image Segmentation

- Score: 0.715
- Authors: Ruxin Wang, Shuyuan Chen, Chaojie Ji, Jianping Fan, et al.
- Year / Venue: 2020 / arXiv
- DOI: 10.1016/j.media.2022.102395
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2005.00966v1
- Abstract: Medical image segmentation can provide a reliable basis for further clinical analysis and disease diagnosis. The performance of medical image segmentation has been significantly advanced with the convolutional neural networks (CNNs). However, most existing CNNs-based methods often produce unsatisfactory segmentation mask without accurate object boundaries. This is caused by the limited context information and inadequate discriminative feature maps after consecutive pooling and convolution operations. In that the medical image is characterized by the high intra-class variation, inter-class indistinction and noise, extracting powerful context and aggregating discriminative features for fine-gr

### 32. UTNet: A Hybrid Transformer Architecture for Medical Image Segmentation

- Score: 0.7141
- Authors: Yunhe Gao, Mu Zhou, Dimitris Metaxas
- Year / Venue: 2021 / Lecture notes in computer science
- DOI: 10.1007/978-3-030-87199-4_6
- Sources: openalex
- Citations: 713
- URL: https://doi.org/10.1007/978-3-030-87199-4_6

### 33. A Novel 3D Decoder with Weighted and Learnable Triple Attention for 3D Microscopy Image Segmentation

- Score: 0.7137
- Authors: Siyavash Shabani, Sahar A. Mohammed, Bahram Parvin
- Year / Venue: 2025 / 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
- DOI: 10.1109/cvprw67362.2025.00456
- Sources: crossref
- Citations: 3
- URL: https://doi.org/10.1109/cvprw67362.2025.00456

### 34. PDAtt-Unet: Pyramid Dual-Decoder Attention Unet for Covid-19 infection segmentation from CT-scans

- Score: 0.6944
- Authors: Fares Bougourzi, Cosimo Distante, Fadi Dornaika, Abdelmalik Taleb-Ahmed
- Year / Venue: 2023 / Medical Image Analysis
- DOI: 10.1016/j.media.2023.102797
- Sources: crossref
- Citations: 149
- URL: https://doi.org/10.1016/j.media.2023.102797

### 35. Multi-decoder Networks for Semi-supervised Medical Image Segmentation

- Score: 0.6937
- Authors: Jianjun Zhang, Zhipeng Zhao, Yixin Chen, Hanqing Liu
- Year / Venue: 2023 / Proceedings of the 15th International Conference on Digital Image Processing
- DOI: 10.1145/3604078.3604098
- Sources: crossref
- Citations: 3
- URL: https://doi.org/10.1145/3604078.3604098

### 36. Multi-attention enhanced encoder–decoder network with hybrid transformer bottleneck for echocardiography image segmentation

- Score: 0.6929
- Authors: Saeed Chamani, Hamid Behnam
- Year / Venue: 2026 / Scientific Reports
- DOI: 10.1038/s41598-026-60699-0
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1038/s41598-026-60699-0

### 37. Hybrid CNN-Transformer Model with Attention-Based Decoding for Accurate Abdominal Organ Segmentation in Medical Imaging

- Score: 0.6904
- Authors: Naga Lakshmi, S. Jeya Shobana, M. V. Suriyaprakash, K. Joshna, et al.
- Year / Venue: 2025 / International Conference on Smart Technologies, Communication and Robotics (STCR)
- DOI: 10.1109/stcr62650.2025.11019937
- Sources: openalex
- Citations: 1
- URL: https://doi.org/10.1109/stcr62650.2025.11019937
- Abstract: Medical image segmentation is important in diagnosing diseases and planning treatments. Abdominal organ segmentation is difficult due to variations in shape, size, and position. Traditional methods struggle with unclear boundaries and overlapping structures. Deep learning models improve segmentation but often miss details or fail with complex shapes. A new deep learning model is designed to improve accuracy. It combines convolutional networks with transformers to capture both local and global features. The model extracts multi-scale features to preserve fine details. A transformer module processes global relationships between organs. An attention-based decoder refines segmentation masks to e

### 38. MGU-Net: a multiscale gate attention encoder-decoder network for medical image segmentation

- Score: 0.6894
- Authors: Yong Wan, Tao Lei, Qi Chen, Jian Su, et al.
- Year / Venue: 2023 / International Journal of Computer Applications in Technology
- DOI: 10.1504/ijcat.2023.132397
- Sources: openalex, crossref
- Citations: 2
- URL: https://doi.org/10.1504/ijcat.2023.132397
- Abstract: Medical image segmentation, the prerequisite of numerous clinical needs, has been significantly prospered from recent advances in encoder-decoder networks. However, uneven reflection of human organs and the subject's tremor and movement cause blurred edges in the image, which is difficult to segment. Hence and more details and context information are needed to resolve this problem. Most of the existing Unet-like architectures do not take into account the multiscale characteristics of medical images and do not make full use of the spatial information and channel information of feature maps, resulting in the loss of detail information. This paper proposes a Multiscale Gate Attention (MGU-Net) 

### 39. Tree-NET: Enhancing 2D Medical Image Segmentation Through Efficient Low-Level Feature Training

- Score: 0.685
- Authors: Orhan Demirci, Bulent Yilmaz
- Year / Venue: 2025 / arXiv
- DOI: 10.1007/s00521-026-12420-5
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2501.02140v2
- Abstract: This paper introduces Tree-NET, a novel framework for medical image segmentation that leverages bottleneck supervision to enhance both segmentation accuracy and computational efficiency. While previous studies have applied bottleneck feature supervision to segmentation tasks, it has typically been limited to the training phase, offering no computational benefits during inference. To the best of our knowledge, this is the first framework to employ dual bottleneck supervision for segmentation, leveraging latent space features at both the input and output stages. This approach reduces input and label dimensions with minimal parameter overhead while preserving accuracy. Tree-NET features a three

### 40. Uncertainty-Guided Hybrid Decoder for Medical Image Segmentation

- Score: 0.6829
- Authors: Lu Chen, Jiajie Mu
- Year / Venue: 2025 / 2025 IEEE International Conference on Pattern Recognition, Machine Vision and Artificial Intelligence (PRMVAI)
- DOI: 10.1109/prmvai65741.2025.11108450
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/prmvai65741.2025.11108450

### 41. CFPFormer: Cross Feature-Pyramid Transformer Decoder for Medical Image Segmentation

- Score: 0.6829
- Authors: Hongyi Cai, Mohammad Mahdinur Rahman, Wenzhen Dong, Jingyu Wu
- Year / Venue: 2025 / 2025 International Joint Conference on Neural Networks (IJCNN)
- DOI: 10.1109/ijcnn64981.2025.11228110
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/ijcnn64981.2025.11228110

### 42. Multi-Encoder Parse-Decoder Network for Sequential Medical Image Segmentation

- Score: 0.6827
- Authors: Dachuan Shi, Ruiyang Liu, Linmi Tao, Zuoxiang He, et al.
- Year / Venue: 2021 / 2021 IEEE International Conference on Image Processing (ICIP)
- DOI: 10.1109/icip42928.2021.9506463
- Sources: crossref
- Citations: 5
- URL: https://doi.org/10.1109/icip42928.2021.9506463

### 43. GMSRAD: A Global-Local Modulated Sub-Pixel Reconstruction Attention Decoder for Medical Image Segmentation

- Score: 0.6779
- Authors: Baoqun Wang, Chenxi Huang, Qinglang Guo, Hao Jin, et al.
- Year / Venue: 2026 / Lecture notes in computer science
- DOI: 10.1007/978-981-92-3513-1_5
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.1007/978-981-92-3513-1_5

### 44. A dual encoder-decoder model with multi-attention for segmentation of variable size lesions in 2D medical images

- Score: 0.6779
- Authors: Aasia Rehman, Sajid Yousuf Bhat
- Year / Venue: 2026 / Multimedia Tools and Applications
- DOI: 10.1007/s11042-026-21738-0
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.1007/s11042-026-21738-0

### 45. MSHAD: A Novel Multi-scale Convolutional Hybrid Attention Decoder for Medical Image Segmentation

- Score: 0.6779
- Authors: Xin Xiao, Kaigui Wu
- Year / Venue: 2026 / Lecture notes in computer science
- DOI: 10.1007/978-981-92-3420-2_20
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.1007/978-981-92-3420-2_20

### 46. MSAD-Swin UNETR: A Hybrid Transformer-CNN Model with Multiscale Attention Decoder for Accurate 3D Medical Image Segmentation

- Score: 0.6779
- Authors: Zainab Adnan Jwad, Israa Hadi Ali
- Year / Venue: 2026 / Communications in computer and information science
- DOI: 10.1007/978-3-032-25568-6_12
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.1007/978-3-032-25568-6_12

### 47. Rethinking Boundary Detection in Deep Learning-Based Medical Image Segmentation

- Score: 0.6779
- Authors: Yi Lin, Dong Zhang, Xiao Fang, Yufan Chen, et al.
- Year / Venue: 2025 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2505.04652v1
- Abstract: Medical image segmentation is a pivotal task within the realms of medical image analysis and computer vision. While current methods have shown promise in accurately segmenting major regions of interest, the precise segmentation of boundary areas remains challenging. In this study, we propose a novel network architecture named CTO, which combines Convolutional Neural Networks (CNNs), Vision Transformer (ViT) models, and explicit edge detection operators to tackle this challenge. CTO surpasses existing methods in terms of segmentation accuracy and strikes a better balance between accuracy and efficiency, without the need for additional data inputs or label injections. Specifically, CTO adheres

### 48. Medical image segmentation using boundary-enhanced guided packet rotation dual attention decoder network

- Score: 0.6758
- Authors: Hongchun Lu, Shengwei Tian, Long Yu, Yan Xing, et al.
- Year / Venue: 2021 / Technology and Health Care
- DOI: 10.3233/thc-202789
- Sources: openalex, crossref
- Citations: 3
- URL: https://doi.org/10.3233/thc-202789
- Abstract: BACKGROUND: The automatic segmentation of medical images is an important task in clinical applications. However, due to the complexity of the background of the organs, the unclear boundary, and the variable size of different organs, some of the features are lost during network learning, and the segmentation accuracy is low. OBJECTIVE: To address these issues, this prompted us to study whether it is possible to better preserve the deep feature information of the image and solve the problem of low segmentation caused by unclear image boundaries. METHODS: In this study, we (1) build a reliable deep learning network framework, named BGRANet,to improve the segmentation performance for medical ima

### 49. MAPNet: A Multi-scale Attention Pooling Network for Ultrasound Medical Image Segmentation

- Score: 0.6629
- Authors: Shixun Wang, Mengjiao Wang
- Year / Venue: 2023 / Unknown
- DOI: 10.21203/rs.3.rs-3721034/v1
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.21203/rs.3.rs-3721034/v1
- Abstract: Abstract Deep convolutional neural networks have achieved great success in the field of medical image segmentation and assisted diagnosis due to their excellent autonomous feature extraction capability and good feature representation ability. Traditional convolutional neural networks usually use standard convolutional layers to extract features. However, in medical image segmentation, a pixel may correspond to multiple different scale structures, such as lesion edges, organ boundaries, etc. Therefore, traditional convolutional layers may not be able to capture this complex feature. In this paper, we propose a multi-scale convolution (MC) which realizes more comprehensive feature extraction b

### 50. PCCA-Model: An Attention Module for Medical Image Segmentation

- Score: 0.6629
- Authors: Linjie Liu, Guanglei Wang, Yanlin Wu, Hongrui Wang, et al.
- Year / Venue: 2023 / Unknown
- DOI: 10.1364/opticaopen.21972767
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1364/opticaopen.21972767
- Abstract: Convolutional neural networks have been increasingly employed in the field of medical image segmentation. Based on this idea that the human visual cortex differs in terms of the size of the receptive field and can sense the stimulus location., we proposed the pyramid channel coordinate attention (PCCA) module to fuse multiscale features in the channel direction, aggregate local and global channel information, combine them with the location information in the spatial direction, and then integrate them into the existing semantic segmentation network. We conducted numerous experiments on the datasets, namely LiTS, ISIC-2018, and CX, and obtained state-of-the-art results.

### 51. Hybrid dilation and attention residual U-Net for medical image segmentation

- Score: 0.6609
- Authors: Zekun Wang, Yanni Zou, Peter Xiaoping Liu
- Year / Venue: 2021 / Computers in Biology and Medicine
- DOI: 10.1016/j.compbiomed.2021.104449
- Sources: openalex
- Citations: 160
- URL: https://doi.org/10.1016/j.compbiomed.2021.104449

### 52. Attention Mechanisms in Medical Image Segmentation: A Survey

- Score: 0.6579
- Authors: Yutong Xie, Bing Yang, Qingbiao Guan, Jianpeng Zhang, et al.
- Year / Venue: 2023 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2305.17937v1
- Abstract: Medical image segmentation plays an important role in computer-aided diagnosis. Attention mechanisms that distinguish important parts from irrelevant parts have been widely used in medical image segmentation tasks. This paper systematically reviews the basic principles of attention mechanisms and their applications in medical image segmentation. First, we review the basic concepts of attention mechanism and formulation. Second, we surveyed over 300 articles related to medical image segmentation, and divided them into two groups based on their attention mechanisms, non-Transformer attention and Transformer attention. In each group, we deeply analyze the attention mechanisms from three aspects

### 53. AEDN: Encoder-Decoder Network with Attention for Semantic Image Segmentation

- Score: 0.6529
- Authors: Shusheng Li, Liang Wan, Lu Tang, Zhining Zhang
- Year / Venue: 2022 / Unknown
- DOI: 10.21203/rs.3.rs-2119917/v1
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.21203/rs.3.rs-2119917/v1
- Abstract: Abstract Enhancing receptive fields using atrous convolution in the domain of semantic segmentation has been proven to be an effective method. However, atrous convolution exists ”grid effect” and does not capture long-range dependencies well, causing problems such as loss of information details and class confusion. To address the above problems, based on two proposed modules: Strip Spatial Attention Module (SSAM) and Adaptive Fusion Module (AFM), we designed a network with an encoder-decoder structure named AEDN. Specifically, we introduced horizontal and vertical strip modules to design a Strip Spatial Attention Module, which did not require association for each pixel, avoided useless codin

### 54. Learning With Context Feedback Loop for Robust Medical Image Segmentation

- Score: 0.645
- Authors: Kibrom Berihu Girum, Gilles Créhange, Alain Lalande
- Year / Venue: 2021 / arXiv
- DOI: 10.1109/tmi.2021.3060497
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2103.02844v1
- Abstract: Deep learning has successfully been leveraged for medical image segmentation. It employs convolutional neural networks (CNN) to learn distinctive image features from a defined pixel-wise objective function. However, this approach can lead to less output pixel interdependence producing incomplete and unrealistic segmentation results. In this paper, we present a fully automatic deep learning method for robust medical image segmentation by formulating the segmentation problem as a recurrent framework using two systems. The first one is a forward system of an encoder-decoder CNN that predicts the segmentation result from the input image. The predicted probabilistic output of the forward system i

### 55. Adaptive encoding and comprehensive attention decoding network for medical image segmentation

- Score: 0.6449
- Authors: Xin Shu, Aoping Zhang, Zhaoyang Xu, Feng Zhu, et al.
- Year / Venue: 2025 / Applied Soft Computing
- DOI: 10.1016/j.asoc.2025.112990
- Sources: openalex
- Citations: 12
- URL: https://doi.org/10.1016/j.asoc.2025.112990

### 56. RAR-U-NET: A Residual Encoder to Attention Decoder by Residual Connections Framework for Spine Segmentation Under Noisy Labels

- Score: 0.6413
- Authors: Ziyang Wang, Zhengdong Zhang, Irina Voiculescu
- Year / Venue: 2021 / 2021 IEEE International Conference on Image Processing (ICIP)
- DOI: 10.1109/icip42928.2021.9506085
- Sources: crossref
- Citations: 33
- URL: https://doi.org/10.1109/icip42928.2021.9506085

### 57. Mutual Consistency Learning for Semi-supervised Medical Image Segmentation

- Score: 0.6379
- Authors: Yicheng Wu, Zongyuan Ge, Donghao Zhang, Minfeng Xu, et al.
- Year / Venue: 2021 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2109.09960v4
- Abstract: In this paper, we propose a novel mutual consistency network (MC-Net+) to effectively exploit the unlabeled data for semi-supervised medical image segmentation. The MC-Net+ model is motivated by the observation that deep models trained with limited annotations are prone to output highly uncertain and easily mis-classified predictions in the ambiguous regions (e.g., adhesive edges or thin branches) for medical image segmentation. Leveraging these challenging samples can make the semi-supervised segmentation model training more effective. Therefore, our proposed MC-Net+ model consists of two new designs. First, the model contains one shared encoder and multiple slightly different decoders (i.e

### 58. Volumetric Attention for 3D Medical Image Segmentation and Detection

- Score: 0.635
- Authors: Xudong Wang, Shizhong Han, Yunqiang Chen, Dashan Gao, et al.
- Year / Venue: 2020 / arXiv
- DOI: 10.1007/978-3-030-32226-7_20
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2004.01997v1
- Abstract: A volumetric attention(VA) module for 3D medical image segmentation and detection is proposed. VA attention is inspired by recent advances in video processing, enables 2.5D networks to leverage context information along the z direction, and allows the use of pretrained 2D detection models when training data is limited, as is often the case for medical applications. Its integration in the Mask R-CNN is shown to enable state-of-the-art performance on the Liver Tumor Segmentation (LiTS) Challenge, outperforming the previous challenge winner by 3.9 points and achieving top performance on the LiTS leader board at the time of paper submission. Detection experiments on the DeepLesion dataset also s

### 59. A regressive encoder-decoder-based deep attention model for segmentation of fetal head in 2D-ultrasound images

- Score: 0.6341
- Authors: Somya Srivastava, Ankit Vidyarthi, Shikha Jain
- Year / Venue: 2023 / Image and Vision Computing
- DOI: 10.1016/j.imavis.2023.104725
- Sources: crossref
- Citations: 9
- URL: https://doi.org/10.1016/j.imavis.2023.104725

### 60. Danet: Dilated Attention Network for Medical Image Segmentation

- Score: 0.6329
- Authors: Reema Wangkheirakpam, Sudipta Roy, Khelchandra Thongam, Chiranjiv Chingangbam
- Year / Venue: 2022 / Unknown
- DOI: 10.2139/ssrn.4226726
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.4226726
- Abstract: Various forms of cancer exists worldwide, with colorectal cancer being the leading cause of cancer-related deaths. Colonoscopy has become the standard for diagnosing the presence of polyps in the colon. As colonoscopy is a human-dependent procedure, so the chances of miss-rate are often high. Early detection of polyps reduces the mortality rate and helps to decrease the incidence of colorectal cancer. With the advancement in deep learning, multiple methods have been developed to solve this task. In this study, we proposed a novel architecture called Dilated Attention Network (DANet) for automatic polyp segmentation. DANet is a simple encoder-decoder architecture which uses a pre-trained ResN

### 61. ALL-FABNET: Acute Lymphocytic Leukemia Segmentation Using a Flipping Attention Block Decoder-Encoder Network

- Score: 0.6233
- Authors: Ammar S. Al-Zubaidi, Mohammed Al-Mukhtar, Ammar Awni Abbas Baghdadi
- Year / Venue: 2024 / Journal of Image and Graphics
- DOI: 10.18178/joig.12.3.239-249
- Sources: crossref
- Citations: 6
- URL: https://doi.org/10.18178/joig.12.3.239-249
- Abstract: Acute Lymphoblastic Leukemia (ALL) is a malignant neoplasm defined by the abnormal proliferation of immature lymphocytes in the hematopoietic system, specifically in the blood or bone marrow. The efficacy of ALL treatment is closely linked to its timely identification. Currently, the first diagnosis of ALL involves clinicians laboriously and fallibly examining stained blood smear microscopy images. Recently, deep learning techniques in biomedical diagnostics, focusing on human-centric approaches, have emerged as a potent tool to aid clinicians in their decision-making processes. As a result, researchers have devised a multitude of computer-aided diagnostic methods to detect ALL in blood imag

### 62. An effective CNN and Transformer complementary network for medical image segmentation

- Score: 0.6191
- Authors: Feiniu Yuan, Zhengxiao Zhang, Zhijun Fang
- Year / Venue: 2022 / Pattern Recognition
- DOI: 10.1016/j.patcog.2022.109228
- Sources: openalex
- Citations: 569
- URL: https://doi.org/10.1016/j.patcog.2022.109228

### 63. MedP-CLIP: Medical CLIP with Region-Aware Prompt Integration

- Score: 0.615
- Authors: Jiahui Peng, He Yao, Jingwen Li, Yanzhou Su, et al.
- Year / Venue: 2026 / arXiv
- DOI: 10.1016/j.media.2026.104193
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2604.11197v3
- Abstract: Contrastive Language-Image Pre-training (CLIP) has demonstrated outstanding performance in global image understanding and zero-shot transfer through large-scale text-image alignment. However, the core of medical image analysis often lies in the fine-grained understanding of specific anatomical structures or lesion regions. Therefore, precisely comprehending region-of-interest (RoI) information provided by medical professionals or perception models becomes crucial. To address this need, we propose MedP-CLIP, a region-aware medical vision-language model (VLM). MedP-CLIP innovatively integrates medical prior knowledge and designs a feature-level region prompt integration mechanism, enabling it 

### 64. SegLite Decoder (SLDC): A Lightweight Decoder With Compressed Channel Attention for Real-Time Egocentric Hand Segmentation

- Score: 0.6129
- Authors: Xiaoxiao Zhang, Xin He, Kaile Wu, Yu Guo
- Year / Venue: 2026 / 2026 7th International Conference on Computer Vision, Image and Deep Learning (CVIDL)
- DOI: 10.1109/cvidl70130.2026.11637572
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/cvidl70130.2026.11637572

### 65. Multi-Path Residual Attention Backbone With Dynamic Normalization Transformer Decoder For Semantic Segmentation

- Score: 0.6029
- Authors: Kefan Zhu, Zhaoyan Li, Zhisheng Gao
- Year / Venue: 2025 / 2025 9th International Conference on Vision, Image and Signal Processing (ICVISP)
- DOI: 10.1109/icvisp68610.2025.11451679
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/icvisp68610.2025.11451679

### 66. CMAD: High-efficiency hybrid CNN–Mamba Attention Decoding network for medical image segmentation

- Score: 0.5979
- Authors: Yehang Li, Hui Zhao
- Year / Venue: 2026 / Signal Processing Image Communication
- DOI: 10.1016/j.image.2026.117731
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.1016/j.image.2026.117731

### 67. Context-Aware and Spectral Attention-Driven Decoding for single-domain generalizable medical image segmentation

- Score: 0.5979
- Authors: Fuxian Sui, Yifan Hu, Mengran Chi, Yongping Du
- Year / Venue: 2026 / Biomedical Signal Processing and Control
- DOI: 10.1016/j.bspc.2026.111508
- Sources: openalex
- Citations: 0
- URL: https://doi.org/10.1016/j.bspc.2026.111508

### 68. Fréchet Radiomic Distance (FRD): A Versatile Metric for Comparing Medical Imaging Datasets

- Score: 0.595
- Authors: Nicholas Konz, Richard Osuala, Preeti Verma, Yuwen Chen, et al.
- Year / Venue: 2024 / arXiv
- DOI: 10.1016/j.media.2026.103943
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2412.01496v2
- Abstract: Determining whether two sets of images belong to the same or different distributions or domains is a crucial task in modern medical image analysis and deep learning; for example, to evaluate the output quality of image generative models. Currently, metrics used for this task either rely on the (potentially biased) choice of some downstream task, such as segmentation, or adopt task-independent perceptual metrics (e.g., Fréchet Inception Distance/FID) from natural imaging, which we show insufficiently capture anatomical features. To this end, we introduce a new perceptual metric tailored for medical images, FRD (Fréchet Radiomic Distance), which utilizes standardized, clinically meaningful, an

### 69. Test-time generative augmentation for medical image segmentation

- Score: 0.595
- Authors: Xiao Ma, Yuhui Tao, Zetian Zhang, Yuhan Zhang, et al.
- Year / Venue: 2024 / arXiv
- DOI: 10.1016/j.media.2025.103902
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2406.17608v2
- Abstract: Medical image segmentation is critical for clinical diagnosis, treatment planning, and monitoring, yet segmentation models often struggle with uncertainties stemming from occlusions, ambiguous boundaries, and variations in imaging devices. Traditional test-time augmentation (TTA) techniques typically rely on predefined geometric and photometric transformations, limiting their adaptability and effectiveness in complex medical scenarios. In this study, we introduced Test-Time Generative Augmentation (TTGA), a novel augmentation strategy specifically tailored for medical image segmentation at inference time. Different from conventional augmentation strategies that suffer from excessive randomne

### 70. MambaMIM: Pre-training Mamba with State Space Token Interpolation and its Application to Medical Image Segmentation

- Score: 0.595
- Authors: Fenghe Tang, Bingkun Nian, Yingtai Li, Zihang Jiang, et al.
- Year / Venue: 2024 / arXiv
- DOI: 10.1016/j.media.2025.103606
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2408.08070v2
- Abstract: Recently, the state space model Mamba has demonstrated efficient long-sequence modeling capabilities, particularly for addressing long-sequence visual tasks in 3D medical imaging. However, existing generative self-supervised learning methods have not yet fully unleashed Mamba's potential for handling long-range dependencies because they overlook the inherent causal properties of state space sequences in masked modeling. To address this challenge, we propose a general-purpose pre-training framework called MambaMIM, a masked image modeling method based on a novel TOKen-Interpolation strategy (TOKI) for the selective structure state space sequence, which learns causal relationships of state spa

### 71. HiDiff: Hybrid Diffusion Framework for Medical Image Segmentation

- Score: 0.595
- Authors: Tao Chen, Chenhui Wang, Zhihao Chen, Yiming Lei, et al.
- Year / Venue: 2024 / arXiv
- DOI: 10.1109/tmi.2024.3424471
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2407.03548v1
- Abstract: Medical image segmentation has been significantly advanced with the rapid development of deep learning (DL) techniques. Existing DL-based segmentation models are typically discriminative; i.e., they aim to learn a mapping from the input image to segmentation masks. However, these discriminative methods neglect the underlying data distribution and intrinsic class characteristics, suffering from unstable feature space. In this work, we propose to complement discriminative segmentation methods with the knowledge of underlying data distribution from generative models. To that end, we propose a novel hybrid diffusion framework for medical image segmentation, termed HiDiff, which can synergize the

### 72. MSAD: efficient multiscale attention decoder for semiconductor image segmentation

- Score: 0.5929
- Authors: Abhishek Goyal, Girish Kumar Ajmera, Arvind Nayak
- Year / Venue: 2026 / Metrology, Inspection, and Process Control XL
- DOI: 10.1117/12.3089370
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1117/12.3089370

### 73. Semantic feature decoupling and contrastive boundary-guided attention for colorectal polyp segmentation: a comparative evaluation of decoder designs (Preprint)

- Score: 0.5929
- Authors: Shenghe Liu, Mingzheng Zhao, Dongchenlei Wang, Zihe Feng, et al.
- Year / Venue: 2026 / Unknown
- DOI: 10.2196/preprints.113258
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2196/preprints.113258
- Abstract: BACKGROUND Comparisons of decoder modules for automated polyp segmentation are complicated by differences in encoder backbones, training protocols, and data splits. This study evaluates decoder modules under matched conditions and investigates a decoder combining semantic feature decoupling with boundary-guided attention. OBJECTIVE To evaluate decoder modules for colorectal polyp segmentation under matched conditions and develop a semantic decoupling and boundary-guided attention framework. METHODS Ten decoder modules were compared with a plain convolutional baseline using a fixed Twins-PCPVT-Small encoder and matched training settings. We developed ConBPoly, combining a Semantic Information

### 74. Spatial Plaid Attention Decoder for Semantic Segmentation

- Score: 0.5929
- Authors: Abolfazl Meyarian, Xiaohui Yuan, Zhinan Qiao
- Year / Venue: 2024 / 2024 IEEE International Conference on Image Processing (ICIP)
- DOI: 10.1109/icip51287.2024.10648129
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.1109/icip51287.2024.10648129

### 75. AD2Former: Alternate Encoder and Dual Decoder Networks Utilizing CNN-Transformer for Medical Image Segmentation

- Score: 0.5902
- Authors: Xinyu Guo, Hongkun Sun, Weigang Wang
- Year / Venue: 2024 / Unknown
- DOI: 10.2139/ssrn.4886268
- Sources: crossref
- Citations: 2
- URL: https://doi.org/10.2139/ssrn.4886268

### 76. Multi-Scale Dual-Attention Frequency Fusion for Joint Segmentation and Deformable Medical Image Registration

- Score: 0.5902
- Authors: Hongchao Zhou, Shunbo Hu
- Year / Venue: 2024 / Unknown
- DOI: 10.2139/ssrn.5031719
- Sources: crossref
- Citations: 2
- URL: https://doi.org/10.2139/ssrn.5031719

### 77. GCA-ResUNet: Medical Image Segmentation Using Grouped Coordinate Attention

- Score: 0.5857
- Authors: Shang Gao
- Year / Venue: 2026 / Unknown
- DOI: 10.2139/ssrn.5996278
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.5996278

### 78. Cross-dimensional transfer learning in medical image segmentation with deep learning

- Score: 0.585
- Authors: Hicham Messaoudi, Ahror Belaid, Douraied Ben Salem, Pierre-Henri Conze
- Year / Venue: 2023 / arXiv
- DOI: 10.1016/j.media.2023.102868
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2307.15872v1
- Abstract: Over the last decade, convolutional neural networks have emerged and advanced the state-of-the-art in various image analysis and computer vision applications. The performance of 2D image classification networks is constantly improving and being trained on databases made of millions of natural images. However, progress in medical image analysis has been hindered by limited annotated data and acquisition constraints. These limitations are even more pronounced given the volumetry of medical imaging data. In this paper, we introduce an efficient way to transfer the efficiency of a 2D classification network trained on natural images to 2D, 3D uni- and multi-modal medical image segmentation applic

### 79. Weakly-Supervised 3D Medical Image Segmentation using Geometric Prior and Contrastive Similarity

- Score: 0.585
- Authors: Hao Du, Qihua Dong, Yan Xu, Jing Liao
- Year / Venue: 2023 / arXiv
- DOI: 10.1109/tmi.2023.3269523
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2302.02125v1
- Abstract: Medical image segmentation is almost the most important pre-processing procedure in computer-aided diagnosis but is also a very challenging task due to the complex shapes of segments and various artifacts caused by medical imaging, (i.e., low-contrast tissues, and non-homogenous textures). In this paper, we propose a simple yet effective segmentation framework that incorporates the geometric prior and contrastive similarity into the weakly-supervised segmentation framework in a loss-based fashion. The proposed geometric prior built on point cloud provides meticulous geometry to the weakly-supervised segmentation proposal, which serves as better supervision than the inherent property of the b

### 80. Segment Anything Model for Medical Image Analysis: an Experimental Study

- Score: 0.585
- Authors: Maciej A. Mazurowski, Haoyu Dong, Hanxue Gu, Jichen Yang, et al.
- Year / Venue: 2023 / arXiv
- DOI: 10.1016/j.media.2023.102918
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2304.10517v3
- Abstract: Training segmentation models for medical images continues to be challenging due to the limited availability of data annotations. Segment Anything Model (SAM) is a foundation model that is intended to segment user-defined objects of interest in an interactive manner. While the performance on natural images is impressive, medical image domains pose their own set of challenges. Here, we perform an extensive evaluation of SAM's ability to segment medical images on a collection of 19 medical imaging datasets from various modalities and anatomies. We report the following findings: (1) SAM's performance based on single prompts highly varies depending on the dataset and the task, from IoU=0.1135 for

### 81. Segment Anything Model for Medical Images?

- Score: 0.585
- Authors: Yuhao Huang, Xin Yang, Lian Liu, Han Zhou, et al.
- Year / Venue: 2023 / arXiv
- DOI: 10.1016/j.media.2023.103061
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2304.14660v7
- Abstract: The Segment Anything Model (SAM) is the first foundation model for general image segmentation. It has achieved impressive results on various natural image segmentation tasks. However, medical image segmentation (MIS) is more challenging because of the complex modalities, fine anatomical structures, uncertain and complex object boundaries, and wide-range object scales. To fully validate SAM's performance on medical data, we collected and sorted 53 open-source datasets and built a large medical segmentation dataset with 18 modalities, 84 objects, 125 object-modality paired targets, 1050K 2D images, and 6033K masks. We comprehensively analyzed different models and strategies on the so-called CO

### 82. X-Net: a dual encoding–decoding method in medical image segmentation

- Score: 0.5811
- Authors: Yuanyuan Li, Ziyu Wang, Yin Li, Zhiqin Zhu, et al.
- Year / Venue: 2021 / The Visual Computer
- DOI: 10.1007/s00371-021-02328-7
- Sources: openalex
- Citations: 161
- URL: https://doi.org/10.1007/s00371-021-02328-7

### 83. Dual-Decoder Data Decoupling Training for Semi-Supervised Medical Image Segmentation

- Score: 0.5811
- Authors: Xin Zhang, Bing Wang, Tai feng Huang, Shuo Yang, et al.
- Year / Venue: 2024 / Unknown
- DOI: 10.2139/ssrn.4880596
- Sources: crossref
- Citations: 1
- URL: https://doi.org/10.2139/ssrn.4880596

### 84. Graph Convolutional Networks with Spatial and Channel Attention for Medical Image Segmentation

- Score: 0.5811
- Authors: Shuo Dai, Xueyan Liu, Lishan Qiao
- Year / Venue: 2024 / Unknown
- DOI: 10.2139/ssrn.4768653
- Sources: crossref
- Citations: 1
- URL: https://doi.org/10.2139/ssrn.4768653

### 85. DeepTriNet: A Tri-Level Attention Based DeepLabv3+ Architecture for Semantic Segmentation of Satellite Images

- Score: 0.5779
- Authors: Tareque Bashar Ovi, Shakil Mosharrof, Nomaiya Bashree, Md Shofiqul Islam, et al.
- Year / Venue: 2023 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2310.06848v1
- Abstract: The segmentation of satellite images is crucial in remote sensing applications. Existing methods face challenges in recognizing small-scale objects in satellite images for semantic segmentation primarily due to ignoring the low-level characteristics of the underlying network and due to containing distinct amounts of information by different feature maps. Thus, in this research, a tri-level attention-based DeepLabv3+ architecture (DeepTriNet) is proposed for the semantic segmentation of satellite images. The proposed hybrid method combines squeeze-and-excitation networks (SENets) and tri-level attention units (TAUs) with the vanilla DeepLabv3+ architecture, where the TAUs are used to bridge t

### 86. Implicit Anatomical Rendering for Medical Image Segmentation with Stochastic Experts

- Score: 0.5779
- Authors: Chenyu You, Weicheng Dai, Yifei Min, Lawrence Staib, et al.
- Year / Venue: 2023 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2304.03209v2
- Abstract: Integrating high-level semantically correlated contents and low-level anatomical features is of central importance in medical image segmentation. Towards this end, recent deep learning-based medical segmentation methods have shown great promise in better modeling such information. However, convolution operators for medical segmentation typically operate on regular grids, which inherently blur the high-frequency regions, i.e., boundary regions. In this work, we propose MORSE, a generic implicit neural rendering framework designed at an anatomical level to assist learning in medical image segmentation. Our method is motivated by the fact that implicit neural representation has been shown to be

### 87. BayeSeg: Bayesian Modeling for Medical Image Segmentation with Interpretable Generalizability

- Score: 0.5779
- Authors: Shangqi Gao, Hangqi Zhou, Yibo Gao, Xiahai Zhuang
- Year / Venue: 2023 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2303.01710v1
- Abstract: Due to the cross-domain distribution shift aroused from diverse medical imaging systems, many deep learning segmentation methods fail to perform well on unseen data, which limits their real-world applicability. Recent works have shown the benefits of extracting domain-invariant representations on domain generalization. However, the interpretability of domain-invariant features remains a great challenge. To address this problem, we propose an interpretable Bayesian framework (BayeSeg) through Bayesian modeling of image and label statistics to enhance model generalizability for medical image segmentation. Specifically, we first decompose an image into a spatial-correlated variable and a spatia

### 88. A residual dense vision transformer for medical image super-resolution with segmentation-based perceptual loss fine-tuning

- Score: 0.5779
- Authors: Jin Zhu, Guang Yang, Pietro Lio
- Year / Venue: 2023 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2302.11184v2
- Abstract: Super-resolution plays an essential role in medical imaging because it provides an alternative way to achieve high spatial resolutions and image quality with no extra acquisition costs. In the past few decades, the rapid development of deep neural networks has promoted super-resolution performance with novel network architectures, loss functions and evaluation metrics. Specifically, vision transformers dominate a broad range of computer vision tasks, but challenges still exist when applying them to low-level medical image processing tasks. This paper proposes an efficient vision transformer with residual dense connections and local feature fusion to achieve efficient single-image super-resol

### 89. Medsd-Di: A Medical Image Segmentation Model Grounded in Sam Distillation and Bidirectional Decoder

- Score: 0.5757
- Authors: Huilei Huang, Quanjin Liu, Benzheng Wei, Yihao Yang, et al.
- Year / Venue: 2025 / Unknown
- DOI: 10.2139/ssrn.5143615
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.5143615

### 90. A Semi-Supervised Medical Image Segmentation Framework Based on Pseudo-Label Discrepancy and Multi-Decoder Collaboration

- Score: 0.5757
- Authors: Dongen Guo, Xin Lu, Xufan Chen, Junze Guo, et al.
- Year / Venue: 2025 / Unknown
- DOI: 10.2139/ssrn.5397011
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.5397011

### 91. Anomaly Detection-Inspired Few-Shot Medical Image Segmentation Through Self-Supervision With Supervoxels

- Score: 0.575
- Authors: Stine Hansen, Srishti Gautam, Robert Jenssen, Michael Kampffmeyer
- Year / Venue: 2022 / arXiv
- DOI: 10.1016/j.media.2022.102385
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2203.02048v1
- Abstract: Recent work has shown that label-efficient few-shot learning through self-supervision can achieve promising medical image segmentation results. However, few-shot segmentation models typically rely on prototype representations of the semantic classes, resulting in a loss of local information that can degrade performance. This is particularly problematic for the typically large and highly heterogeneous background class in medical image segmentation problems. Previous works have attempted to address this issue by learning additional prototypes for each class, but since the prototypes are based on a limited number of slices, we argue that this ad-hoc solution is insufficient to capture the backg

### 92. CapsNet for Medical Image Segmentation

- Score: 0.5679
- Authors: Minh Tran, Viet-Khoa Vo-Ho, Kyle Quinn, Hien Nguyen, et al.
- Year / Venue: 2022 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2203.08948v1
- Abstract: Convolutional Neural Networks (CNNs) have been successful in solving tasks in computer vision including medical image segmentation due to their ability to automatically extract features from unstructured data. However, CNNs are sensitive to rotation and affine transformation and their success relies on huge-scale labeled datasets capturing various input variations. This network paradigm has posed challenges at scale because acquiring annotated data for medical segmentation is expensive, and strict privacy regulations. Furthermore, visual representation learning with CNNs has its own flaws, e.g., it is arguable that the pooling layer in traditional CNNs tends to discard positional information

### 93. Segmentation Ability Map: Interpret deep features for medical image segmentation

- Score: 0.5679
- Authors: Sheng He, Yanfang Feng, P. Ellen Grant, Yangming Ou
- Year / Venue: 2022 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2212.09206v1
- Abstract: Deep convolutional neural networks (CNNs) have been widely used for medical image segmentation. In most studies, only the output layer is exploited to compute the final segmentation results and the hidden representations of the deep learned features have not been well understood. In this paper, we propose a prototype segmentation (ProtoSeg) method to compute a binary segmentation map based on deep features. We measure the segmentation abilities of the features by computing the Dice between the feature segmentation map and ground-truth, named as the segmentation ability score (SA score for short). The corresponding SA score can quantify the segmentation abilities of deep features in different

### 94. Scribble-Supervised Medical Image Segmentation via Dual-Branch Network and Dynamically Mixed Pseudo Labels Supervision

- Score: 0.5679
- Authors: Xiangde Luo, Minhao Hu, Wenjun Liao, Shuwei Zhai, et al.
- Year / Venue: 2022 / arXiv
- DOI: None
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2203.02106v1
- Abstract: Medical image segmentation plays an irreplaceable role in computer-assisted diagnosis, treatment planning, and following-up. Collecting and annotating a large-scale dataset is crucial to training a powerful segmentation model, but producing high-quality segmentation masks is an expensive and time-consuming procedure. Recently, weakly-supervised learning that uses sparse annotations (points, scribbles, bounding boxes) for network training has achieved encouraging performance and shown the potential for annotation cost reduction. However, due to the limited supervision signal of sparse annotations, it is still challenging to employ them for networks training directly. In this work, we propose 

### 95. A Fused Attention-Based Hybrid Model for Semi-Supervised Medical Image Segmentation

- Score: 0.5657
- Authors: Masum Shah Junayed, Sheida Nabavi
- Year / Venue: 2024 / Unknown
- DOI: 10.2139/ssrn.5033738
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.5033738

### 96. Segmentation-Driven Attention Method for Enhanced Medical Image Registration

- Score: 0.5657
- Authors: Sahar Almahfouz Nasser, Mohit Meena, Garweet Sresth, Amit Sethi
- Year / Venue: 2024 / Unknown
- DOI: 10.2139/ssrn.4835615
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.4835615

### 97. Edge Harmony Attention Network for Semi-Supervised Medical Image Segmentation

- Score: 0.5657
- Authors: Manti Lu, Yang Liu, Yunzhu Chen, Feng Yang
- Year / Venue: 2024 / Unknown
- DOI: 10.2139/ssrn.4944587
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.4944587

### 98. Data synthesis and adversarial networks: A review and meta-analysis in cancer imaging

- Score: 0.565
- Authors: Richard Osuala, Kaisar Kushibar, Lidia Garrucho, Akis Linardos, et al.
- Year / Venue: 2021 / arXiv
- DOI: 10.1016/j.media.2022.102704
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2107.09543v2
- Abstract: Despite technological and medical advances, the detection, interpretation, and treatment of cancer based on imaging data continue to pose significant challenges. These include inter-observer variability, class imbalance, dataset shifts, inter- and intra-tumour heterogeneity, malignancy determination, and treatment effect uncertainty. Given the recent advancements in Generative Adversarial Networks (GANs), data synthesis, and adversarial training, we assess the potential of these technologies to address a number of key challenges of cancer imaging. We categorise these challenges into (a) data scarcity and imbalance, (b) data access and privacy, (c) data annotation and segmentation, (d) cancer

### 99. Deep multi-scale attentional features for medical image segmentation

- Score: 0.5639
- Authors: Sahadev Poudel, Sang-Woong Lee
- Year / Venue: 2021 / Applied Soft Computing
- DOI: 10.1016/j.asoc.2021.107445
- Sources: openalex
- Citations: 74
- URL: https://doi.org/10.1016/j.asoc.2021.107445

### 100. Test-Time Adaptable Neural Networks for Robust Medical Image Segmentation

- Score: 0.555
- Authors: Neerav Karani, Ertunc Erdil, Krishna Chaitanya, Ender Konukoglu
- Year / Venue: 2020 / arXiv
- DOI: 10.1016/j.media.2020.101907
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2004.04668v4
- Abstract: Convolutional Neural Networks (CNNs) work very well for supervised learning problems when the training dataset is representative of the variations expected to be encountered at test time. In medical image segmentation, this premise is violated when there is a mismatch between training and test images in terms of their acquisition details, such as the scanner model or the protocol. Remarkable performance degradation of CNNs in this scenario is well documented in the literature. To address this problem, we design the segmentation CNN as a concatenation of two sub-networks: a relatively shallow image normalization CNN, followed by a deep CNN that segments the normalized image. We train both the

### 101. Uncertainty-aware multi-view co-training for semi-supervised medical image segmentation and domain adaptation

- Score: 0.555
- Authors: Yingda Xia, Dong Yang, Zhiding Yu, Fengze Liu, et al.
- Year / Venue: 2020 / arXiv
- DOI: 10.1016/j.media.2020.101766
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2006.16806v1
- Abstract: Although having achieved great success in medical image segmentation, deep learning-based approaches usually require large amounts of well-annotated data, which can be extremely expensive in the field of medical image analysis. Unlabeled data, on the other hand, is much easier to acquire. Semi-supervised learning and unsupervised domain adaptation both take the advantage of unlabeled data, and they are closely related to each other. In this paper, we propose uncertainty-aware multi-view co-training (UMCT), a unified framework that addresses these two tasks for volumetric medical image segmentation. Our framework is capable of efficiently utilizing unlabeled data for better performance. We fi

### 102. DAB-UNET: Dual Attention Block UNET Segmentation for Diabetic Retinopathy Utilizing an Encoder-Decoder Residual

- Score: 0.5387
- Authors: Haithem Kareem Abass
- Year / Venue: 2025 / Journal of Image and Graphics
- DOI: 10.18178/joig.13.3.253-266
- Sources: crossref
- Citations: 4
- URL: https://doi.org/10.18178/joig.13.3.253-266

### 103. GAN-GA: A Generative Model based on Genetic Algorithm for Medical Image Generation

- Score: 0.505
- Authors: M. AbdulRazek, G. Khoriba, M. Belal
- Year / Venue: 2023 / arXiv
- DOI: 10.3389/978-2-8325-1231-9
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2401.00314v1
- Abstract: Medical imaging is an essential tool for diagnosing and treating diseases. However, lacking medical images can lead to inaccurate diagnoses and ineffective treatments. Generative models offer a promising solution for addressing medical image shortage problems due to their ability to generate new data from existing datasets and detect anomalies in this data. Data augmentation with position augmentation methods like scaling, cropping, flipping, padding, rotation, and translation could lead to more overfitting in domains with little data, such as medical image data. This paper proposes the GAN-GA, a generative model optimized by embedding a genetic algorithm. The proposed model enhances image f

### 104. Privacy Preserving Image Registration

- Score: 0.495
- Authors: Riccardo Taiello, Melek Önen, Francesco Capano, Olivier Humbert, et al.
- Year / Venue: 2022 / arXiv
- DOI: 10.1007/978-3-031-16446-0_13
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2205.10120v7
- Abstract: Image registration is a key task in medical imaging applications, allowing to represent medical images in a common spatial reference frame. Current approaches to image registration are generally based on the assumption that the content of the images is usually accessible in clear form, from which the spatial transformation is subsequently estimated. This common assumption may not be met in practical applications, since the sensitive nature of medical images may ultimately require their analysis under privacy constraints, preventing to openly share the image content.In this work, we formulate the problem of image registration under a privacy preserving regime, where images are assumed to be c

### 105. Evaluating Procedures for Establishing Generative Adversarial Network-based Stochastic Image Models in Medical Imaging

- Score: 0.495
- Authors: Varun A. Kelkar, Dimitrios S. Gotsis, Frank J. Brooks, Kyle J. Myers, et al.
- Year / Venue: 2022 / arXiv
- DOI: 10.1117/12.2612893
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2204.03547v1
- Abstract: Modern generative models, such as generative adversarial networks (GANs), hold tremendous promise for several areas of medical imaging, such as unconditional medical image synthesis, image restoration, reconstruction and translation, and optimization of imaging systems. However, procedures for establishing stochastic image models (SIMs) using GANs remain generic and do not address specific issues relevant to medical imaging. In this work, canonical SIMs that simulate realistic vessels in angiography images are employed to evaluate procedures for establishing SIMs using GANs. The GAN-based SIM is compared to the canonical SIM based on its ability to reproduce those statistics that are meaning

### 106. AAU-net: An Adaptive Attention U-net for Breast Lesions Segmentation in Ultrasound Images

- Score: 0.495
- Authors: Gongping Chen, Yu Dai, Jianxun Zhang, Moi Hoon Yap
- Year / Venue: 2022 / arXiv
- DOI: 10.1109/tmi.2022.3226268
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2204.12077v3
- Abstract: Various deep learning methods have been proposed to segment breast lesion from ultrasound images. However, similar intensity distributions, variable tumor morphology and blurred boundaries present challenges for breast lesions segmentation, especially for malignant tumors with irregular shapes. Considering the complexity of ultrasound images, we develop an adaptive attention U-net (AAU-net) to segment breast lesions automatically and stably from ultrasound images. Specifically, we introduce a hybrid adaptive attention module, which mainly consists of a channel self-attention block and a spatial self-attention block, to replace the traditional convolution operation. Compared with the conventi

### 107. Multi-Objective Dual Simplex-Mesh Based Deformable Image Registration for 3D Medical Images -- Proof of Concept

- Score: 0.495
- Authors: Georgios Andreadis, Peter A. N. Bosman, Tanja Alderliesten
- Year / Venue: 2022 / arXiv
- DOI: 10.1117/12.2605498
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2202.11001v1
- Abstract: Reliably and physically accurately transferring information between images through deformable image registration with large anatomical differences is an open challenge in medical image analysis. Most existing methods have two key shortcomings: first, they require extensive up-front parameter tuning to each specific registration problem, and second, they have difficulty capturing large deformations and content mismatches between images. There have however been developments that have laid the foundation for potential solutions to both shortcomings. Towards the first shortcoming, a multi-objective optimization approach using the Real-Valued Gene-pool Optimal Mixing Evolutionary Algorithm (RV-GO

### 108. A General Two-branch Decoder Architecture for Improving Encoder-decoder Image Segmentation Models

- Score: 0.4883
- Authors: Sijie Hu, Fabien Bonardi, Samia Bouchafa, Désiré Sidibé
- Year / Venue: 2022 / Proceedings of the 17th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications
- DOI: 10.5220/0010818800003124
- Sources: crossref
- Citations: 1
- URL: https://doi.org/10.5220/0010818800003124

### 109. TransMorph: Transformer for unsupervised medical image registration

- Score: 0.485
- Authors: Junyu Chen, Eric C. Frey, Yufan He, William P. Segars, et al.
- Year / Venue: 2021 / arXiv
- DOI: 10.1016/j.media.2022.102615
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2111.10480v6
- Abstract: In the last decade, convolutional neural networks (ConvNets) have been a major focus of research in medical image analysis. However, the performances of ConvNets may be limited by a lack of explicit consideration of the long-range spatial relationships in an image. Recently Vision Transformer architectures have been proposed to address the shortcomings of ConvNets and have produced state-of-the-art performances in many medical imaging applications. Transformers may be a strong candidate for image registration because their substantially larger receptive field enables a more precise comprehension of the spatial correspondence between moving and fixed images. Here, we present TransMorph, a hyb

### 110. Improved Polyp Segmentation with Attention and Attention-Bilstm Using Encoder-Decoder Model: A Step Towards Better Performance

- Score: 0.4757
- Authors: Mayuri Gupta, Ashish Mishra
- Year / Venue: 2023 / Unknown
- DOI: 10.2139/ssrn.4625567
- Sources: crossref
- Citations: 0
- URL: https://doi.org/10.2139/ssrn.4625567

### 111. A distance-based loss for smooth and continuous skin layer segmentation in optoacoustic images

- Score: 0.475
- Authors: Stefan Gerl, Johannes C. Paetzold, Hailong He, Ivan Ezhov, et al.
- Year / Venue: 2020 / arXiv
- DOI: 10.1007/978-3-030-59725-2_30
- Sources: arxiv
- Citations: 0
- URL: http://arxiv.org/abs/2007.05324v1
- Abstract: Raster-scan optoacoustic mesoscopy (RSOM) is a powerful, non-invasive optical imaging technique for functional, anatomical, and molecular skin and tissue analysis. However, both the manual and the automated analysis of such images are challenging, because the RSOM images have very low contrast, poor signal to noise ratio, and systematic overlaps between the absorption spectra of melanin and hemoglobin. Nonetheless, the segmentation of the epidermis layer is a crucial step for many downstream medical and diagnostic tasks, such as vessel segmentation or monitoring of cancer progression. We propose a novel, shape-specific loss function that overcomes discontinuous segmentations and achieves smo

## Coverage Note

This report supports screening and does not guarantee exhaustive coverage. Final scholarly use requires human review.
