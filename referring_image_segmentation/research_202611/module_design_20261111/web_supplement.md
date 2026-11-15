# Primary-source supplement

| Work | Source | Evidence inspected | Relevance |
|---|---|---|---|
| HMT-UNet | https://arxiv.org/html/2408.11289v1 | paper method and official model code | direct MambaVision medical predecessor |
| MVSeg | https://www.sciencedirect.com/science/article/pii/S1077314226001542 | publisher abstract and method preview only | direct MambaVision prediction-guided nearest work; full code not confirmed |
| PraNet | https://arxiv.org/abs/2006.11392 | paper abstract and official forward implementation | reverse-attention residual refinement |
| UACANet | https://arxiv.org/abs/2107.02368 | paper abstract and official UACA implementation | foreground background uncertainty context; confident-error limitation |
| DCA | https://arxiv.org/html/2303.17696v1 | paper method and official DCA implementation | cross-scale feature bridge |
| EMCAD | https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html | conference paper page and official decoders.py | multi-kernel depthwise refinement and gated skips |
| MEGANet | https://arxiv.org/abs/2309.03329 | paper abstract and official EGA implementation | edge/reverse guidance boundary-focused alternative |
| PraNet-V2 | https://arxiv.org/html/2504.10986v2 | paper method and official binary_seg/lib/pranet.py | dual foreground/background supervision and logit fusion |
| EGE-UNet | https://arxiv.org/abs/2307.08473 | abstract and official repository description only | skin lesion grouped prediction-guided bridges |

API records remain separate. No complete-reproduction claim. Retrieval not exhaustive.
