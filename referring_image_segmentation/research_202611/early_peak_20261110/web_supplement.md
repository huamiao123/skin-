# Web supplement

Retrieved 2026-11-10. These sources were found or verified through web searches and primary pages; they are not merged into API papers.json.

## guo2017: On Calibration of Modern Neural Networks

- Source: https://proceedings.mlr.press/v70/guo17a/guo17a.pdf
- Type: ICML paper
- Verified passage / claim: Figure 3 and section 3: NLL can worsen while classification error improves.
- Applicability limit: Classification evidence; our combined BCE+Dice loss needs decomposition.

## mehrtash2020: Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation

- Source: https://arxiv.org/html/1911.13273v1
- Type: TMI paper, accessible 2019 preprint
- Verified passage / claim: Brain/heart/prostate segmentation; compares CE and Dice calibration; VI-A uses validation DSC checkpoint selection and plateau LR reduction.
- Applicability limit: Different organs/models; ours uses combined BCE+Dice.

## kumar2022: Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution

- Source: https://arxiv.org/html/2202.10054v1
- Type: ICLR paper
- Verified passage / claim: LP-FT can mitigate feature distortion under studied classification distribution shifts.
- Applicability limit: Does not establish catastrophic forgetting in MambaVision/ISIC; our fine-tuning outperforms frozen control.

## li2018: Explicit Inductive Bias for Transfer Learning with Convolutional Networks

- Source: https://proceedings.mlr.press/v80/li18a.html
- Type: ICML paper
- Verified passage / claim: L2-SP regularizes weights towards pretrained initialization.
- Applicability limit: Ordinary AdamW weight decay is not this regularizer; benefit here untested.

## budget2020: Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints

- Source: https://arxiv.org/html/1905.04753v3
- Type: Research paper
- Verified passage / claim: Budget-aware LR schedules compared on classification, detection and segmentation; truncating a long schedule differs from a schedule matched to a short budget.
- Applicability limit: Does not prove our encoder needs a particular decay horizon.

## dominic2023: Improving Data-Efficiency and Robustness of Medical Imaging Segmentation Using Inpainting-Based Self-Supervised Learning

- Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC9951871/
- Type: Medical segmentation paper
- Verified passage / claim: Appendix B compares transfer choices and LRs; immediate encoder fine-tuning at 1e-3 was preferred in their setup; too small LR also underperformed.
- Applicability limit: Counterevidence to universally prescribing warmup or ever-smaller LR; absolute LR not portable.

## gu2025: How to build the best medical image segmentation algorithm using foundation models: a comprehensive empirical study with Segment Anything Model

- Source: https://arxiv.org/abs/2404.09957
- Type: MELBA paper, 2024 preprint
- Verified passage / claim: 18 adaptation combinations on 17 medical datasets; parameter-efficient adaptation strong in their comparisons.
- Applicability limit: SAM-specific study, not proof adapters outperform our MambaVision baseline.

## rousseau2025: Post hoc calibration of medical segmentation models

- Source: https://link.springer.com/article/10.1007/s42452-025-06587-0
- Type: Medical segmentation paper
- Verified passage / claim: Calibration can improve without a matching Dice improvement; some methods/tasks lose Dice.
- Applicability limit: Calibration diagnosis is not a guaranteed segmentation remedy.

## huang2016: Deep Networks with Stochastic Depth

- Source: https://arxiv.org/abs/1603.09382
- Type: ECCV paper
- Verified passage / claim: Random residual branch dropping during training is a regularization/training technique.
- Applicability limit: Original ResNet study; no guaranteed benefit for this small-data fine-tuning.

## pytorch_eval: PyTorch Autograd mechanics: Evaluation Mode

- Source: https://docs.pytorch.org/docs/stable/notes/autograd
- Type: Official documentation
- Verified passage / claim: eval is orthogonal to gradient disabling; training-mode-dependent modules change behavior.
- Applicability limit: Current web docs checked against local Torch 2.2.2 and actual model.

## timm_drop: timm DropPath implementation

- Source: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
- Type: Official source
- Verified passage / claim: DropPath returns identity when training=False.
- Applicability limit: Local installed timm 1.0.22 source also checked; runtime_mode_audit.json confirms.

## nnunet_epochs: nnU-Net trainer defaults

- Source: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
- Type: Official source
- Verified passage / claim: Default 1000 epochs, 250 iterations/epoch; epoch counts cannot be compared across recipes directly.
- Applicability limit: Not an argument to train our transfer model for 1000 epochs.

Search families: calibration NLL vs accuracy; medical segmentation confidence calibration; fine-tuning feature distortion; pretrained weight regularization; budget-aware schedules; medical fine-tuning empirical studies; nnU-Net iteration defaults; PyTorch eval semantics; timm DropPath training behavior.

Access limitations: PMC7704933 open redirected to a browser challenge, so its arXiv v1 full text was checked instead. Requested arXiv1911.13273v3 does not exist; v1 is the accessible author manuscript. GitHub semantic_segmentation directory request failed; no claim relies on that failed fetch. Forum reports were discovered but not used as technical evidence.
