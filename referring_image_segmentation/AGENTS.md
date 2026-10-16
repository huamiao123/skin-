# Project experiment rules

- Every experiment must report both per-sample mean (macro) and pooled foreground Dice/IoU. Never label either simply Dice/IoU without defining aggregation.
- Also report macro and pooled pixel precision/recall, Prec@0.5/0.6/0.7/0.8/0.9 (sample IoU >= threshold), mean boundary F1, mean HD95 in pixels and normalized by the image diagonal, and per-sample records. On ISIC add mean threshold Jaccard (IoU < 0.65 becomes zero).
- Use the common `evaluation.py` definitions and document any benchmark-specific exceptions before running an experiment.
- Freeze the protocol before training. Preserve original completed artifacts, source, checkpoints and metrics in `stage1_visual/`; write retrospective extended metrics into `evaluations/`.
- Preserve minimum sample-weighted validation loss selection for the archived stage-1/stage-2 comparison. Stage 3 prospectively uses maximum validation mean Dice for both decoders across new seeds 43/44/45, retaining best-loss and epoch-300 exports as secondary analyses. Do not select metrics or epochs retrospectively to favor a method.
- Report seeds, split, preprocessing, trainable/total parameters and evaluation resolution. The existing 1886/808 split is development validation, not an independent test or a verified patient-disjoint split.
- Change one experimental factor at a time; do not launch additional model families, text encoders, fusion or seed sweeps without an evidence-based next-step decision.
