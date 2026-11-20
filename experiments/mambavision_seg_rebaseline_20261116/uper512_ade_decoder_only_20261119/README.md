# MambaVision-S with an ADE20K-pretrained UPer512 decoder

Four independently trained skin-lesion segmentation models. The MambaVision-S encoder starts from ImageNet-1K. The decoder loads 72 compatible tensors and buffers from NVIDIA's ADE20K segmentation checkpoint. A newly initialized one-logit output replaces its 150-class classifier. The whole decoder and encoder stages 3 and 4 are trainable. The input is 256 x 256 and the two deep-stage windows are 16 and 8.

| Dataset | Train / validation / test | UPer64 test Dice (%) | UPer512 test Dice (%) | Difference (pp) | Selected epoch / epochs run |
|---|---|---:|---:|---:|---:|
| ISIC2018 | 1886 / 762 / 867 | 89.8869 | 89.3681 | -0.5188 | 21 / 46 |
| PH2 | 120 / 40 / 40 | 95.4676 | 95.4981 | +0.0305 | 45 / 70 |
| Waterloo | 124 / 41 / 41 | 92.9444 | 93.9964 | +1.0519 | 66 / 91 |
| HAM10000 | 8012 / 1002 / 1001 | 94.6961 | 94.6328 | -0.0633 | 6 / 31 |

Dice is the arithmetic mean of per-image foreground Dice on the held-out test split, at a probability threshold of 0.5. The earliest checkpoint with the highest validation Dice is evaluated once. All UPer512 runs use seed 42, batch 64, a 100-epoch cosine schedule and patience 25. The PH2 and HAM10000 UPer64 comparisons use historical 200-epoch schedules. The comparison changes decoder width and initialization together.

## Files

- Each dataset directory contains the actual configuration, decoder loading checks, training history, selected checkpoint metadata and per-image test scores.
- [Test summary](test_summary.csv) and [best checkpoint SHA-256 digests](best_checkpoints.json).
- The [training entry point](train.py), [queue](queue.sh), [split protocol](queue_protocol.json) and [decoder checkpoint provenance](checkpoint_source.json) record the training environment. The original project uses machine-specific paths and a CUDA Mamba environment.
- ISIC2018, PH2 and Waterloo splits and UPer64 controls are under [four_dataset_matched_20261119](../four_dataset_matched_20261119/). The HAM10000 split and control are under [ham10000_window16_20g_20261119](../ham10000_window16_20g_20261119/). Split CSVs preserve the training row order and byte-level checksums.
- Training applies paired random horizontal flip, vertical flip and arbitrary-angle rotation, each independently with probability 0.5. Validation and test use fixed preprocessing.
- HAM10000 resumed from epoch 20 on a fresh instance with its training state restored and input cache rebuilt. Its original first-step diagnostic was restored from the persistent copy for final archiving.

## Checkpoint provenance

The four named checkpoint digests and byte sizes identify the trained models. NVIDIA's upstream ADE20K initialization can be retrieved with [fetch_decoder.py](fetch_decoder.py).
