# Seed42 corrected transformer reruns — 2026-10-09

User requested background retraining of ImageNet pretrained Swin-Unet and PTU
with encoder depths [2,2,6,2], following the corrected EGE seed42 recipe.

Both: legacy 1886 train / 808 development validation, 256x256, seed42, 300 epochs,
physical train batch64 (no accumulation; Swin uses activation checkpointing), validation batch8, final evaluation
batch1, four workers, FP32, AdamW lr1e-4/wd0.01/betas(0.9,0.999)/eps1e-8,
CosineAnnealingLR T_max300/eta_min1e-5, gradient clipping1.0.
Select best by minimum mean validation batch loss, matching EGE's convention.
All reported DSC/IoU are pooled pixel metrics on V_dev, not independent test.

Augmentation uses frozen copies of corrected EGE utils/dataset: H/V flip each
p0.5, rotation p0.5 with angle re-sampled each call, bilinear image and nearest
mask geometry, normalization and resize256. The legacy source masks already
contain gray edge pixels; these source labels are preserved, not binarized for
training. Hard metrics use target>=0.5.

Swin uses historical embed96, depths2262, decoder[2,2,2,1], window8,
drop_path0.2; official Swin-T ImageNet weights and decoder reverse mapping,
relative position bias interpolation. Load audit confirms 301/354 keys loaded.
PTU uses embed[64,128,256,512], encoder2262, decoder222, heads[2,4,8,16],
SR[4,2,1,1], drop_path0.1, random initialization. Both retain their architectural
single final BCE+Dice loss (no EGE five-head deep supervision).

Both passed one real train batch64 forward/backward/optimizer update and
eight-image validation before launch. Preflight weights are never saved as
training initialization; each real job reinitializes using seed42.

Local execution/data/results: /home/featurize/medseg_transformer_seed42_20261009.
5388 image/mask files were checked byte-identical against work/isic2018_data.
Code and data hash inventories are stored in the local execution directory.
The work directory links expose live logs/results without NFS writes per batch.

Background session: medseg_transformer42_20261009. Sequential order: Swin, PTU.
Each epoch writes atomic latest.pth with optimizer/scheduler and Python/NumPy/
torch/CUDA/DataLoader RNG state, plus best.pth, history.json, status.json.
Re-launching run_queue.py resumes committed epochs and skips completed runs.
DONE.json is only written after 300 epochs and final best checkpoint evaluation.

Initial Swin continuous run ran out of memory on its second batch before any epoch commit. The failed attempt is retained in swin_pretrained.log. Built-in activation checkpointing was enabled, then two real batch64 updates and validation passed. PTU was paused after committed epochs and resumes its complete RNG/optimizer checkpoint when queued again. Initial launcher was renamed from queue.py to run_queue.py to avoid a stdlib module name collision. No completed legacy results are overwritten.
