# Seed 43 training throughput audit — 2026-09-29

## Observed throughput

For EGE-Dual differential LR, parsing `train.info.log` from epoch start (`iter:0`) to validation completion gives:

| Run | Epochs measured | Mean / median epoch | 90th percentile | Epochs over 90 s |
| --- | ---: | ---: | ---: | ---: |
| Seed 42 retrain | 300 | 28.9 / 26 s | 30 s | 4 |
| Seed 43 stability | 208 | 61.7 / 51 s | 85 s | 17 |

The 808-image validation progress bar takes roughly 24–28 s per seed-43 epoch, versus 17–18 s in seed 42. Validation is executed every epoch; `val_interval=10` only gates full metric logging in `engine.py`. The validation loader uses `batch_size=1`, so each epoch performs 808 individual inference calls.

## Causes and checks

- Seed 42 used `num_workers=4`; the seed-43 launcher sets `num_workers=0` and `torch.set_num_threads(1)` with `OMP_NUM_THREADS=1`. Its CPU image loading, normalization, augmentation, and resizing therefore run serially in the training process. This leaves the GPU idle between operations.
- A read-only 640-image training-loader benchmark on the same data and machine took 5.758/5.988 s with 0 workers and 2.161/2.084 s with 4 workers (both orders tested). This demonstrates a roughly 2.7× data preparation difference, not a 2.7× end-to-end training speedup.
- The project, dataset, checkpoints, and logs are on an NFS mount. During this audit, the training process and independent read-only commands simultaneously entered `rpc_wait_bit_killable` disk wait; the GPU showed 0% utilization. These intermittent storage stalls explain some long-tail epochs and make ETA uncertain. The exact split of NFS time between image reads and result writes has not been profiled.
- The RTX 4090 had about 12.2/24.6 GiB allocated. In a 16 s one-second-sampled window, GPU utilization repeatedly dropped to 0% and briefly reached 100%, consistent with an input/CPU/I/O constrained pipeline rather than a consistently compute-saturated GPU.

## Intervention at user request (2026-09-29 11:10–11:14 UTC)

The user requested changing the bottleneck settings now and resuming from a checkpoint. Training was interrupted during epoch 225, after the complete epoch-224 checkpoint had been written. The checkpoint was loaded and checked (`epoch=224`, `micro_step=6720`). Its original NFS run directory and console logs were retained under `_nfs_backup_...` names.

The 5,389-file, 251,898,689-byte dataset was copied to `/home/featurize/medseg_seed43_local/data/isic2018` and `rsync --checksum --dry-run` found no differences. The current Dual run directory and console logs were copied and verified on local disk; the original workspace paths now point to those local copies. The upcoming SCF and EGE baseline output paths also point to local disk. A separate `medseg_seed43_sync` tmux session mirrors local results to `results/seed43_local_mirror_20260929` every ten minutes and once after the queue ends.

`num_workers` was changed from 0 to 4 for the remaining Dual epochs and upcoming SCF and EGE baseline runs. The serial queue restarted at Dual and resumed from epoch 224. The first complete resumed epoch (225) took about 25 s from its first train batch to validation completion, versus the previous 51 s median. More epochs are needed to estimate sustained throughput.

The validation loader remains batch size 1, validation still runs each epoch, and model/checkpoint selection semantics are unchanged. Worker count changed after epoch 224 and the checkpoint does not preserve DataLoader generator state, so this seed-43 continuation is not a bitwise-equivalent uninterrupted run. The earlier seed-43 interruptions already imposed the same qualification.

## Validation batch equivalence probe (2026-09-29)

At the user's request, `research/probe_val_batch_equivalence.py` evaluated the same saved Dual epoch-224 checkpoint on all 808 validation images with batch sizes 1 and 8, using the same deterministic validation transform and model in eval mode. The batch-1 pass took 19.37 s and the batch-8 pass 3.91 s. However, mean validation loss changed from 0.9030088755 to 0.9031408704 (absolute difference 0.000132), 2,189 pixels crossed the 0.5 threshold, and pooled Dice changed from 0.8784734834 to 0.8784649549. Maximum per-pixel probability difference was 0.007265; gate output was unchanged.

A 64-image repeat check returned *exactly identical* batch-1 outputs and loss on two passes. On the first batch of eight, batch loss versus the mean of eight separate per-image loss calls differed by only 1.3e-8. Thus the observed difference comes from batch-size-dependent model numerics, not a changing loss reduction or a nondeterministic repeat of batch-1 evaluation. Since validation loss selects the best checkpoint, the queue initially retained validation batch size 1 pending the user's decision.

## Validation batch changed with user authorization (2026-09-29 11:33 UTC)

After reviewing the measured difference, the user explicitly requested switching to batch 8. The Dual process was stopped and a complete epoch-259 checkpoint was verified and copied to `latest_before_valbatch8_epoch259.pth`. The seed-43 queue restarted from epoch 260 with training batch 64 and validation batch 8. The validation loader now has 101 batches for 808 images; the first resumed validation took about 3 seconds. The final `test_one_epoch` call retains a separate batch-1 loader so its image export and final report still use per-image inference. The same validation batch-8 setting is queued for SCF and EGE baseline.

Validation loss values and pooled metrics before and after epoch 259 use slightly different inference numerics, and the best-checkpoint selection compares those values. This continuation must be identified as a mixed validation-batch experiment. It remains a legacy development-split run, not an independent test.
