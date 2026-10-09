# Stability sweep recovery — 2026-09-28

## Evidence

- Prior two runs stopped producing output after epoch 90 validation inference (808/808). Last completed checkpoint was epoch 89, not 90.
- The old validation path retained the entire validation set predictions/targets, expanded ~52.95 million labels to int64, and invoked sklearn confusion_matrix on CPU. This is a costly post-inference path; no stack was captured from the original stalled process, so its precise failure cause is unproven.
- On recovery, a SIGUSR1 Python stack identified CPU torchvision rotation (`_gen_affine_grid`) during training data loading. GPU presence/memory allocation alone is not a liveness check.
- Previous claims that num_workers was proven responsible, or that save_interval prevented latest.pth updates, were incorrect. latest.pth is saved every completed epoch.

## Changes

- engine.py now accumulates an int64 2x2 confusion matrix per batch in validation and final evaluation, preserving >= thresholds and pooled metric definitions. It does not retain the entire prediction set.
- Four tests verify exact equivalence to sklearn across thresholds, uneven batches, and single-class cases. Entire suite: 26 passed.
- The stability launcher sets PyTorch intra/inter-op thread counts to 1; launch environment also limits OMP/MKL/OpenBLAS threads to 1.
- SIGUSR1 dumps Python stacks for future diagnosis without terminating training.
- Logger handlers are removed between jobs to prevent later runs from polluting earlier logs. Shell wrapper records exit code.

## Continuation

Resume the Wave v1 seed43 checkpoint at completed epoch89, then continue the original six-job queue. num_workers remains 0 as in the most recent attempted recovery. Original checkpoints are preserved.

Seed43 includes interruptions and a loader-worker change. Historical checkpoints do not save DataLoader generator state, so this resume is not bitwise equivalent to uninterrupted training. Record this qualification when summarizing across seeds; do not infer statistical stability solely from this interrupted run.

## Observed recovery

At 2026-09-28 14:37:44 CST, epoch100 completed (including full validation metric reporting) and latest.pth persisted epoch100, micro_step3000, best epoch100, val loss0.7943978746523066. Pooled Dice0.8837595905416665, IoU0.791728719954261. Epoch101 began and reached iteration20. Thus both former failure boundary90 and next reporting boundary100 were crossed, with eleven completed epochs after the old epoch89 checkpoint. tmux session: medseg_stability; console: results/stability_serial_v1_console.log.
