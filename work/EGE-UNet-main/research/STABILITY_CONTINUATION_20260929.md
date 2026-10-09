# Stability sweep continuation — 2026-09-29

The six-job seed 43/44 queue was resumed in tmux session `medseg_stability` using
`research/launch_stability_serial_v1.sh`. Output is appended to
`results/stability_resume_20260929_console.log`; the launcher writes an exit code
when the queue ends. The queue order is Wave v1, Dual differential LR, and
Dual+SCF for seed 43, followed by the same three models for seed 44.

The local machine has an RTX 4090 with 24,564 MiB VRAM. The stale
`data/isic2018` symlink to `/root/isic2018_data` was repointed to the local
`../../isic2018_data` copy. Its first image and mask SHA-256 values match the
frozen manifest, and the loader sees 1,886 train and 808 validation pairs.
Training dependencies were installed in the current Python environment.
The relevant regression tests passed: 11 passed, 9 warnings.

The Wave seed 43 `latest.pth` checkpoint was epoch 105, micro_step 3150 before
resumption. A copy was saved as
`latest_before_resume_20260929_epoch105.pth`. At 2026-09-29 04:29 UTC, the
resumed process had completed epoch 106, and `latest.pth` recorded epoch 106,
micro_step 3180, validation loss 0.8020183398102475. GPU training was active.

This is the legacy 1,886/808 split stability sweep, not a completed
`research_v1` manifest-split experiment. Seed 43 includes interruptions and
does not represent bitwise-equivalent uninterrupted training.

## Scope change during the run

The user requested completing seed 43 only and cancelling seed 44. The queue
was stopped after Wave seed 43 completed epoch 221; its checkpoint was copied
to `latest_before_seed44_cancel_epoch221.pth`. The queue script now contains
only seed 43 jobs. Restarting it resumes Wave from the latest complete epoch,
then runs Dual differential LR and Dual+SCF for seed 43. No seed 44 job is
scheduled. This adds another interruption to the seed 43 reproducibility
qualification above.

## Added EGE baseline seed 43

The user added EGE baseline seed 43 to the stability comparison. It is queued
after the Wave, Dual differential LR, and Dual+SCF seed 43 jobs, so GPU training
remains serial. `research/launch_ege_baseline_seed43_after_stability.sh` waits
for the `medseg_stability` tmux session to end, then runs
`research/run_ege_baseline_seed43.py` with 300 epochs and the same seed-43
batch size, worker count, and validation cadence. Output is recorded in
`results/stability_ege_baseline_seed43_console.log`.

## Stopped at user request (2026-09-29 06:35 UTC)

The user requested stopping training. The queued baseline tmux session was
cancelled, and the running Wave seed 43 process received SIGINT. The stability
launcher recorded `END_UTC=2026-09-29T06:35:36Z EXIT_CODE=130`. No training
process or tmux training session remains. Wave seed 43's `latest.pth` records
epoch 298; the interruption occurred before completion of the planned 300
epochs. Dual differential LR, Dual+SCF, and EGE baseline seed 43 did not start.

## Resumed at user request (2026-09-29 07:11 UTC)

The user requested continuing training. Before restart, the epoch 298 Wave
checkpoint was copied to `checkpoints/latest_before_resume_20260929_epoch298.pth`
and verified byte-for-byte. The `medseg_stability` serial queue and the
`medseg_baseline43_queued` watcher were restarted. The queue contains only
seed 43: Wave, Dual differential LR, Dual+SCF, then EGE baseline. This remains
the legacy 1886/808 development split experiment.

## Faster local continuation at user request (2026-09-29 11:14 UTC)

After Wave completed, Dual differential LR was stopped during epoch 225 and
resumed from its complete epoch-224 checkpoint. The dataset and active output
paths were moved to verified byte-identical local disk copies, with original
NFS files retained as backups and a ten-minute NFS mirror. DataLoader workers
changed from 0 to 4 for the remaining Dual, SCF, and EGE baseline runs. The
queue now starts at Dual; Wave remains complete. The exact paths, checks, and
performance measurements are in `PERFORMANCE_AUDIT_20260929.md`.

## Validation batch change (2026-09-29 11:33 UTC)

At the user's request, the queue was resumed from Dual's complete epoch-259
checkpoint with validation batch size 8, while training batch size remains 64.
The final per-image evaluation keeps batch size 1. The same setting is queued
for SCF and EGE baseline. The measured small batch-size effect and the mixed
checkpoint-selection qualification are documented in
`PERFORMANCE_AUDIT_20260929.md`.
