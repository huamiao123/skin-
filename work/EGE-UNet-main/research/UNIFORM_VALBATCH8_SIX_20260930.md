# Six uniform validation-batch runs (2026-09-30)

The user requested six fresh runs, in this serial order:

1. Wave v1, seed 43
2. EGE-Dual differential LR, seed 43
3. EGE-UNet baseline, seed 42
4. Wave v1, seed 42
5. EGE-Dual differential LR, seed 42
6. EGE-Dual + SCF differential LR, seed 42

All six use train batch 64, validation batch 8, final per-image evaluation batch 1,
4 data-loading workers, 300 epochs, gradient accumulation 1, and the existing
architecture-specific optimizer settings. `research/run_uniform_valbatch8_six.py`
defines the exact model configurations. The queue uses separate Python processes
for each job. `DONE.json` marks a successfully completed job; on relaunch, finished
jobs are skipped and an unfinished job resumes from its `latest.pth` checkpoint.

The active results path is `results/rebatch8_20260930/`, symlinked to local disk
at `/home/featurize/medseg_seed43_local/results/rebatch8_20260930/` to avoid NFS
stalls. `medseg_uniform6_20260930` is the training tmux session and
`medseg_uniform6_sync_20260930` mirrors results every 10 minutes to
`results/rebatch8_20260930_mirror/`. The console log is
`results/rebatch8_20260930/queue_console.log`. Earlier results are untouched.

The local data link is `data/isic2018` (1,886 train images and 808 validation
images). These runs still use the legacy train/validation split. The final
`#----------Testing----------#` report reuses V_dev; it is not an independent
test-set result. The seed-43 SCF and baseline runs from 2026-09-29 already used
validation batch 8 throughout and are not part of this six-job rerun.
