# Stage28: late-stage DropPath during MambaVision fine-tuning

Prospective protocol, 2026-11-10. This is a training-policy ablation, not a new architecture or novelty claim. Existing completed experiment artifacts remain unchanged.

## Motivation and decision
Stage27 coverage supervision produced -0.0150 percentage points validation macro Dice versus its paired control (88.7960% vs 88.8110%). Severe undercoverage (per-image recall<0.5) was 22 vs21, and the three recurring cases moved in different directions. Do not promote this auxiliary head as an established improvement or add more modules based on it.
The two original fine-tuning seeds show validation BCE increasing from 0.1958 to0.5937 and0.1764 to0.5805 between best-Dice and last; soft Dice loss slightly improves. Among incorrect pixels, confidence>0.99 rises from31.27% to76.13% and25.07% to76.59%. This supports confidence deterioration, not a proof that DropPath fixes it.
Stage22 explicitly kept the encoder in eval to isolate weight adaptation. Runtime inspection confirms12 late DropPath modules are disabled, with configured probabilities0.070588...0.2. We now isolate that previously fixed training choice.

## Arms and single experimental factor
- control: MambaVision-S late levels2/3 trainable, frozen stem/levels0/1, same Swin pixel decoder; all encoder modules eval.
- drop_path: identical architecture, parameters and initialization, but enable ONLY DropPath modules belonging to levels2/3 while training. Their original probabilities remain unchanged. All other encoder modules, all frozen prefix modules and all BatchNorm buffers remain eval. Evaluation disables every DropPath.
- No auxiliary head, auxiliary loss, changed fusion, LR schedule adjustment, augmentation, decoder warmup or new data.
- Both arms have45,989,760 trainable encoder parameters and500,321 adapter/decoder parameters;50,640,905 total.
- Seed42; identical official pretrained model and seed42 decoder. Full model state must match before optimization; evaluation logits must match at initialization. During training the candidate is intentionally stochastic.

## Data, optimization and selection
- Reuse identity_partition.json unchanged: train1886, validation762, eligible heldout867. Training never reads heldout images/masks or evaluates test scores. Identity screening does not prove patient independence.
-256x256, existing normalization and geometric augmentations; preserve grayscale training targets and BCE+softDice objective. Evaluation truth/foreground threshold0.5.
-300 epochs, effective batch64,30 updates/epoch including tail. Preserve original300-epoch cosine factor and initial encoderLR1e-4, decoderLR1e-3. AdamW wd0.01, existing betas/eps; decoder and encoder gradient clipping separately at1.
- BF16 forward, FP32 parameters/loss/optimizer. Use original batch64 if it passes memory checks; microbatch fallback only for OOM and disclose because stochastic microbatch sampling can differ.
- Data order/geometry use a dedicated generator with seed42+epoch*100003; DropPath random draws cannot change sampled data or augmentation. Only the candidate uses forward stochasticity. Save/restore global Python/NumPy/Torch/CUDA RNG states.
- Keep GPU prefix caching; all trainable suffix features recomputed on every optimizer step and validation. No cached trainable features used.
- Primary checkpoint: maximum validation per-image mean foreground Dice, earliest tie; retain best-loss and epoch300 as secondary. No early stopping or switching selection based on new results.
- Record macro and pooled foreground Dice/IoU, precision/recall, Prec@IoU0.5/0.6/0.7/0.8/0.9, thresholdJaccard0.65, boundaryF1, HD95 pixels/diagonal, per-image rows via existing evaluation definitions.
- Add read-only evaluation diagnostics: BCE and softDice terms, mean predicted-label confidence on incorrect pixels, fraction of incorrect pixels with confidence>0.99. Denominators and definitions fixed; these are not checkpoint-selection criteria.
- Final comparisons include all762 paired cases, fixed GT area quartiles, recall<0.5 counts, recurring IDs305/581/791. These are development diagnostics, not new heldout hypotheses or independent significance evidence.
- Image-bootstrap interval is exploratory; it does not represent seed variability or patient-level uncertainty. No model is claimed superior from a tiny one-seed increase.

## Required verification and execution
- Mode audit: exactly12 late DropPath modules enabled only in candidate training; frozen prefix/BN unchanged; evaluation disables all.
- Equal initial state/logits in evaluation; training outputs stochastic only for candidate; RNG-restored stochastic forward exact.
- Positive finite gradients and changes in each intended trainable stage, unchanged frozen weights/buffers.
- Reuse established checkpoint replay checks: model/optimizer/RNG restoration exact; scalar loss and decoder update exact. Installed selective_scan backward has known non-bitwise behavior: encoder replay relative-to-updateL2<=0.01 and maximum absolute difference<=1e-4.
- Two-epoch smoke with a real process stop after epoch1 and resume at epoch2. Preserve first-epoch history and check completion before formal launch.
- Store source hashes; completed/ongoing sources immutable. Formal run starts only after all smoke and resume checks pass.
- Pair runs sequentially within each batch to fit one RTX4090; reserve safe GPU memory.
- Automatic post-training analysis only. No automatic LR/seed/module sweep: review this result before another intervention.

## Commands
Use /home/waas/mambavision_stage22/env/bin/python -B -u run_pipeline.py from this directory.
The pipeline logs smoke_first.log, smoke_resume.log, formal.log and analysis.log, and writes pipeline_failure.json on failure.
