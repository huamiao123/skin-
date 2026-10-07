# LocalContour Phase 3: controlled contour selection

This directory implements the 2026-10-07 Phase-3 protocol in `protocols/LocalContour_Phase3_下一轮实验计划书_2026-10-07.txt`. It is a development experiment on ISIC 2017, not an independent test result.

**Current result: HOLD.** The controlled 18-model run is complete. On locked dev100, TG_A−N0_A was +0.000396 Dice (95% image-paired CI −0.000952 to +0.001758), below the +0.001 gate; TG_D−the cal50-selected S96_D was −0.001020 (CI −0.003786 to +0.001367). TG_A also increased correct-to-incorrect node rate by 0.0351 relative to N0_A, beyond the preregistered +0.005 guardrail. See `results_phase3/controlled/development_report.md` and `results_phase3/controlled/decision.json`. Official test600 remains sealed.

## Data roles

`prepare_phase3.py` froze image-level fit1800, stop200, cal50, dev100, and sealed official test600 identities in `splits/phase3_manifest.json`. The public training metadata has no patient identifier, so patient-level separation cannot be asserted. Frozen CNN and boundary-head features originate from earlier work with possible historical validation exposure. New model gradients use fit only; checkpoint selection uses stop only; alpha and DP smoothness use cal only; dev is read once after configuration files are hash-locked. Test images and GT remain unopened until the plan's P5 entry criteria are met.

## Models and controls

S64 and S96 score the same 65 candidate positions. N0 is a 286,401-parameter pointwise residual refiner and shares the S64 score, 44-dimensional context, normalization, target, and output interface with T8/T32/TG. The three Transformer models differ only in circular attention range. The exact 65-state cyclic DP uses the same alpha/lambda grid for every family; its JIT rasterizer was checked pixelwise against the original reconstruction rule.

Local weights and caches are deliberately ignored by Git. The archived Fixed weights must remain in `../LocalContour_Phase2_Fixed_20261006/models`; `phase3_archive.py` checks every SHA256 against the historical manifest. No weights are included in the public source or result package.

## Run order

Use Python with NumPy, SciPy, PyTorch, contourpy, pytest, and Numba. This run used NumPy 1.26.4, SciPy 1.13.1, PyTorch 2.2.2+cu121, contourpy 1.2.1, and Numba 0.68.0 on an RTX 4090.

1. `python -m pytest -q tests/test_core_repair.py tests/test_phase3_controls.py tests/test_phase3_dp.py`
2. `python build_feature_cache.py`, `python build_candidate_cache.py --radius 32`, `python build_node_context.py`
3. `python phase3_archive.py`
4. `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase3_calibrate.py --layer archive`
5. `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase3_readout.py --layer archive`
6. Commit source and protocol outputs, then `python run_phase3.py` for the resumable 18-model controlled queue and locked readout.
7. `python phase3_audit.py`, `python phase3_finalize.py`, `python phase3_case_panels.py`, and `python phase3_runtime.py` generate the recomputation audit, decision tables, visual cases, and measured system timing.

The readout is a development comparison. `phase3_analyse.py` gives C1/C2 image-paired bootstrap intervals and a metric gate; the full GO/HOLD/REDIRECT decision also requires movement-budget and risk analyses from the plan. Do not open the official test set based on the metric gate alone.
