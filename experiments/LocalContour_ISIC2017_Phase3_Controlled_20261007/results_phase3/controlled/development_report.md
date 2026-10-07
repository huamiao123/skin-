# LocalContour Phase 3 controlled development report

**Decision: HOLD.** The preregistered cross-node value claim did not pass the development gate. This is a controlled development result on 100 ISIC2017 validation images, not an independent test finding. Official test600 remains unopened.

The new train2000 was split by SHA256 image ID into fit1800 and stop200; public metadata has no patient/lesion group ID, so patient independence is unknown. Frozen CNN and boundary head were inherited from earlier work with possible historical validation exposure. New S64/S96/N0/T8/T32/TG gradients used fit only, checkpoints used stop only, and all postprocessing parameters used cal50 only. The archived Fixed checkpoints were separately re-evaluated and are not mixed into this table.

| dev100 method | Dice, 3-seed mean | BF1, 3-seed mean |
|---|---:|---:|
| CNN | 0.853051 | 0.513784 |
| G1 zero reconstruction | 0.853075 | 0.513483 |
| S64_A | 0.854553 | 0.533626 |
| S64_D | 0.855620 | 0.530661 |
| S96_A | 0.854654 | 0.535242 |
| S96_D | 0.855663 | 0.533156 |
| N0_A | 0.854641 | 0.531017 |
| N0_D | 0.854475 | 0.526509 |
| T32_A | 0.853461 | 0.521544 |
| TG_A | 0.855037 | 0.536516 |
| TG_D | 0.854643 | 0.532965 |

**C1:** TG_A−N0_A = +0.000396 Dice, paired image-bootstrap 95% CI [-0.0009524662007992108, 0.0017579009819734294]; seed differences [0.0006344364925128802, -0.0011420859246013555, 0.0016949548366923973]. The +0.001 engineering gate was missed. TG_A increased correct-to-incorrect rate by +0.0351 relative to N0_A on 92 images with a defined denominator, beyond the +0.005 guardrail, even though mean BF1 improved.

**C2:** TG_D−S96_D = -0.001020 Dice, 95% CI [-0.0037862826713452266, 0.0013666968476169796]; seed differences [-0.004299440526410942, 0.0009503698331436417, 0.00028900649036879317]. S96_D was selected as BestSimple only from the three family means on cal50. All `_D` methods used exact full-65 cyclic DP over the same 8 alpha × 7 lambda grid. 6/18 selected lambda=1.0, the grid's upper boundary.

At cal50-matched 10% movement, dev100 mean Dice was N0 0.854254 and TG 0.854308. At the 25% target, actual dev movement differs materially between methods; the comparison is near-budget only. The 50% target was unreachable for many model/seed combinations and was not forced with negative alpha. Full exploratory alpha curves are in `alpha_curve.csv`; alpha=0 causes substantial harmful overmovement in several methods.

The fit-defined separated-peak low-margin, fixable, S64-failed group contained 3426, 3127, 3216 nodes across the three seed-specific analyses. TG corrected only 93, 109, 82 of them (about 2.5–3.5%). The same-input GT-distance Oracle reached Dice 0.943500; 95.3% of contour nodes had a <=2px candidate and 93.1% of GT boundary pixels were in the search-band quadrilateral union. This Oracle uses GT and is neither deployable nor a strict Dice upper bound.

On ten fixed cal50 images, seed17 full-system pure-compute mean latency was 182.8 ms for N0_A, 183.1 ms for TG_A, 183.1 ms for TG_D, and 189.1 ms for S96_D; startup and disk I/O are excluded. Most time was spent constructing candidates and preserving the original prediction context, while the N0/TG refiner itself was below 1.1 ms on average. This ten-image seed17 profile is not a population latency guarantee; see `runtime_profile.json` for p50/p95 and per-image values.

The independent audit passed: 18 checkpoints and score caches had matching hashes, all 18 cal50 configurations were locked, 4,200 dev rows covered the same 100 IDs, and 120 fixed image/method/seed recomputations matched every stored Dice/IoU/BF1/HD95/ASSD exactly. Operational logs have no test access. New-model training did not hit the 40-epoch budget edge. These checks do not remove historical upstream validation exposure or image-level-only split limitations.

The main intervals remain wide enough to include modest positive effects, so REDIRECT/equivalence is not established. The evidence supports pausing the global-contour necessity claim and, if pursued, pre-registering one focused correction/harm experiment before any independent test. No P5 final lock was issued.
