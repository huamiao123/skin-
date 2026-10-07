# Fixed archive fair re-evaluation (development only)

All 15 archived checkpoints matched the historical SHA256 manifest. Their val150 logits were regenerated twice; the GT-free re-export was byte-identical. The same cal50 selected alpha and, for `_D`, full 65-candidate cyclic-DP lambda independently for each archived model/seed. All 15 choices were hash-locked before dev100 was opened. No official test image or GT was opened.

| Method | dev100 mean Dice across 3 seeds | mean BF1 |
|---|---:|---:|
| Frozen CNN | 0.853051 | 0.513784 |
| G1 zero reconstruction | 0.853075 | 0.513483 |
| S64_A | 0.853936 | 0.538057 |
| S64_D | 0.855111 | 0.535832 |
| S96_A | 0.854003 | 0.536338 |
| S96_D | 0.856405 | 0.535249 |
| T8_A | 0.854412 | 0.530695 |
| T8_D | 0.854272 | 0.528995 |
| T32_A | 0.855127 | 0.534887 |
| T32_D | 0.855168 | 0.534581 |
| TG_A | 0.855760 | 0.540349 |
| TG_D | 0.855933 | 0.536532 |

S96_D exceeds TG_D by 0.000472 mean Dice. Per seed, S96_D versus TG_D is 0.857281 versus 0.854677 (17), 0.857785 versus 0.856076 (23), and 0.854147 versus 0.857046 (42). This archive does not establish that full-contour communication is necessary; it motivates the matched N0 and stronger local controls in the new controlled layer.

**Limit:** All archived checkpoints were selected using the entire historical val150, which includes today's cal50/dev100. Recalibrating their postprocessing on cal50 does not undo checkpoint-selection exposure. These numbers explain previous results; they are not independent confirmation or the Phase-3 main comparison.

Primary artifacts: `archive_checkpoint_manifest.json`, `reexport_verification.json`, per-model `*_calibration_grid.csv`, `*_chosen_config.json` with SHA256 lock files, `dev100_per_image.csv`, and `dev100_summary.csv`.
