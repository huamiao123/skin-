# INVALID RUN

该目录由错误导入 `setting_config` 产生，实际 `network=ege_wave_unet`、默认 `fusion_type=spatial_gate`，不是 EGE-Dual difflr。保留作审计记录，不得用于结果汇总。

正确入口为 `research/run_retrain_ege_dual_difflr_v1.py`，启动前会断言配置类、network、fusion 和 diff-lr。
