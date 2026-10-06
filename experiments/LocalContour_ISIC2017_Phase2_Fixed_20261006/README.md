# LocalContour Phase 2 修复复验（2026-10-06）

状态：`HOLD_GLOBAL_PRIMARY_NO_INDEPENDENT_TEST`。旧阶段 B/C 把 R32 稠密候选的“保持原位”错误标成索引 0（实际偏移 −32；正确索引 32），其模型比较结论作废。修复版三种子开发集结果中，全局注意力平均 Dice 为 0.85576，局部注意力为 0.85441，局部大模型为 0.85399；但全局相对局部大模型的配对区间跨过 0，也没有在全部种子上胜出。**不能沿用旧版“全局没有增益”的判断，也尚不能宣布全局方法验证成功。**完整方法、主表和限制见 [修复版报告](REPAIRED_REPORT.md)。官方 ISIC2017 test 继续封存。

本修复版本使用 `candidate_layout.py` 统一 R32 索引；普通和门控训练都校验零位与标签有效性。特征缓存构建器写入已核实的 GT 形状数，审计器接受新建和历史迁移两种经过验证的二维 GT 来源。局部模型改为逐像素通道分组归一化，避免新增网络通过整图 GroupNorm 统计交换信息；冻结 CNN 特征本身仍可包含全图信息。旧实验的 GroupNorm 版本保留在 Git 历史和 `historical_pre_audit/` 的原报告中。

已重建 train/val 缓存、重新训练 21 组模型（3 种子 × 7 族 × 8 epoch）、重新校准与评分，旧权重和旧候选分数未复用。旧结果和原报告原样保留在 `historical_pre_audit/`，审计原件存入 `protocols/source_audit/phase2_audit/`。全部 val150 用于选择 checkpoint，cal50/val100 也是已复用的开发数据；所有区间仅供探索。

合成检查 `python -m pytest tests/test_core_repair.py -q` 共 7 项通过；缓存清理前的完整审计 `python audit_repaired_phase2.py` 结果为 `PASS_REPAIRED_DEVELOPMENT_AUDIT`。审计后已删除约 12.67 GB 可重建缓存与模型分数，见 `results/cache_provenance/cleanup.json`。公开版本不包含权重或原始数据。
