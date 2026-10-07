Phase2 code audit package / 源码审计包

先读 Phase2_源码审计报告_2026-10-05.txt。
关键结论：dense R32的keep索引为32，但两个训练函数仍把keep目标设成索引0。
原仓库文件未修改。minimal_core_fix.patch仅是核心修复建议，需要重训。

运行示例（不需要数据集/权重）：
python reproduce_audit.py --repo "D:\project\skin--main" --out audit_results.json
python check_archived_results.py --repo "D:\project\skin--main" --out archived_result_checks.json

报告中的数值来自CPU合成测试或已发布CSV重算，不是重新训练得到的性能。
make_deliverables.py为本次内部整理脚本，不包含在交付包；不需要运行。
