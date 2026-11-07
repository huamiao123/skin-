# 上游源码恢复

公开归档不重新分发 HEDS-Net、SegMAN 和 HGM 的上游源码快照。需要复现时，从项目目录执行 `python publication/restore_stage20_21_references.py`；脚本从各自固定的官方 commit 获取源文件并逐一核验 SHA256。请根据上游的许可使用源码；恢复的文件被本目录的 `.gitignore` 排除。
