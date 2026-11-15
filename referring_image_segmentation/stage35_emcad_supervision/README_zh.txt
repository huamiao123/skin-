Stage35：新服务器上的 EMCAD 多尺度监督对照
服务器：workspace.featurize.cn:32098，featurize；RTX4090 24GB。凭据不写入文件。
代码、数据、环境、Stage34结果和最佳权重已在新实例存在，无需重复传输。

研究对象：审计发现现有适配版取消了作者四级监督。Stage34修正初始化和扩展倍率后未建立稳定增益，本轮单独检验监督拓扑。
A emcad_single：EMCAD64 / expansion2 / ReLU6，单输出 BCE+逐图softDice。
B emcad_deep：同一编码器、适配器、原始EMCAD解码器及最终头初值；增加三个1×1辅助头（195参数），监督原作者四个解码输出及四输出logit之和。
B将五个基础loss等权取平均，避免直接引入五倍整体倍率。保留公共BCE+Dice，不使用作者weightedBCE+weightedIoU。因此本轮是监督结构适配，不是完整作者训练复现。
两组推理均只使用最终精细输出，无辅助头集成、无阈值搜索。

两组共享：seed42，100轮，原300轮cosine的前100轮，encoder LR1e-4、decoder LR1e-3、weight_decay0.01；BF16 forward、FP32 loss；stage3/4参数可训练，encoder整体eval。
实际micro batch从旧48+16改为64，effective batch仍64。此项对两组共同生效；因GPU与BN microbatch都改变，主比较是新卡上重新训练的A/B，旧A4000结果仅作历史参照。
数据1886train/762val/867test；按最大验证macro Dice选最佳（平局取较早），测试阈值0.5。
两个完整训练均完成后，再分别执行固定最佳权重的完整测试；指标由公共evaluation定义。
原始测试集已经暴露，单seed不能证明跨种子稳定性；不宣称患者级完全独立。

已通过：共享初始化逐元素一致、初始推理一致、五项监督公式核对、所有可训练参数加入优化器、真实数据梯度、冻结参数未变、两组两轮实际进程重启与验证回放。
旧EMCAD最佳权重在4090回放的验证macro变化仅-0.000248个百分点；旧权重哈希已验证。
当前阶段与轮次读取pipeline_status.json和各arm/results/run_status.json。
正式流水线串行：emcad_single100 → emcad_deep100 → 两组验证/测试 → FINAL_COMPARISON.json。
只保留验证最佳权重，续训latest在完成评估后清理；此前已完成实验不修改。
