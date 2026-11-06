# MambaVision-S：VMamba SS2D式四路扫描消融

2026-10-31 正式训练前冻结。根据用户“尝试SS2D的VMamba”的后续指令，本轮执行前次讨论的最小扫描替换方案：使用VMamba官方四路扫描/合并，保留原SSM投影权重；不是整块SS2D或完整VMamba Encoder替换。已向用户提供两种范围选择，在未收到其他选择后明确按本方案推进。只新增一个模型、seed42；旧stage1–14保持只读，不推送远端。

## 改变与保留

主参考是已完成stage14的原始NVIDIA MambaVision-S + Swin，seed42、同初始化和训练预算。实际原始权重、config、验证清单、原图及数据增强、代码和完整结果哈希先核验，不重训原模型。DINOv3/VMamba stage4仅作为背景，不追加训练。

只改变MambaVision-S后两级7个MambaVisionMixer的SSM分支扫描：level2 blocks0–3，level3 blocks0–2。保留原in_proj、conv1d_x/z、x_proj、dt_proj、A_log、D、out_proj，保留CNN、窗口划分/填充、Attention、MLP、残差和非SSM旁路。conv1d_x完成后将窗口序列还原为14×14或7×7网格，使用VMamba原始cross_scan_fn(scans=0)产生行、列、反向行、反向列序列。四路共享同一份原NVIDIA投影和SSM参数，分别执行原mamba-ssm selective_scan_fn，使用VMamba原cross_merge_fn对齐，FP32求和除4后还原dtype。均值合并是本次预先固定的方案，不搜索方向权重或额外增益系数。

VMamba官方commit `2ed52ead062a51a64521ed3871d52914bf532876`，直接导入既有vendor/vmamba.py，核验Git对象与文件一致。cross_scan/merge使用官方PyTorch CUDA实现force_torch=True；SSM使用与stage14相同的官方mamba-ssm1.2.2 CUDA扩展，避免被历史2.2.4扩展覆盖。无新增/随机初始化参数，无新增norm/门控，无更改SSM状态维数，无方向独立权重、无解冻。

全部原预训练参数及缓冲区数值、键、摘要与stage14相同，但扫描前向已经改变。因此明确称为“共享预训练参数的SS2D式四路扫描消融”，不能称为未经修改的官方MambaVision，也不代表完整VMamba SS2D模块。全参数相同只证明权重/容量相同，不代表前向相同。

## 数据、训练与选模

完全复用stage3 Confirmation ImageData与engine：ISIC2018旧1886训练/808开发验证，256输入，RGB/Mask/ImageNet normalization与源文件SHA不变。H/V flip及rotation各p.5，旋转0–360°，原顺序/zero-padding/mask nearest；seed+epoch×100003同时控制增强和顺序。每轮重新编码。验证非官方独立测试，患者身份未认证。

Frozen Encoder递归eval/no_grad，训练前后参数和BN状态不变；仅训练原Adapter+Swin+Head，同stage14初始Probe及500321个可训练参数。新模型从相同初值独立训练，不加载stage14分割权重。AdamW lr.001、wd.01、betas.9/.999、eps1e-8；batch64，保留尾批30，无梯度累积，30更新/epoch；300完整epochs/9000更新，无early stop。Cosine T_max300、eta_min1e-5、无warmup、轮末scheduler.step；原BCE+逐图SoftDice(smooth1)，BF16 autocast/FP32 loss、clip1。cudnn deterministicTrue、benchmarkFalse，不承诺所有算子逐位确定。

主best_dice按最高验证macro Dice，精确并列取早；辅助best_loss按最低样本加权val loss、last固定第300轮。三种权重和完整latest保留。恢复必须包含optimizer、scheduler、全部RNG、更新计数和selectors。单独2轮真实数据smoke先中断后恢复，不能混入正式权重。

## 扫描与模型预检

原模型严格加载HF revision `e4f9ea93877fba6f1e2d231fd0f0a391368ad895` 的同一S-1K权重；共享stage14官方资产/runtime只读软链接，绝不写入其目录。权重SHA和扩展路径必须相同。

真实窗口输入逐块验证：只保留单路线时，7个修改块和整个Encoder的输出必须与原始前向完全相同；开启四路后前两级特征不变、后两级确实变化、输出尺寸和有限性正确。用独立矩形坐标序列核对官方四路次序，并验证scan/merge除4的精确往返。参数/缓冲区键和值必须完全保持，无新增参数。再检查batch64真实训练梯度、冻结、显存、完整推理和两轮跨进程恢复。

## 评价与预定结论

原evaluation.py、stage1.analyze.metrics、256网格、threshold0.5及所有空mask/边界/插值规则保持不变。对三个checkpoint分别导出808张概率/Mask/逐图JSON与CSV，报告macro/pooled Dice、IoU、Precision、Recall、Prec@IoU.5/.6/.7/.8/.9、ISIC阈值Jaccard、BoundaryF1、HD95像素和对角线归一化。独立复算像素TP/FP/FN/TN和Dice/IoU；不得按各指标分别挑轮。

主比较four-route−stage14 original，所有数据、预训练参数、可训练参数、初值、选模和预算相同；固定4路均值不是调参结果。对照stage4 DINO/VMamba仅描述。历史概率SHA及ID一致时做纠正/新增错误像素；分组GT比例<.1、.1–.3、>=.3；展示相对原始MambaVision最大改善、最大退步与自身最差病例，源图仅本地。学习曲线及逐图分布完整保存。图像配对2000次bootstrap seed42仅探索性，不代表患者/训练seed或独立测试证据。

完整batch1计时包含Encoder/Adapter/Swin/Head，无特征缓存，BF16，预热3/重复10、CUDA同步；记录实际24GB4090及峰值显存。原stage14速度是原会话测量，不重跑历史模型。无完整scan FLOPs时不报告部分FLOPs为全模型计算量。

若四路版本提高，只支持这组固定权重/窗口内/共享参数方案，不能直接推广为完整SS2D一定更强；若下降，也不否定通过预训练/微调的完整SS2D。原始权重按单路训练，四路改变分布是潜在限制，不单独归因于扫描优劣。本轮不微调、不增加完整SS2D或额外seed。

300轮/9000更新、冻结与方向配置、三选模、808全指标、源码哈希及报告一致性全部通过后才标complete。
