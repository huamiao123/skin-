# 第九轮数据与依赖恢复说明

仓库是已完成实验的源码/证据存档，不包含可直接开训的图片、Mask、第三方文本、模型权重或Python/CUDA扩展。先运行 `python referring_image_segmentation/publication/verify_archive.py` 校验发布快照。`validation.json` 是原机器的完成证据，不代表克隆后自动复现。

## 文本标注

固定来源为 `https://github.com/naamiinepal/medvlsm`，commit `dc36a3af83035efa344c8ce26397528ad173569e`。三份文件位于 `data/isic/anns/{train,val,test}.json`；原始记录数810/90/379。下载到项目的 `text_stage_preparation/isic_text_audit/medvlsm/`，与其中已发布的 `download_audit.json` 核对文件大小、SHA256。

例如在仓库根目录执行以下代码，仅恢复约2.3MB第三方文本，不启动训练：

```python
import hashlib, json
from pathlib import Path
from urllib.request import urlopen

root = Path('referring_image_segmentation/text_stage_preparation/isic_text_audit/medvlsm')
audit = json.loads((root / 'download_audit.json').read_text())
for split, record in audit['splits'].items():
    with urlopen(record['url'], timeout=60) as response:
        content = response.read()
    assert len(content) == record['bytes']
    assert hashlib.sha256(content).hexdigest() == record['sha256']
    (root / f'{split}.json').write_bytes(content)
```

## 图像、Mask和权重

`stage9_isic_text/prepare_data.py` 指向ISIC官方2016 Task1的4个ZIP，核验解压后删除ZIP，保留成员文件和SHA。原机目标为 `/home/featurize/rsi_data/benchmarks/isic2016`。脚本固定test优先去重，得到806训练/90验证/379测试，再生成模型使用的归一化P3文本及256输入缓存。不要复用旧ISIC2018的1886/808划分，也不要直接以包含GT位置/面积的P6/P9替代本轮P3。

视觉预训练权重和依赖来源见 `stage1_visual/provenance.json`、本目录 `third_party.json`、第八轮 `requirements-local.txt` 与安装日志。VMamba需要原版本selective-scan CUDA扩展；普通CPU环境不能直接运行。

CLIP完整文件存放在 `stage8_text_baselines/assets/clip-vit-base-patch32/`，来源为 `openai/clip-vit-base-patch32` revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`。只需恢复第九轮 `text_provenance.json` 中 `source_file_hashes` 所列文件并逐一核对哈希，不需要重新下载RefCOCO图像。第九轮仅用CLIP文本Encoder，不用CLIP视觉分支或RefCOCO训练出的分割权重。

## 新运行与原存档

本阶段的main脚本会发现已完成的DONE/validation文件后跳过。复现实验应使用独立运行副本，保留源存档，在新副本移除旧运行状态/结果/资产清单，再恢复依赖并重新生成数据、文本、预检和运行provenance。不能把仓库中的已完成标记当作新机器的校验通过。

原代码与哈希保留原路径。需要跨机器更改绝对数据路径时，应在独立复现实验中记录改动和新哈希；不要修改此存档后声称与原训练字节一致。恢复上述条件后，原入口为 `python stage9_isic_text/launch.py`。原协议是单seed42、四模型、100epochs，不因克隆或恢复而扩展种子/模型矩阵。
