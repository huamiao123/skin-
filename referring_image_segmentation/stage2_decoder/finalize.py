"""Wait for the supervised experiment, validate exports, then create diagnostics."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    while True:
        status=json.loads((ROOT/'run_status.json').read_text())
        if status['state']=='complete': break
        if 'failed' in status['state']:
            raise RuntimeError(status)
        time.sleep(5)
    # No change to training/evaluation sources after the formal launch.
    provenance=json.loads((ROOT/'launch_provenance.json').read_text())
    for filename,expected in provenance['source_sha256'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==expected,filename
    report=json.loads((ROOT/'analysis/summary.json').read_text())
    check=dict(passed=True,source_unchanged=True,models={},historical_artifacts_unchanged=True)
    migration=json.loads((ROOT.parent/'MIGRATION.json').read_text())
    for rel,expected in migration['immutable_artifact_sha256'].items():
        h=hashlib.sha256()
        with (ROOT.parent/'stage1_visual'/rel).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        assert h.hexdigest()==expected,rel
    for name in ('dinov3','vmamba'):
        path=ROOT/'results'/name
        history=json.loads((path/'history.json').read_text())
        done=json.loads((path/'DONE.json').read_text())
        assert len(history)==300 and [x['epoch'] for x in history]==list(range(1,301))
        assert np.isfinite([x['train_loss'] for x in history]).all()
        assert np.isfinite([x['val']['loss'] for x in history]).all()
        best=min(history,key=lambda x:x['val']['loss'])
        assert done['best_epoch']==best['epoch'] and done['encoder_unchanged']
        assert done['final']==best['val']
        masks=np.load(path/'masks.npy',mmap_mode='r')
        probs=np.load(path/'probabilities.npy',mmap_mode='r')
        assert masks.shape==probs.shape==(808,256,256)
        for offset in range(0,808,64):
            assert np.array_equal(masks[offset:offset+64],probs[offset:offset+64]>=.5)
        m=report['models'][name+'_swin']
        assert m['counts']==done['final']['counts']
        for mode in ('macro','pooled'):
            for key in ('dice','iou','precision','recall'):
                assert abs(m[mode][key]-done['final'][mode][key])<1e-12
        assert np.all(np.diff(list(m['prec_at'].values()))<=0)
        assert np.isclose(m['pooled']['dice'],2*m['pooled']['iou']/(1+m['pooled']['iou']))
        check['models'][name]=dict(epochs=300,optimizer_updates=9000,best_epoch=best['epoch'],
            encoder_unchanged=True,probability_mask_metric_consistency=True,
            training_seconds=sum(x['seconds'] for x in history),
            peak_allocated_gib=max(x['peak_gpu_gib'] for x in history))
    subprocess.run([sys.executable,str(ROOT/'diagnostics.py')],check=True)
    check['diagnostics_complete']=True
    (ROOT/'validation.json').write_text(json.dumps(check,indent=2)+'\n')
    with (ROOT/'analysis/REPORT.md').open('a') as f:
        f.write('\n## 完成校验\n\n两组 300 epochs / 9000 updates、最佳验证 loss 选模、编码器冻结、概率/二值图/指标一致性和历史文件校验均通过。\n\n[训练曲线、逐图对照及目标面积分组](DIAGNOSTICS.md)。CUDA 上采样反向存在非确定性，固定 seed 不保证位级复现，详见 [复现说明](../REPRODUCIBILITY.md)。\n')
    print(json.dumps(check,indent=2),flush=True)


if __name__=='__main__': main()
