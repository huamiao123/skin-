"""Independently verify all 300-epoch reports and create standalone result figures."""
from pathlib import Path
import sys,json
STAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STAGE))
from common import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from engine import restore

def main():
    assert json.loads((ROOT/'analysis/validation.json').read_text())['passed']
    report=json.loads((ROOT/'analysis/summary.json').read_text());history=json.loads((ROOT/'results/seed42/history.json').read_text())
    config=json.loads((ROOT/'results/seed42/config.json').read_text())
    assert len(history)==300 and config['epochs']==300 and config['total_schedule_steps']==15300
    assert config['warmup_steps']==255 and config['previous_segmentation_weights_loaded'] is False
    checks=[]
    for n,m in report['models'].items():
        best=max(history,key=lambda r:r['models'][n]['val']['macro']['dice']);assert best['epoch']==m['selected_epoch']
        for split,metric in m['splits'].items():
            rows=json.loads((ROOT/'analysis'/n/f'{split}_per_image.json').read_text())
            tp=np.array([r['tp'] for r in rows],np.int64);fp=np.array([r['fp'] for r in rows],np.int64);fn=np.array([r['fn'] for r in rows],np.int64)
            d=np.divide(2*tp,2*tp+fp+fn,out=np.ones(len(rows),np.float64),where=(2*tp+fp+fn)>0)
            macro=float(d.mean());pooled=float(2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum()))
            assert abs(macro-metric['macro']['dice'])<1e-12 and abs(pooled-metric['pooled']['dice'])<1e-12
            checks.append(dict(model=n,split=split,n=len(rows),independent_macro_dice=macro,independent_pooled_dice=pooled))
    dump(ROOT/'analysis/independent_report_check.json',dict(passed=True,checks=checks,all_selected_epochs_match_validation_history=True,
        all_five_trained_fresh_300=True,updates_per_model=15300))
    lines=['# 统一300轮预算的结果解释','',
        '五组均从相同seed42规则重新初始化，独立完成300轮/15300步；使用完整300轮cosine，warmup255步。没有加载100轮分割权重续训。','',
        '## 相同预算内的主要配对','']
    for name,c in report['paired_test_comparisons'].items():
        lines.append(f"- {name}: {c['macro_dice_delta']*100:+.3f}个百分点；图像配对95%区间 {[round(v*100,3) for v in c['paired_image_bootstrap95']]}。")
    lines+=['','## 100轮与300轮的预算背景','']
    for n,c in report['budget_300_minus_100'].items():lines.append(f"- {n}: 300-100为{c['macro_dice_delta']*100:+.3f}个百分点。")
    lines+=['','100/300的总长度cosine不同，因此300轮前100轮不是原100轮训练的完全重复轨迹。除预算/调度总步数外，数据、初始参数、增强、batch、optimizer及选模规则保持一致。',
        '', '前后融合方案同时改变词级/句子表示、交互位置与输出头；主要文字收益应看各方案内逐图-固定的配对。单seed图像区间不估计患者或训练随机性，不将小差异自动解释为稳定提升。',
        '', '全部10张模型/划分Dice表由逐图TP/FP/FN独立复算通过，最佳轮符合完整验证历史。已用于开发的测试集和未验证患者身份保持原有解释范围。']
    (ROOT/'analysis/FINDINGS.md').write_text('\n'.join(lines)+'\n')
    labels={NAMES[0]:'Visual only',NAMES[1]:'Late fixed text',NAMES[2]:'Late image text',NAMES[3]:'Predecoder fixed text',NAMES[4]:'Predecoder image text'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for n in NAMES:
        epochs=[r['epoch'] for r in history]
        axes[0].plot(epochs,[r['models'][n]['train_loss'] for r in history],label=labels[n],lw=1.2)
        axes[1].plot(epochs,[r['models'][n]['val']['macro']['dice']*100 for r in history],label=labels[n],lw=1.2)
    axes[0].set(xlabel='Epoch',ylabel='Training BCE + Dice loss')
    axes[1].set(xlabel='Epoch',ylabel='Validation mean Dice (%)',ylim=(80,95))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('ISIC2016 / seed42 / five fresh 300-epoch runs')
    fig.savefig(ROOT/'analysis/learning_curves.png',dpi=160);fig.savefig(ROOT/'analysis/learning_curves.svg');plt.close(fig)
    x=np.arange(len(NAMES));a=[report['reference_100_epoch_results'][n]['splits']['test']['macro']['dice']*100 for n in NAMES]
    b=[report['models'][n]['splits']['test']['macro']['dice']*100 for n in NAMES]
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    ax.bar(x-.18,a,.36,label='Historical 100 epochs');ax.bar(x+.18,b,.36,label='Fresh 300 epochs')
    ax.set_xticks(x,[labels[n] for n in NAMES],rotation=12,ha='right');ax.set_ylabel('Test mean Dice (%)');ax.set_ylim(80,95)
    ax.legend();ax.grid(axis='y',alpha=.2);fig.savefig(ROOT/'analysis/budget_comparison.png',dpi=160);plt.close(fig)
    rows=json.loads((ASSETS/'test.json').read_text());logits={n:np.load(ROOT/'analysis'/n/'test_logits64.npy',mmap_mode='r') for n in NAMES}
    records={n:json.loads((ROOT/'analysis'/n/'test_per_image.json').read_text()) for n in NAMES}
    panel=Image.new('RGB',(7*256,6*282),(20,20,20));draw=ImageDraw.Draw(panel)
    for i,r in enumerate(rows[:6]):
        y=i*282;panel.paste(Image.open(r['image_path']).convert('RGB').resize((256,256)),(0,y+26));draw.text((4,y+3),r['id'],fill='white')
        panel.paste(Image.open(r['mask_path']).convert('RGB').resize((256,256),Image.Resampling.NEAREST),(256,y+26));draw.text((260,y+3),'Ground truth',fill='white')
        for c,n in enumerate(NAMES,2):
            mask=Image.fromarray((restore(logits[n][i],r)>=0).astype(np.uint8)*255)
            panel.paste(mask.resize((256,256),Image.Resampling.NEAREST).convert('RGB'),(c*256,y+26))
            draw.text((c*256+4,y+3),f"{labels[n]}: {records[n][i]['dice']:.3f}",fill='white')
    panel.save(ROOT/'analysis/fixed_id_prediction_panel.png')
    p=ROOT/'README.md';s=p.read_text();s+='\n## 已完成\n\n五组均完成300轮/15300步，原图评估及独立计数复算通过。\n\n| 模型 | 最佳轮 | 测试macro Dice | 测试pooled Dice |\n|---|---:|---:|---:|\n'
    for n,m in report['models'].items():
        a=m['splits']['test'];s+=f"| {n} | {m['selected_epoch']} | {a['macro']['dice']*100:.3f}% | {a['pooled']['dice']*100:.3f}% |\n"
    s+='\n完整解释、配对区间与100/300预算对照见analysis/FINDINGS.md和REPORT.md。\n';p.write_text(s)
    print('INDEPENDENT_REPORT_AND_FIGURES_PASSED',flush=True)

if __name__=='__main__':main()
