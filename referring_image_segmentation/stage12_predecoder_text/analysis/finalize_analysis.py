"""Independent count checks, paired interpretation and static result figures."""
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
    checks=[]
    for n,m in report['models'].items():
        base=ROOT if m['newly_trained'] else CONTROLS
        if m['newly_trained']:
            best=max(history,key=lambda r:r['models'][n]['val']['macro']['dice'])
            assert best['epoch']==m['selected_epoch']
        for split,metric in m['splits'].items():
            rows=json.loads((base/'analysis'/n/f'{split}_per_image.json').read_text())
            tp=np.array([r['tp'] for r in rows],np.int64);fp=np.array([r['fp'] for r in rows],np.int64);fn=np.array([r['fn'] for r in rows],np.int64)
            d=np.divide(2*tp,2*tp+fp+fn,out=np.ones(len(rows),np.float64),where=(2*tp+fp+fn)>0)
            macro=float(d.mean());pooled=float(2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum()))
            assert abs(macro-metric['macro']['dice'])<1e-12 and abs(pooled-metric['pooled']['dice'])<1e-12
            checks.append(dict(model=n,split=split,n=len(rows),macro_dice=macro,pooled_dice=pooled))
    dump(ROOT/'analysis/independent_report_check.json',dict(passed=True,tables=checks,selected_epochs_match_validation_history=True))
    comparisons=report['paired_test_comparisons'];main=comparisons['predecoder_image_minus_fixed'];late=comparisons['predecoder_image_minus_late_image']
    words='本轮逐图文本的测试均值高于固定文本' if main['macro_dice_delta']>0 else '本轮逐图文本的测试均值未高于固定文本'
    lines=['# Decoder前词级交互的结果解释','',
        '固定文本与逐图属性文本两组均独立从seed42初始化，完成100轮、5100次更新。Decoder之后只有普通视觉Mask Head；没有句子到像素点积、温度或文本偏置。','',
        f"主比较逐图-固定：{main['macro_dice_delta']*100:+.3f}个百分点，图像配对95%区间 {[round(v*100,3) for v in main['paired_image_bootstrap95']]} 个百分点。{words}；一个seed不能推断训练稳定性。",
        f"新逐图-历史晚期逐图：{late['macro_dice_delta']*100:+.3f}个百分点，区间 {[round(v*100,3) for v in late['paired_image_bootstrap95']]} 个百分点。",'',
        '新旧比较同时改变融合位置、句子/词级表示和输出头，所以属于整体机制替换。逐图-固定的新配对具有相同结构容量和公共初值，是解释文字额外信息的主要依据。','',
        '所有模型macro/pooled Dice均由逐图TP/FP/FN独立重算通过，新模型最佳轮与完整验证历史一致。十张模型/划分表保留完整指标。历史实验没有改写或重训。','',
        '806/90/379划分和P3颜色/形状属性保持原样，word bank含50种字符串（含固定提示）；PAD/BOS/EOS屏蔽。文本不等同于临床报告。患者独立性未验证，测试集此前用于开发；本轮结果不能作为全新外部验证或通用多目标RIS的证明。',
        '', '学习曲线见learning_curves.png，固定ID预测对照见fixed_id_prediction_panel.png，全部指标与区间见summary.json。']
    (ROOT/'analysis/FINDINGS.md').write_text('\n'.join(lines)+'\n')
    old_history=json.loads((PREVIOUS/'results/seed42/history.json').read_text())
    visual_history=json.loads((CONTROLS/'results/seed42/history.json').read_text())
    curves={'E0_visual':visual_history}
    for name,old in [('E1_fixed_text','dinov3_fixed_text'),('E2_image_text','dinov3_text')]:
        curves[name]=[dict(epoch=r['epoch'],train_loss=r['models'][old]['train_loss'],val=r['models'][old]['val']) for r in old_history]
    for n in NAMES:curves[n]=[dict(epoch=r['epoch'],train_loss=r['models'][n]['train_loss'],val=r['models'][n]['val']) for r in history]
    labels={'E0_visual':'Visual only','E1_fixed_text':'Late fixed text','E2_image_text':'Late image text',
        NAMES[0]:'Predecoder fixed text',NAMES[1]:'Predecoder image text'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for n,rows in curves.items():
        epochs=[r['epoch'] for r in rows]
        axes[0].plot(epochs,[r['train_loss'] for r in rows],label=labels[n],lw=1.3)
        axes[1].plot(epochs,[r['val']['macro']['dice']*100 for r in rows],label=labels[n],lw=1.3)
    axes[0].set(xlabel='Epoch',ylabel='Training BCE + Dice loss')
    axes[1].set(xlabel='Epoch',ylabel='Validation mean Dice (%)',ylim=(85,94))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('ISIC2016 / seed42 / early word interaction and historical controls')
    fig.savefig(ROOT/'analysis/learning_curves.png',dpi=160);fig.savefig(ROOT/'analysis/learning_curves.svg');plt.close(fig)
    test=json.loads((ASSETS/'test.json').read_text());names=('E2_image_text',)+NAMES
    bases={names[0]:CONTROLS,NAMES[0]:ROOT,NAMES[1]:ROOT}
    logits={n:np.load(bases[n]/'analysis'/n/'test_logits64.npy',mmap_mode='r') for n in names}
    records={n:json.loads((bases[n]/'analysis'/n/'test_per_image.json').read_text()) for n in names}
    panel=Image.new('RGB',(5*256,6*282),(20,20,20));draw=ImageDraw.Draw(panel)
    for i,r in enumerate(test[:6]):
        y=i*282;panel.paste(Image.open(r['image_path']).convert('RGB').resize((256,256)),(0,y+26));draw.text((4,y+3),r['id'],fill='white')
        panel.paste(Image.open(r['mask_path']).convert('RGB').resize((256,256),Image.Resampling.NEAREST),(256,y+26));draw.text((260,y+3),'Ground truth',fill='white')
        for c,n in enumerate(names,2):
            image=Image.fromarray((restore(logits[n][i],r)>=0).astype(np.uint8)*255)
            panel.paste(image.resize((256,256),Image.Resampling.NEAREST).convert('RGB'),(c*256,y+26))
            label={'E2_image_text':'Late image',NAMES[0]:'Pre fixed',NAMES[1]:'Pre image'}[n]
            draw.text((c*256+4,y+3),f"{label}: Dice {records[n][i]['dice']:.3f}",fill='white')
    panel.save(ROOT/'analysis/fixed_id_prediction_panel.png')
    print('INDEPENDENT_REPORT_AND_FIGURES_PASSED',flush=True)

if __name__=='__main__':main()
