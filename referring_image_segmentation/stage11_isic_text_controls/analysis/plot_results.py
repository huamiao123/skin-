"""Standalone learning curves and fixed-ID examples after locked evaluation."""
from pathlib import Path
import sys,json
STAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STAGE))
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw
from engine import restore

def main():
    assert json.loads((ROOT/'analysis/validation.json').read_text())['passed']
    old=json.loads((PREVIOUS/'results/seed42/history.json').read_text())
    new=json.loads((ROOT/'results/seed42/history.json').read_text())
    curves={NAMES[0]:new}
    for n,source in REUSED.items():
        curves[n]=[dict(epoch=r['epoch'],train_loss=r['models'][source]['train_loss'],val=r['models'][source]['val']) for r in old]
    labels={NAMES[0]:'E0: no text (new)',NAMES[1]:'E1: fixed text (historical)',NAMES[2]:'E2: image attributes (historical)'}
    colors=('#2468b4','#e29c24','#289351')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for (n,rows),color in zip(curves.items(),colors):
        epochs=[r['epoch'] for r in rows]
        axes[0].plot(epochs,[r['train_loss'] for r in rows],label=labels[n],color=color,lw=1.5)
        axes[1].plot(epochs,[100*r['val']['macro']['dice'] for r in rows],label=labels[n],color=color,lw=1.5)
    axes[0].set(xlabel='Epoch',ylabel='Training BCE + Dice loss')
    axes[1].set(xlabel='Epoch',ylabel='Validation mean Dice (%)',ylim=(80,95))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('ISIC2016 / seed 42 / matched 100-epoch training')
    fig.savefig(ROOT/'analysis/learning_curves.png',dpi=160);fig.savefig(ROOT/'analysis/learning_curves.svg');plt.close(fig)
    rows=json.loads((ASSETS/'test.json').read_text())
    logits={n:np.load(ROOT/'analysis'/n/'test_logits64.npy',mmap_mode='r') for n in NAMES}
    records={n:json.loads((ROOT/'analysis'/n/'test_per_image.json').read_text()) for n in NAMES}
    panel=Image.new('RGB',(5*256,6*282),(20,20,20));draw=ImageDraw.Draw(panel)
    for i,r in enumerate(rows[:6]):
        y=i*282;panel.paste(Image.open(r['image_path']).convert('RGB').resize((256,256)),(0,y+26));draw.text((4,y+3),r['id'],fill='white')
        panel.paste(Image.open(r['mask_path']).convert('RGB').resize((256,256),Image.Resampling.NEAREST),(256,y+26));draw.text((260,y+3),'Ground truth',fill='white')
        for c,n in enumerate(NAMES,2):
            image=Image.fromarray((restore(logits[n][i],r)>=0).astype(np.uint8)*255)
            panel.paste(image.resize((256,256),Image.Resampling.NEAREST).convert('RGB'),(c*256,y+26))
            draw.text((c*256+4,y+3),f"{n}: Dice {records[n][i]['dice']:.3f}",fill='white')
    panel.save(ROOT/'analysis/fixed_id_prediction_panel.png')
    print('PLOTS_READY',flush=True)

if __name__=='__main__':main()
