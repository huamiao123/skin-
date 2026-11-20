"""Run the HAM recipe unchanged on one target split, with local cache / durable attribution."""
from pathlib import Path
import sys,json,argparse,hashlib
from prepare_matched import BASE,SCRATCH,sha
HAM=BASE.parent/'ham10000_window16_20g_20261119'
sys.path.insert(0,str(HAM))
import train_ham as recipe

def run(dataset,arm):
    runroot=BASE/dataset
    recipe.ROOT=runroot
    recipe.CACHE=SCRATCH/dataset
    source=recipe.config
    def matched_config(name):
        c=source(name)
        c.update(dataset=dataset,adapter_sha256=sha(Path(__file__)),prep_sha256=sha(BASE/'prepare_matched.py'),
                 augmentation='paired horizontal/vertical flips and rotation, common seed-and-epoch transform per image',
                 objective='FP32 BCEWithLogits plus per-image soft Dice',clip_grad_per_param_group=1.)
        ref=json.loads((HAM/('swin_local16' if name=='swin_local16' else name)/'config.json').read_text())
        for key in ['seed','windows','backbone','backbone_sha256','max_epochs','patience','selection','batch','image_size',
                    'lr_encoder','lr_decoder','weight_decay','scheduler','threshold','stage3_4_trainable','encoder_eval','sdi']:
            assert c[key]==ref[key],(key,c[key],ref[key])
        return c
    recipe.config=matched_config
    recipe.fit(arm)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset',choices=['ISIC2018','PH2','Waterloo'])
    parser.add_argument('arm',choices=recipe.ARMS)
    args=parser.parse_args();run(args.dataset,args.arm)
