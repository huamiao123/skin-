"""Export fixed-selector predictions and verify the saved validation metrics."""
from common import *
import argparse
from models import Segmentor
from train import load_trainable,NAMES
from stage3_confirmation.data import ImageData
from engine import evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    setup()
    out=ROOT/('smoke' if args.smoke else 'results')
    done=json.loads((out/'DONE.json').read_text())
    assert done['epochs']==(2 if args.smoke else 300)
    assert all(v==done['epochs']*30 for v in done['updates'].values())
    config=json.loads((out/'config.json').read_text())
    data=ImageData(42)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    result={}
    for name in NAMES:
        model=Segmentor(name).cuda().eval()
        assert model.encoder.frozen_state_hash()==done['encoder_hash']
        result[name]={}
        for criterion in ('best_dice','best_loss','last'):
            ckpt=torch.load(out/name/(criterion+'.pth'),map_location='cpu')
            assert ckpt['frozen_hash']==done['encoder_hash']
            load_trainable(model,ckpt['model'])
            metrics,probs,counts=evaluate(model,data,config['evaluation_batch'],export=criterion=='best_dice')
            assert metrics['counts']==ckpt['metrics']['counts'],(name,criterion,'counts changed')
            assert abs(metrics['loss']-ckpt['metrics']['loss'])<2e-5,(name,criterion,'loss changed')
            rec=dict(name=name,criterion=criterion,selected_epoch=ckpt['epoch'],metrics=metrics)
            dest=out/name/criterion;dest.mkdir(exist_ok=True)
            dump(dest/'metrics.json',rec)
            if criterion=='best_dice':
                assert probs.shape==(808,256,256) and np.isfinite(probs).all()
                np.save(dest/'probabilities.npy',probs)
                np.save(dest/'masks.npy',(probs>=.5).astype(np.uint8))
                np.save(dest/'counts.npy',counts)
                dump(dest/'ids.json',data.ids['val'])
            result[name][criterion]=rec
            print('EXPORTED',name,criterion,'epoch',ckpt['epoch'],'macro Dice',metrics['macro']['dice'],flush=True)
        assert model.encoder.frozen_state_hash()==done['encoder_hash']
        del model
        torch.cuda.empty_cache()
    dump(out/'evaluation.json',dict(passed=True,n_val=808,smoke_only=args.smoke,results=result))


if __name__=='__main__':main()
