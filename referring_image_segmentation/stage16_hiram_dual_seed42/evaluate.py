"""Export fixed-selector predictions; no checkpoint or threshold retuning."""
from common import *
import argparse
from models import BaselineProbe,RegionProbe,tensor_hash
from stage4_fusion.models import EncoderPair
from stage3_confirmation.data import ImageData
from stage3_confirmation.engine import encode
from engine import evaluate

NAMES=('paired_baseline','region_only','hiram')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    setup()
    out=ROOT/('smoke' if args.smoke else 'results')
    done=json.loads((out/'DONE.json').read_text())
    assert done['epochs']==(2 if args.smoke else 300)
    assert all(v==done['epochs']*30 for v in done['updates'].values())
    data=ImageData(42)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    encoder=EncoderPair().cuda().eval()
    assert all(tensor_hash(getattr(encoder,n))==h for n,h in done['encoder_hashes'].items())
    features,targets=encode(encoder,data,'val',64)
    result={}
    for name in NAMES:
        model=(BaselineProbe() if name=='paired_baseline' else RegionProbe(name)).cuda().eval()
        result[name]={}
        for criterion in ('best_dice','best_loss','last'):
            ckpt=torch.load(out/name/(criterion+'.pth'),map_location='cpu')
            model.load_state_dict(ckpt['model'],strict=True)
            metrics,probs,counts=evaluate(model,features,targets,64,export=criterion=='best_dice')
            assert metrics['counts']==ckpt['metrics']['counts'],(name,criterion,'counts changed')
            assert abs(metrics['loss']-ckpt['metrics']['loss'])<2e-5,(name,criterion,'loss changed')
            rec=dict(name=name,criterion=criterion,selected_epoch=ckpt['epoch'],metrics=metrics)
            destination=out/name/criterion
            destination.mkdir(exist_ok=True)
            dump(destination/'metrics.json',rec)
            if criterion=='best_dice':
                assert probs.shape==(808,256,256) and np.isfinite(probs).all()
                np.save(destination/'probabilities.npy',probs)
                np.save(destination/'masks.npy',(probs>=.5).astype(np.uint8))
                np.save(destination/'counts.npy',counts)
                dump(destination/'ids.json',data.ids['val'])
            result[name][criterion]=rec
            print('EXPORTED',name,criterion,'epoch',ckpt['epoch'],'macro Dice',metrics['macro']['dice'],flush=True)
        del model
    assert all(tensor_hash(getattr(encoder,n))==h for n,h in done['encoder_hashes'].items())
    dump(out/'evaluation.json',dict(passed=True,n_val=808,smoke_only=args.smoke,results=result))


if __name__=='__main__':main()
