"""Re-evaluate fixed checkpoints on all historical development samples."""
from common import *
from models import DecoderProbe, FrozenEncoder, tensor_hash
from stage3_confirmation.data import ImageData
from train import NAMES, encode_val, evaluate


def main():
    setup()
    out=ROOT/'results'
    done=json.loads((out/'DONE.json').read_text())
    assert done['epochs']==300 and all(v==9000 for v in done['updates'].values())
    data=ImageData(42)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    encoder=FrozenEncoder().cuda().eval()
    assert tensor_hash(encoder)==done['encoder_hash']
    config=json.loads((out/'config.json').read_text())
    features,targets=encode_val(encoder,data,config['encoder_batch'])
    results={}
    for name in NAMES:
        model=DecoderProbe(name).cuda().eval()
        results[name]={}
        for criterion in ('best_dice','best_loss','last'):
            saved=torch.load(out/name/(criterion+'.pth'),map_location='cpu')
            model.load_state_dict(saved['model'],strict=True)
            result,prob,counts=evaluate(model,features,targets,config['evaluation_batch'],export=criterion=='best_dice')
            assert result['counts']==saved['metrics']['counts'],(name,criterion,'pixel counts changed')
            assert abs(result['loss']-saved['metrics']['loss'])<2e-5,(name,criterion,'loss changed')
            record=dict(name=name,criterion=criterion,selected_epoch=saved['epoch'],metrics=result)
            dest=out/name/criterion
            dest.mkdir(exist_ok=True)
            dump(dest/'metrics.json',record)
            if criterion=='best_dice':
                assert prob.shape==(808,256,256) and np.isfinite(prob).all()
                np.save(dest/'probabilities.npy',prob)
                np.save(dest/'masks.npy',(prob>=.5).astype(np.uint8))
                np.save(dest/'counts.npy',counts)
                dump(dest/'ids.json',data.ids['val'])
            results[name][criterion]=record
            print('EXPORTED',name,criterion,saved['epoch'],result['macro']['dice'],flush=True)
        del model
    assert tensor_hash(encoder)==done['encoder_hash']
    dump(out/'evaluation.json',dict(passed=True,n_val=808,results=results))


if __name__=='__main__':main()
