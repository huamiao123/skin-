"""Evaluate all fixed checkpoints; export best-Dice per-image predictions."""
from hgm_model import *
from engine import encode_val, evaluate
from stage17_mambavision_decoder_hiram_seed42.models import FrozenEncoder
from stage3_confirmation.models import tensor_hash
from stage3_confirmation.data import ImageData


def main():
    setup()
    local=Path(__file__).resolve().parent
    out=local/'results'
    done=json.loads((out/'DONE.json').read_text())
    assert done['epochs']==300 and done['updates']==9000 and done['encoder_unchanged']
    data=ImageData(42)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    encoder=FrozenEncoder().cuda().eval()
    assert tensor_hash(encoder)==done['encoder_hash']
    features,targets=encode_val(encoder,data,64)
    model=MambaVisionHGMProbe().cuda().eval()
    results={}
    for criterion in ('best_dice','best_loss','last'):
        saved=torch.load(out/(criterion+'.pth'),map_location='cpu')
        model.load_state_dict(saved['model'],strict=True)
        result,prob,counts=evaluate(model,features,targets,64,export=criterion=='best_dice')
        assert result['counts']==saved['metrics']['counts'],(criterion,'pixel counts changed')
        assert abs(result['loss']-saved['metrics']['loss'])<2e-5,(criterion,'loss changed')
        rec=dict(criterion=criterion,selected_epoch=saved['epoch'],metrics=result)
        dest=out/criterion
        dest.mkdir(exist_ok=True)
        dump(dest/'metrics.json',rec)
        if criterion=='best_dice':
            assert prob.shape==(808,256,256) and np.isfinite(prob).all()
            np.save(dest/'probabilities.npy',prob)
            np.save(dest/'masks.npy',(prob>=.5).astype(np.uint8))
            np.save(dest/'counts.npy',counts)
            dump(dest/'ids.json',data.ids['val'])
        results[criterion]=rec
        print('EXPORTED',criterion,saved['epoch'],result['macro']['dice'],flush=True)
    assert tensor_hash(encoder)==done['encoder_hash']
    dump(out/'evaluation.json',dict(passed=True,n_val=808,results=results))


if __name__=='__main__':main()
