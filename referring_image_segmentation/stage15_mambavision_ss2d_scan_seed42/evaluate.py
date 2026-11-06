"""Export all three fixed selectors with historical full metrics and dense outputs."""
from common import *
from models import FrozenEncoder,build_probe,tensor_hash
from data import ImageData
from engine import encode,evaluate
from stage1_visual.analyze import metrics as image_metrics
from evaluation import summarize_rows
from concurrent.futures import ThreadPoolExecutor
import argparse,csv

def csv_rows(path,rows):
    with Path(path).open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    setup();verify_frozen_sources();out=ROOT/('smoke' if args.smoke else 'results')
    done=json.loads((out/'DONE.json').read_text())
    assert done['epochs']==(2 if args.smoke else 300) and done['updates']==(60 if args.smoke else 9000)
    if args.smoke:assert done['resumed'] and done['resume_audit']['python_numpy_cpu_cuda_rng_exact']
    encoder=FrozenEncoder().cuda();probe=build_probe().cuda();data=ImageData()
    assert tensor_hash(encoder)==done['initial_encoder_hash']==done['final_encoder_hash']
    features,targets=encode(encoder,data,'val',64)
    gt=(targets[:,0]>=.5).cpu().numpy();all_metrics={};checks=[]
    analysis=ROOT/('smoke_analysis' if args.smoke else 'analysis');analysis.mkdir(exist_ok=True)
    history=json.loads((out/'history.json').read_text())
    expected_epochs=dict(best_dice=max(history,key=lambda r:r['val']['macro']['dice'])['epoch'],
        best_loss=min(history,key=lambda r:r['val']['loss'])['epoch'],last=done['epochs'])
    for criterion in CRITERIA:
        checkpoint=torch.load(out/f'{criterion}.pth',map_location='cpu')
        assert checkpoint['criterion']==criterion and checkpoint['epoch']==expected_epochs[criterion]
        assert checkpoint['encoder_hash']==tensor_hash(encoder)
        assert sha(out/f'{criterion}.pth')==done['checkpoints'][criterion]['sha256']
        probe.load_state_dict(checkpoint['probe'],strict=True)
        region,probabilities,_=evaluate(probe,features,targets,export=True)
        assert region==checkpoint['metrics'],'Saved selector scores must reproduce exactly'
        assert probabilities.shape==(808,256,256) and np.isfinite(probabilities).all()
        indices=range(16 if args.smoke else 808)
        def sample(i):
            prediction=probabilities[i]>=.5;truth=gt[i]
            row=image_metrics(prediction,truth)
            # Separate TP/FP/FN calculation and explicit Dice/IoU formula.
            tp=int(np.logical_and(prediction,truth).sum())
            fp=int(np.logical_and(prediction,np.logical_not(truth)).sum())
            fn=int(np.logical_and(np.logical_not(prediction),truth).sum())
            tn=int(np.logical_and(np.logical_not(prediction),np.logical_not(truth)).sum())
            assert (tp,fp,fn,tn)==tuple(row[k] for k in ('tp','fp','fn','tn'))
            dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
            iou=tp/(tp+fp+fn) if tp+fp+fn else 1.
            assert abs(row['dice']-dice)<1e-12 and abs(row['iou']-iou)<1e-12
            return dict(id=data.ids['val'][i],index=i,foreground_fraction=float(truth.mean()),**row)
        with ThreadPoolExecutor(8) as pool:rows=list(pool.map(sample,indices))
        complete=summarize_rows(rows)
        if not args.smoke:
            for group in ('macro','pooled'):
                for key in ('dice','iou','precision','recall'):assert abs(complete[group][key]-region[group][key])<1e-12
            assert complete['counts']==region['counts']
        complete['loss']=region['loss'];complete['selected_epoch']=checkpoint['epoch']
        complete['checkpoint_sha256']=sha(out/f'{criterion}.pth')
        complete['evaluation_resolution']='historical 256x256 grid; not native-image evaluation'
        destination=out/criterion;destination.mkdir(exist_ok=True)
        if not args.smoke:
            np.save(destination/'probabilities.npy',probabilities)
            np.save(destination/'masks.npy',(probabilities>=.5).astype(np.uint8))
            dump(destination/'ids.json',data.ids['val'])
            complete['probability_sha256']=sha(destination/'probabilities.npy')
        dump(analysis/f'per_image_{criterion}.json',rows)
        csv_rows(analysis/f'per_image_{criterion}.csv',rows)
        dump(destination/'metrics.json',complete);all_metrics[criterion]=complete
        if criterion=='best_dice':
            csv_rows(analysis/'per_image.csv',rows)
        checks.append(dict(criterion=criterion,n=len(rows),selected_epoch=checkpoint['epoch'],
            selector_matches_history=True,saved_region_metrics_reproduced_exactly=True,independent_counts_dice_iou_passed=True))
        print('EVALUATED',criterion,complete['macro']['dice'],complete['pooled']['dice'],flush=True)
    assert tensor_hash(encoder)==done['final_encoder_hash'];verify_frozen_sources()
    dump(analysis/'metrics.json',all_metrics)
    dump(analysis/'evaluation_validation.json',dict(passed=True,smoke_only=args.smoke,checks=checks,
        encoder_parameters_and_buffers_unchanged=True,all_three_checkpoints_retained=True))
    print('EVALUATION_PASSED',flush=True)

if __name__=='__main__':main()
