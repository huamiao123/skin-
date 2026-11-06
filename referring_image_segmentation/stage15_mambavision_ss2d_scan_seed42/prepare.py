"""Prove that the completed original baseline can be reused without retraining."""
from common import *

def main():
    base=json.loads((BASELINE/'config.json').read_text());new=json.loads((ROOT/'config.json').read_text())
    keys=['seed','epochs','channels','input_size','train_samples','validation_samples','batch','encoder_batch',
        'evaluation_batch','updates_per_epoch','total_updates','drop_last','gradient_accumulation',
        'optimizer','lr','weight_decay','betas','eps','scheduler','T_max','eta_min','warmup','scheduler_step',
        'loss','clip','precision','frozen_encoder','augmentation_seed','normalization','data_processing',
        'evaluation','primary_selection','secondary_selection','final_selection','encoder_initial_hash','probe_initial_hash','parameters']
    assert all(base[k]==new[k] for k in keys),[(k,base[k],new[k]) for k in keys if base[k]!=new[k]]
    for key in ('torch','cuda_runtime','timm','transformers','mamba-ssm','selective_scan_cuda_sha256'):
        assert base['environment'][key]==new['environment'][key]
    assert json.loads((BASELINE/'validation.json').read_text())['passed']
    done=json.loads((BASELINE/'results/DONE.json').read_text());assert done['epochs']==300 and done['updates']==9000
    summary=json.loads((BASELINE/'analysis/summary.json').read_text());ref=summary['models']['best_dice']
    assert sha(BASELINE/'results/best_dice.pth')==ref['checkpoint_sha256']
    assert sha(BASELINE/'results/best_dice/probabilities.npy')==ref['probability_sha256']
    dump(ROOT/'baseline_reuse.json',dict(passed=True,baseline_stage=str(BASELINE),baseline_retrained=False,
        all_matched_config_fields=keys,pretrained_state_identical=True,trainable_initialization_identical=True,
        total_and_trainable_parameters_identical=True,baseline_selected_epoch=ref['selected_epoch'],
        baseline_metrics=ref,source_config_sha256=sha(BASELINE/'config.json'),
        source_report_sha256=sha(BASELINE/'analysis/summary.json'),
        reference_probabilities_sha256=sha(BASELINE/'results/best_dice/probabilities.npy'),
        only_treatment='seven SSM branches use VMamba shared four-route scan/mean-merge'))
    print('ORIGINAL_BASELINE_REUSE_VERIFIED',flush=True)

if __name__=='__main__':main()
