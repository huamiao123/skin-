"""Audit class collapse in both decoder positions and recover compact masks.

This reads trained checkpoints and never alters historical result artifacts.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from diagnose import (BATCH, HERE, OUT, PROJECT, S17, ImageData, Segmentor,
                      append_route, counts, morphology_soft, route_values,
                      save_json, setup, summarize_route, vector)
import stage17_mambavision_decoder_hiram_seed42.models as m17
import stage18_hiram_post_swin32_seed42.models as m18


def main():
    setup()
    data=ImageData(42);ids=data.ids['val'];n=len(ids)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    encoder=Segmentor('native').encoder.cuda().eval()
    specifications=[('stage17_region_only',S17,'region_only',m17),('stage17_hiram',S17,'hiram',m17),
      ('stage18_region_only',PROJECT/'stage18_hiram_post_swin32_seed42','region_only',m18),
      ('stage18_hiram',PROJECT/'stage18_hiram_post_swin32_seed42','hiram',m18)]
    models={};acc={}; refs={};actual={};mismatches={};mismatch_max={};packed={}
    for name,root,mode,mod in specifications:
        ck=torch.load(root/'results'/mode/'best_dice.pth',map_location='cpu')
        net=mod.DecoderProbe(mode).cuda().eval();net.load_state_dict(ck['model'],strict=True)
        models[name]=net;acc[name]=[defaultdict(list),defaultdict(list)];actual[name]=[]
        refs[name]=np.load(root/'results'/mode/'best_dice/probabilities.npy',mmap_mode='r')
        mismatches[name]=0;mismatch_max[name]=0.
        if name.startswith('stage17_'):packed[name]=np.empty((n,256,32),dtype=np.uint8)
    print('AUDIT 4 DECODER ROUTERS',flush=True)
    with torch.no_grad():
      for start in range(0,n,BATCH):
        end=min(n,start+BATCH);image,gt=data.batch('val',torch.arange(start,end,device='cuda'))
        target=morphology_soft(gt,32)
        with torch.autocast('cuda',dtype=torch.bfloat16):f,_=encoder(image)
        for name,root,mode,_ in specifications:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,regions=models[name](f,return_regions=True)
            pred=logits.float().sigmoid();actual[name].append(counts(pred,gt))
            for collector,region in zip(acc[name],regions):
                append_route(collector,route_values(region,target))
            observed=vector(pred[:,0]);old=np.asarray(refs[name][start:end]);
            mismatches[name]+=int(((old>=.5)!=(observed>=.5)).sum())
            mismatch_max[name]=max(mismatch_max[name],float(np.abs(observed-old).max()))
            if name in packed:packed[name][start:end]=np.packbits(observed>=.5,axis=-1)
        if start%256==0:print(f'  {end}/{n}',flush=True)
    results={}
    recovery=OUT/'recovered_masks';recovery.mkdir(exist_ok=True)
    for name,root,mode,_ in specifications:
        actual_count=np.concatenate(actual[name]);saved=np.load(root/'results'/mode/'best_dice/counts.npy')
        assert np.array_equal(actual_count,saved),f'{name} per-image count mismatch'
        source=root/'results'/mode/'best_dice/probabilities.npy'
        results[name]=dict(all_808_per_image_counts_match=True,
          saved_probability_max_abs_difference=mismatch_max[name],
          saved_binary_pixel_mismatch=mismatches[name],
          source_allocated_bytes=source.stat().st_blocks*512,source_logical_bytes=source.stat().st_size,
          coarse=summarize_route(acc[name][0]),skip=summarize_route(acc[name][1]))
        if name in packed:
            dest=recovery/f'{name}_bitpacked_masks.npy'
            np.save(dest,packed[name])
            assert np.load(dest,mmap_mode='r').shape==(808,256,32)
            results[name]['recovered_binary_masks']=str(dest)
    save_json('decoder_router_audit.json',dict(n=n,classes=['background','boundary','interior'],
      area_definition='256 GT mask morphology (5x5, width 2) area-pooled to 32x32',
      generated_without_changing_original_files=True,
      recovered_mask_format='uint8 bit-packed along width, np.unpackbits(..., axis=-1), threshold >=0.5',
      models=results))
    print('FINISHED',json.dumps({k:{'pixel_mismatch':v['saved_binary_pixel_mismatch'],
       'coarse_hard_bg':v['coarse']['classes']['background']['predicted_hard_patch_fraction'],
       'coarse_hard_boundary':v['coarse']['classes']['boundary']['predicted_hard_patch_fraction']}
       for k,v in results.items()}),flush=True)


if __name__=='__main__':main()
