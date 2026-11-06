"""Spatial error and feature changes caused by the fixed-weight scan switch."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from diagnose import (BATCH, OUT, PROJECT, S19, ImageData, Segmentor,
                      load_trainable, region_targets, save_json, setup)


def main():
    setup();data=ImageData(42);n=len(data.ids['val'])
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    model=Segmentor('region_scan').cuda().eval()
    ck=torch.load(S19/'results/region_scan/best_dice.pth',map_location='cpu')
    load_trainable(model,ck['model'])
    taps={}
    def hook(name):
        def save(_m,_i,o):taps[name]=o.detach()
        return save
    handles=[model.encoder.mixer.register_forward_hook(hook('mixer')),
             model.probe.decoder.blocks[0].register_forward_hook(hook('decoder'))]
    names=('background','boundary','interior')
    total={k:dict(pixels=0,changed=0,corrected=0,introduced=0,prob_abs=0.,
       mixer_delta_weighted=0.,mixer_energy_weighted=0.,mixer_area_units=0.,
       decoder_delta_weighted=0.,decoder_energy_weighted=0.,decoder_area_units=0.) for k in names}
    print('ROI INTERVENTION AUDIT',flush=True)
    with torch.no_grad():
      for start in range(0,n,BATCH):
        end=min(n,start+BATCH);x,y=data.batch('val',torch.arange(start,end,device='cuda'))
        model.encoder.mixer.mode='region_scan'
        with torch.autocast('cuda',dtype=torch.bfloat16):
            fs,_=model.encoder(x);z=model.probe(fs)
        m=model.encoder.mixer.window_map(taps['mixer']).float();d=taps['decoder'].float();p=z.float().sigmoid()
        model.encoder.mixer.mode='ordered_prior'
        with torch.autocast('cuda',dtype=torch.bfloat16):
            fs0,_=model.encoder(x);z0=model.probe(fs0)
        m0=model.encoder.mixer.window_map(taps['mixer']).float();d0=taps['decoder'].float();p0=z0.float().sigmoid()
        fg=(y>=.5);hit=F.conv2d(fg.float(),torch.ones((1,1,5,5),device=fg.device),padding=2)
        bd=(hit>0)^(hit==25);inside=fg&~bd;bg=~bd&~inside
        masks=(bg,bd,inside);regions16=region_targets(y);regions64=F.avg_pool2d(torch.cat([t.float() for t in masks],1),4)
        pred=p>=.5;old=p0>=.5;new_error=pred!=fg;old_error=old!=fg
        mixer_delta=(m-m0).square().mean(1).sqrt();mixer_energy=m.square().mean(1).sqrt()
        decoder_delta=(d-d0).square().mean(1).sqrt();decoder_energy=d.square().mean(1).sqrt()
        for k,name in enumerate(names):
            z=masks[k];r=total[name]
            for key,value in [('pixels',z.sum()),('changed',(z&(pred!=old)).sum()),
               ('corrected',(z&old_error&~new_error).sum()),('introduced',(z&~old_error&new_error).sum())]:
                r[key]+=int(value.cpu())
            r['prob_abs']+=float(((p-p0).abs()*z).sum().cpu())
            for label,field,w in [('mixer_delta_weighted',mixer_delta,regions16[:,k]),
                                  ('mixer_energy_weighted',mixer_energy,regions16[:,k]),
                                  ('decoder_delta_weighted',decoder_delta,regions64[:,k]),
                                  ('decoder_energy_weighted',decoder_energy,regions64[:,k])]:
                r[label]+=float((field*w).sum().cpu())
            r['mixer_area_units']+=float(regions16[:,k].sum().cpu())
            r['decoder_area_units']+=float(regions64[:,k].sum().cpu())
        if start%256==0:print(f' {end}/{n}',flush=True)
    for h in handles:h.remove()
    for r in total.values():
        r['changed_fraction_within_region']=r['changed']/r['pixels']
        r['mean_abs_probability_change']=r['prob_abs']/r['pixels']
        r['mixer_mean_delta_rms']=r['mixer_delta_weighted']/r['mixer_area_units']
        r['mixer_mean_feature_rms']=r['mixer_energy_weighted']/r['mixer_area_units']
        r['decoder_mean_delta_rms']=r['decoder_delta_weighted']/r['decoder_area_units']
        r['decoder_mean_feature_rms']=r['decoder_energy_weighted']/r['decoder_area_units']
    save_json('roi_intervention_audit.json',dict(n=n,seed=42,
       comparison='trained region_scan minus same weights switched at inference to ordered_prior',
       regions='GT background, 5x5 morphological boundary, interior; GT used for analysis only',
       model_intervention_may_be_out_of_distribution=True,regions_summary=total))
    print('FINISHED',json.dumps(total),flush=True)


if __name__=='__main__':main()
