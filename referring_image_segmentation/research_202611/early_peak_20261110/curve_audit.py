import json, math, pathlib, hashlib, datetime
root=pathlib.Path('/home/waas/mambavision_stage22')
p=root/'source/referring_image_segmentation'
out=root/'research/early_peak_20261110'
result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'train_images':1886,'val_images':762,'batch_size':64,'steps_per_epoch':30,'runs':[]}
curves=[]
for stage in ['stage22_partial_finetune','stage23_encoder_lr3e5','stage24_seed43_confirmation','stage27_coverage_supervision']:
 f=p/stage/'results/history.json'; raw=f.read_bytes(); hist=json.loads(raw)
 for arm in hist[0]['validation']:
  if stage=='stage23_encoder_lr3e5' and arm=='frozen': continue
  best=max(hist,key=lambda x:x['validation'][arm]['macro']['dice'])
  bl=min(hist,key=lambda x:x['validation'][arm]['loss'])
  last=hist[-1]; score=best['validation'][arm]['macro']['dice']
  near=[h['epoch'] for h in hist if score-h['validation'][arm]['macro']['dice']<=.001]
  r=dict(stage=stage,arm=arm,history_sha256=hashlib.sha256(raw).hexdigest(),observed_epochs=len(hist),completed=len(hist)==300,best_dice_epoch=best['epoch'],best_dice=score,best_dice_updates=30*best['epoch'],best_loss_epoch=bl['epoch'],best_loss=bl['validation'][arm]['loss'],loss_at_best_dice=best['validation'][arm]['loss'],last_dice=last['validation'][arm]['macro']['dice'],last_loss=last['validation'][arm]['loss'],last_train_loss=last['train_loss'][arm],train_loss_at_best_dice=best['train_loss'][arm],last_minus_best_dice_pp=100*(last['validation'][arm]['macro']['dice']-score),within_point1pp={'count':len(near),'first':min(near),'last':max(near)},lr_at_best=best['learning_rates'][arm])
  result['runs'].append(r);curves.append((stage,arm,hist))
  print(stage,arm,'best',r['best_dice_epoch'],round(score*100,4),'lossbest',bl['epoch'],'last',round(r['last_dice']*100,4),'near0.1pp',r['within_point1pp'])
result['schedule_factors']={str(e):.01+.99*(1+math.cos(math.pi*(e-1)/300))/2 for e in [7,13,16,18,30,60,100,150,200,300]}
(out/'curve_evidence.json').write_text(json.dumps(result,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for stage,arm,hist in curves:
 if stage not in ['stage22_partial_finetune','stage24_seed43_confirmation']: continue
 seed=42 if '22_' in stage else 43
 label=f'seed {seed} / {arm}'
 e=[x['epoch'] for x in hist]
 axs[0,0].plot(e,[x['validation'][arm]['macro']['dice']*100 for x in hist],label=label,linewidth=1)
 axs[0,1].plot(e,[x['validation'][arm]['loss'] for x in hist],label=label,linewidth=1)
 axs[1,0].plot(e,[x['train_loss'][arm] for x in hist],label=label,linewidth=1)
axs[0,0].set(title='Validation macro Dice',ylabel='Dice (%)',ylim=(86.5,89.2))
axs[0,1].set(title='Validation loss (BCE + soft Dice)',ylabel='Loss')
axs[1,0].set(title='Augmented training loss',ylabel='Loss',ylim=(0,.3))
e=list(range(1,301))
axs[1,1].plot(e,[.01+.99*(1+math.cos(math.pi*(x-1)/300))/2 for x in e])
axs[1,1].axvspan(7,18,alpha=.2,color='orange',label='fine-tuning best epochs 7-18')
axs[1,1].set(title='Actual cosine learning-rate factor',ylabel='Fraction of initial LR')
for a in axs.flat:
 a.set_xlabel('Epoch (30 optimizer updates per epoch)');a.grid(alpha=.2)
axs[0,0].legend(fontsize=8);axs[1,1].legend(fontsize=8)
fig.suptitle('Locked validation; primary loss only. Train and validation use different augmentation.')
fig.savefig(out/'training_dynamics.png',dpi=180);fig.savefig(out/'training_dynamics.pdf')
plt.close(fig)
config=json.loads((p/'stage14_mambavision_s_isic2018_seed42/official/hf/config.json').read_text())
print('HFCONFIG', {k:v for k,v in config.items() if 'drop' in k or k in ['depths','dim','model_type']})
print('PARAMS', (p/'stage22_partial_finetune/results/parameters.json').read_text())
