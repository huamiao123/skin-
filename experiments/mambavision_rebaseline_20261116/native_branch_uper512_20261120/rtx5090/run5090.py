"""Keep the previous trainer intact and apply only the device/cache migration."""
import json,os,sys
from pathlib import Path
from resident_images import HERE,Base,ResidentImages,official
import train_branch
method=os.environ.get('MV_LOADER','resident')
official.MappedImages=ResidentImages if method=='resident' else Base
original=(HERE.parent/'train_branch.py').read_text()
anchor="restore_rng(ck['rng']);hist=ck['history'];best=ck['best'];del ck"
assert original.count(anchor)==1
code=original.replace(anchor,anchor+"""
        migration=json.loads((OUT/'rtx5090'/'check5090.json').read_text())
        if ds=='ISIC2018' and len(hist)==migration['resume_epoch'] and best['epoch']==migration['old_best_epoch']:
            if abs(best['score']-migration['old_best_validation'])>1e-9:
                raise RuntimeError('Resume checkpoint changed since migration check')
            best=dict(best,score=migration['same_weights_new_gpu_validation'])
            print('RESUMED VERIFIED 5090: checkpoint epoch',len(hist),'next epoch',len(hist)+1,flush=True)""")
space=dict(__file__=str(HERE.parent/'train_branch.py'),__name__='adapted_trainer')
exec(compile(code,str(HERE.parent/'train_branch.py'),'exec'),space)
space['run'](sys.argv[1],64)

