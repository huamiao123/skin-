"""Keep frequent optimizer checkpoints on the server's fast disk. Publish verified results to work."""
from pathlib import Path
import json,os,sys,time,shutil,fcntl
import matched_core as core
from prepare_matched import BASE,SCRATCH,sha

def copy_atomic(src,dst):
    tmp=dst.with_name(dst.name+'.copying')
    shutil.copyfile(src,tmp)
    os.replace(tmp,dst)

def main(dataset,arm):
    dst=BASE/dataset/arm
    dst.mkdir(parents=True,exist_ok=True)
    local=SCRATCH/'training'
    job=local/dataset/arm
    job.mkdir(parents=True,exist_ok=True)
    split=job.parent/'split.csv'
    if not split.exists():split.symlink_to(BASE/dataset/'split.csv')
    assert split.resolve()==(BASE/dataset/'split.csv').resolve()
    # Continue an interrupted run from its last committed server snapshot.
    if not (job/'resume.pth').exists() and not (job/'test_result.json').exists():
        snapshot=dst/'recovery'
        if snapshot.is_dir():
            for f in snapshot.iterdir():copy_atomic(f,job/f.name)
    publish=core.recipe.write_json
    last=time.monotonic()
    def record(path,value):
        nonlocal last
        publish(path,value)
        if path.name in ('config.json','first_update.json','history.json'):
            publish(dst/path.name,value)
        # The trainer commits a resume checkpoint every 5 epochs. Copy a consistent pair.
        if path.name=='history.json' and value[-1]['epoch']%5==0 and time.monotonic()-last>=240:
            best=max(value,key=lambda r:r['val_macro_dice'])['epoch']
            tmp=dst/'recovery.next';tmp.mkdir(exist_ok=True)
            for nm in ('resume.pth',f'best_epoch_{best}.pth','config.json'):
                copy_atomic(job/nm,tmp/nm)
            old=dst/'recovery.old'; prev=dst/'recovery'
            if old.exists():shutil.rmtree(old)
            if prev.exists():prev.rename(old)
            tmp.rename(prev)
            if old.exists():shutil.rmtree(old)
            last=time.monotonic()
    source_link=local/'prepare_matched.py'
    if not source_link.exists():source_link.symlink_to(BASE/'prepare_matched.py')
    core.BASE=local
    core.recipe.write_json=record
    with (dst/'fit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        core.run(dataset,arm)
        done=json.loads((job/'fit_complete.json').read_text())
        best=done['best_checkpoint']
        assert sha(job/best)==done['best_sha256']
        for nm in (best,'config.json','first_update.json','history.json','fit_complete.json','test_per_image.csv'):
            copy_atomic(job/nm,dst/nm)
        assert sha(dst/best)==done['best_sha256']
        copy_atomic(job/'test_result.json',dst/'test_result.json')
        for name in ('recovery','recovery.old','recovery.next'):
            tmp=dst/name
            if tmp.exists():shutil.rmtree(tmp)
if __name__=='__main__':
    main(*sys.argv[1:])
