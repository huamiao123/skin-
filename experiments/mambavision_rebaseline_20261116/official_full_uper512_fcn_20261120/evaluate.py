import csv
import json
import statistics
from official import *
def measure(net,data,part):
    score,rows=recipe.evaluate(net,data,part)
    assert [x[0] for x in rows]==data.ids[part]
    assert abs(statistics.mean(v for _,v in rows)-score)<1e-12
    return score,rows

def result(net,ds,label,epoch,model_sha):
    dest=ROOT/ds
    dest.mkdir(exist_ok=True)
    data=dataset(ds,['test'])
    net.eval()
    stamp={k:v._version for k,v in net.state_dict().items()}
    score,rows=measure(net,data,'test')
    assert all(v._version==stamp[k] for k,v in net.state_dict().items())
    assert all(p.grad is None for p in net.parameters())
    path=dest/(label+'_per_image.csv')
    tmp=path.with_suffix('.new')
    with tmp.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image_id','foreground_dice']);w.writerows(rows)
    tmp.replace(path)
    write(dest/(label+'.json'),dict(epoch=epoch,model_sha256=model_sha,split_sha256=SPLITS[ds]['split_sha256'],
            test_macro_dice=score,n=len(rows),threshold=.5,per_image_sha256=sha(path)))
    print('TEST',label,ds,len(rows),score,flush=True)

if __name__=='__main__':
    initialize()
    shared=json.loads((ROOT/'initialization.json').read_text())['initial_sha256']
    net=load_model()
    for ds in DATASETS:
        f=ROOT/ds/'epoch0_test.json'
        if f.exists():
            old=json.loads(f.read_text())
            assert old['model_sha256']==shared and old['per_image_sha256']==sha(f.parent/'epoch0_test_per_image.csv')
        else:
            result(net,ds,'epoch0_test',0,shared)
