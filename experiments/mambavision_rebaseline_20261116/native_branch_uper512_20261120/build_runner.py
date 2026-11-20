from pathlib import Path
d=Path.cwd();base=d.parent/'official_full_uper512_fcn_20261120'
s=(base/'fit.py').read_text()
def once(before,after):
    global s
    assert s.count(before)==1,before
    s=s.replace(before,after)
once('from official import *','from branch_model import OUT, load_branch_model, method_config\nfrom official import *')
once("result=evaluation['result']","result=evaluation['result']\nresult.__globals__['ROOT']=OUT")
once("    folder=ROOT/ds","    folder=OUT/ds\n    folder.mkdir(exist_ok=True)")
once("    if (folder/'config.json').exists()", "    cfg['method']=method_config()\n    if (folder/'config.json').exists()")
start=s.index('def smoke(');s=s[:start]
once('    net=load_model()','    net=load_branch_model()')
s+="\nif __name__=='__main__':\n    parser=argparse.ArgumentParser()\n    parser.add_argument('dataset',choices=DATASETS)\n    run(parser.parse_args().dataset,64)\n"
(d/'train_branch.py').write_text(s)

