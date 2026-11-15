from pathlib import Path
import ast,json,shutil,hashlib
P=Path('/home/waas/mambavision_stage22/source/referring_image_segmentation')
S=P/'stage33_author_modules';S.mkdir(exist_ok=False)
R=P.parent.parent/'research/topconf_open_source_20261111'
def replace_def(src,name,new):
 tree=ast.parse(src);node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
 lines=src.splitlines(True)
 start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])-1
 return ''.join(lines[:start])+new.rstrip()+'\n\n'+''.join(lines[node.end_lineno:])
shutil.copy2(P/'stage32_main_route_attention/identity_partition.json',S/'identity_partition.json')
refs=json.loads((P/'stage32_main_route_attention/REFERENCE_MODELS.json').read_text())
refs['models']=[x for x in refs['models'] if x['arm'] in ['control','local_fusion']]
(S/'REFERENCE_MODELS.json').write_text(json.dumps(refs,indent=2))
author=(R/'integration_prototypes.py').read_text().replace('R=Path(__file__).resolve().parent',f'R=Path({str(R)!r})')
author=author.replace('Research wiring only. Author module algorithms and production files are unchanged.','Stage33 wiring: exact author modules; existing archived training files unchanged.')
(S/'author_modules.py').write_text(author)
protocol=dict(stage=S.name,status='fixed_before_real_data_smoke',seed=42,arms=['emcad64','swin_cga'],
 rationale='Use actual published author implementations after Stage29-32 custom fusion did not show stable gains; vetted in research/topconf_open_source_20261111.',
 changes={'emcad64':'Replace Swin by original EMCAD(channels=[64]*4), retain adapters and readout initialization.',
          'swin_cga':'Retain original Swin weights; original MambaLiteUNet CGA(64,64,32) at 16/32/64 skip additions.'},
 authors_internal_code_modified=False,extra_supervision=False,
 training=dict(epochs=100,lr_horizon=300,lr_note='First100 of archived300 cosine kept to match archived controls; not an optimized100-epoch cosine.',micro_batch=48,effective_batch=64,encoder_lr=1e-4,decoder_lr=1e-3,weight_decay=.01,precision='BF16 forward FP32 loss',encoder='late two stages trainable, all encoder modules eval; author decoder BN remains train in training'),
 split=dict(train=1886,val=762,test=867,test_already_exposed=True,patient_disjointness_verified=False),
 selection='max validation per-image foreground Dice, earliest tie; threshold0.5 fixed',
 completion='Two-epoch real-data process restart and validation replay before100-epoch formal run. Train both fixed arms before testing either. Full867 test on fixed best-Dice weights.',
 retention='Keep best-Dice model; latest optimizer state only while unfinished. Best-loss value recorded without extra model copy. Remove smoke model files after final success.',
 metrics='Common evaluation.py macro and pooled Dice/IoU/precision/recall, Prec@ thresholds, boundary F1, HD95, threshold Jaccard and per-image rows.',
 references='Archived Stage29 control/local first100 reused only with unchanged protocol and replayed validation; no test-based next-stage selection.',
 next_step_screen='Validation only: require >=0.2 percentage point macro Dice improvement over strongest archived validation of the two references and no increase in severe undercoverage before seed confirmation; exploratory screen, not a publication criterion.')
(S/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
src=(P/'stage32_main_route_attention/train.py').read_text()
src=src.replace('Stage32: one controlled change to channel-attention application location.','Stage33: independently train original EMCAD and CGA adaptations.')
src=src.replace('from stage30_channel_fusion.channel_fusion import ChannelDecoder\nfrom main_route import MainRouteDecoder','from author_modules import AuthorModuleDecoder')
src=src.replace("NAMES = ('main_context_channel',)","ARMS = ('emcad64','swin_cga')\nNAMES = ARMS")
src=replace_def(src,'System',"""class System(base.System):
    def __init__(self,arm='emcad64'):
        super().__init__(finetune=True,stochastic_depth=False)
        assert arm in ARMS
        self.arm=arm
        self.probe.decoder=AuthorModuleDecoder(arm,self.probe.decoder,seed=42)
        self.train()
""")
src=replace_def(src,'preflight',"""def preflight(models,cache,data,val_cache,micro,out):
    initial={n:cpu_tree(m.state_dict()) for n,m in models.items()}
    reference=base.System(finetune=True,stochastic_depth=False).cuda()
    ids=torch.arange(8,device='cuda');images,_=data.batch('train',ids)
    reference.eval()
    shared={}
    for name,m in models.items():
        m.eval()
        assert all(torch.equal(v,m.encoder.state_dict()[k]) for k,v in reference.encoder.state_dict().items())
        assert all(torch.equal(v,m.probe.adapter.state_dict()[k]) for k,v in reference.probe.adapter.state_dict().items())
        if name=='swin_cga':
            base.assert_tree_exact(m.probe.decoder.baseline.state_dict(),reference.probe.decoder.state_dict())
        else:
            base.assert_tree_exact(m.probe.decoder.readout.state_dict(),reference.probe.decoder.readout.state_dict())
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            _,official=m.encoder(images)
            f1,f2,mid=m.prefix(images);f3,f4=m.suffix(mid)
            assert all(torch.equal(a,b) for a,b in zip(official,[f1,f2,f3,f4]))
            prediction=m(images)
            assert prediction.shape==(8,1,256,256) and torch.isfinite(prediction).all()
        shared[name]=dict(encoder_and_adapters_match=True,shared_swin_or_readout_match=True,official_encoder_features_identical=True)
    del reference,images,prediction,official,f1,f2,mid,f3,f4
    gc.collect();torch.cuda.empty_cache()
    checks={}
    for name,model in models.items():
        model.train();opt=base.optimizer(model);frozen=base.frozen_hash(model)
        assert all(not mod.training for mod in model.encoder.modules())
        for _ in range(3):
            loss,norms=base.update(model,opt,cache,torch.arange(64,device='cuda'),micro,check=True)
        assert base.frozen_hash(model)==frozen
        grads={k:float(p.grad.abs().sum()) for k,p in model.probe.decoder.named_parameters() if p.requires_grad}
        assert len(grads)==sum(p.requires_grad for p in model.probe.decoder.parameters())
        assert all(np.isfinite(v) for v in grads.values())
        if name=='swin_cga':
            for i in (2,1,0):
                for group in ['mamba_g','mamba_x','combine']:
                    assert sum(v for k,v in grads.items() if k.startswith(f'author_gates.{i}.{group}'))>0
        else:assert sum(v for k,v in grads.items() if k.startswith('author_decoder.'))>0
        state=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=base.rng())
        expected_loss,_=base.update(model,opt,cache,torch.arange(64,128,device='cuda'),micro)
        expected=cpu_tree(model.state_dict())
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);base.set_rng(state['rng'])
        base.assert_tree_exact(model.state_dict(),state['model']);base.assert_tree_exact(opt.state_dict(),state['optimizer'])
        replay_loss,_=base.update(model,opt,cache,torch.arange(64,128,device='cuda'),micro)
        assert expected_loss==replay_loss,(expected_loss,replay_loss)
        replay={}
        for group,prefix in [('encoder','encoder.'),('decoder','probe.')]:
            d2=u2=max_abs=0.
            for k,v in model.state_dict().items():
                if not k.startswith(prefix):continue
                actual=v.detach().cpu();delta=(actual-expected[k]).double()
                d2+=float(delta.square().sum());u2+=float((expected[k]-state['model'][k]).double().square().sum())
                max_abs=max(max_abs,float(delta.abs().max()))
            rel=(d2/max(u2,1e-30))**.5
            assert rel<=.01 and max_abs<=.0001,(name,group,rel,max_abs)
            replay[group]=dict(relative_l2=rel,max_abs=max_abs)
        checks[name]=dict(shared=shared[name],real_data_loss=loss,gradient_norms=norms,author_gradient_l1=grads,
                         frozen_unchanged=True,restored_state_exact=True,replay_loss_exact=True,replay_update=replay,
                         note='Mamba CUDA backward may not be bitwise deterministic; <=1% update-L2 and1e-4 max_abs tolerance.')
        del opt,state,expected;model.zero_grad(set_to_none=True)
    for n,m in models.items():m.load_state_dict(initial[n])
    base.setup();dump(out/'preflight.json',dict(passed=True,checks=checks))
""")
src=replace_def(src,'source_hashes',"""def source_hashes():
    research=PROJECT.parent.parent/'research'
    files=[ROOT/'train.py',ROOT/'author_modules.py',ROOT/'evaluate.py',ROOT/'run_pipeline.py',ROOT/'PROTOCOL.json',
           ROOT/'identity_partition.json',ROOT/'REFERENCE_MODELS.json',ROOT/'verify_reference.py',
           PROJECT/'stage28_drop_path/train.py',PROJECT/'evaluation.py',
           PROJECT/'stage1_visual/data.py',PROJECT/'stage1_visual/models.py',
           PROJECT/'stage2_decoder/models.py',PROJECT/'stage3_confirmation/models.py',
           PROJECT/'stage3_confirmation/data.py',base.HF/'config.json',
           research/'module_design_20261111/official_code/SLDGroup__EMCAD/lib/decoders.py',
           research/'topconf_open_source_20261111/official_code/maklachur__MambaLiteUNet/models/MambaLiteUNet.py']+list(base.HF.glob('*.py'))
    return {os.path.relpath(p,PROJECT):digest(p) for p in files}
""")
src=src.replace("ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')",
"""global NAMES
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--arm',required=True,choices=ARMS)""")
src=src.replace("args=ap.parse_args()\n    base.setup();out=ROOT/('smoke' if args.smoke else 'results');out.mkdir(exist_ok=True)",
"""args=ap.parse_args();NAMES=(args.arm,)
    base.setup();out=ROOT/args.arm/('smoke' if args.smoke else 'results');out.mkdir(parents=True,exist_ok=True)""")
src=src.replace("ROOT/'smoke","ROOT/args.arm/'smoke")
src=src.replace("reference_reuse='Stage30 cross_channel_scale and Stage29 reference first100 schedule and batch matched; all validation replay verified'","reference_reuse='Stage29 control/local first100 schedule and batch matched; validation replay required before formal training'")
src=src.replace("stage='stage32_main_route_attention'","stage='stage33_author_modules',arm=args.arm")
src=src.replace("""                    save(out/f'{n}_{criterion}.pth',dict(epoch=epoch,model=cpu_tree(models[n].state_dict()),metric=metric,config=config))""",
"""                    if criterion=='best_dice':
                        save(out/f'{n}_{criterion}.pth',dict(epoch=epoch,model=cpu_tree(models[n].state_dict()),metric=metric,config=config))""")
src=src.replace("""    for n in NAMES:
        save(out/f'{n}_last.pth',dict(epoch=config['epochs'],model=cpu_tree(models[n].state_dict()),config=config))
""","")
src=src.replace("""        out=ROOT/('smoke' if '--smoke' in sys.argv else 'results')""",
"""        arm=sys.argv[sys.argv.index('--arm')+1] if '--arm' in sys.argv else 'unknown'
        out=ROOT/arm/('smoke' if '--smoke' in sys.argv else 'results')""")
assert 'MainRouteDecoder' not in src and "ROOT/args.arm/'smoke" in src
ast.parse(src);(S/'train.py').write_text(src)
print('CREATED',S,flush=True)

