import ast, gc, hashlib, importlib.util, io, json, sys, unittest
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from model import NativeHeads,SegmentationSystem,segmentation_loss,objective,PROJECT,sha256
sys.path.insert(0,str(PROJECT))
from evaluation import aggregate_counts

torch.set_num_threads(8)

def features(size,batch=1,requires_grad=False):
    return [torch.randn(batch,c,size//stride,size//stride,requires_grad=requires_grad) for c,stride in zip((96,192,384,768),(4,8,16,32))]

class StageATest(unittest.TestCase):
    def test_01_real_native_heads_and_shapes(self):
        for arm in ('A-64-noaux','B-512-noaux','C-512-aux'):
            print('SHAPES',arm,flush=True)
            head=NativeHeads(arm,42).eval()
            self.assertEqual(head.decode_head.channels,64 if arm.startswith('A') else 512)
            if arm.startswith('C'):
                self.assertEqual(head.auxiliary_head.in_channels,384)
                self.assertEqual(head.auxiliary_head.in_index,2)
                self.assertEqual(head.auxiliary_head.channels,256)
                self.assertEqual(head.auxiliary_head.num_convs,1)
                self.assertFalse(head.auxiliary_head.concat_input)
            for size in (256,512):
                with torch.inference_mode(): out=head(features(size),(size,size))
                self.assertEqual(set(out),{'main'})
                self.assertEqual(tuple(out['main'].shape),(1,1,size,size))
                self.assertTrue(torch.isfinite(out['main']).all())
                print('SHAPE_OK',arm,size,flush=True)
                del out;gc.collect()
            del head;gc.collect()

    def test_01b_native_reference_initialization(self):
        spec=importlib.util.spec_from_file_location('historical_stage45_author',PROJECT/'stage45_upernet_decoder/author_modules.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        legacy=module.NativeUPerProbe(64,seed=42).decoder
        current=NativeHeads('A-64-noaux',42).decode_head
        self.assertEqual(set(legacy.state_dict()),set(current.state_dict()))
        self.assertTrue(all(torch.equal(v,current.state_dict()[k]) for k,v in legacy.state_dict().items()))
        del legacy,current;gc.collect()
        b=NativeHeads('B-512-noaux',42);c=NativeHeads('C-512-aux',42)
        self.assertTrue(all(torch.equal(v,c.decode_head.state_dict()[k]) for k,v in b.decode_head.state_dict().items()))

    def test_02_seed_and_auxiliary_gradient(self):
        a=NativeHeads('A-64-noaux',42); b=NativeHeads('A-64-noaux',42); c=NativeHeads('A-64-noaux',43)
        self.assertTrue(torch.equal(a.decode_head.conv_seg.weight,b.decode_head.conv_seg.weight))
        self.assertFalse(torch.equal(a.decode_head.conv_seg.weight,c.decode_head.conv_seg.weight))
        del a,b,c;gc.collect()
        head=NativeHeads('C-512-aux',42).train()
        inp=features(64,batch=2,requires_grad=True); out=head(inp,(64,64));self.assertEqual(set(out),{'main','aux'})
        target=torch.rand(2,1,64,64); total,parts=objective(out,target)
        self.assertTrue(torch.allclose(total,parts['main']+.4*parts['aux']))
        total.backward()
        self.assertGreater(float(head.decode_head.conv_seg.weight.grad.abs().sum()),0)
        self.assertGreater(float(head.auxiliary_head.conv_seg.weight.grad.abs().sum()),0)
        self.assertGreater(float(inp[2].grad.abs().sum()),0)
        self.assertGreater(float(inp[3].grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(total))

    def test_03_prefix_frozen_suffix_trainable(self):
        class Level(nn.Module):
            def __init__(self,a,b): super().__init__();self.conv=nn.Conv2d(a,b,3,stride=2,padding=1);self.bn=nn.BatchNorm2d(b)
            def forward(self,x):
                y=F.relu(self.bn(self.conv(x)));return y,y
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.model=nn.Module(); self.model.patch_embed=nn.Conv2d(3,48,3,stride=2,padding=1)
                self.model.levels=nn.ModuleList([Level(48,96),Level(96,192),Level(192,384),Level(384,768)])
        m=SegmentationSystem(Toy(),'C-512-aux').train()
        before=m.encoder.model.levels[0].bn.running_mean.clone()
        out=m(torch.randn(2,3,64,64));objective(out,torch.rand(2,1,64,64))[0].backward()
        for i,l in enumerate(m.encoder.model.levels):
            grads=[p.grad for p in l.parameters()]
            if i<2:self.assertTrue(all(g is None for g in grads))
            else:self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in grads))
        self.assertTrue(torch.equal(before,m.encoder.model.levels[0].bn.running_mean))
        self.assertFalse(m.encoder.training)

    def test_04_loss_matches_original(self):
        src=(PROJECT/'stage1_visual/models.py').read_text(); tree=ast.parse(src)
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='segmentation_loss')
        scope={'F':F};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'legacy_loss','exec'),scope)
        logits=torch.randn(3,1,7,9,requires_grad=True); target=torch.rand_like(logits)
        expected=scope['segmentation_loss'](logits,target);actual=segmentation_loss(logits,target)
        self.assertTrue(torch.equal(expected,actual))
        a=torch.autograd.grad(actual,logits,retain_graph=True)[0];b=torch.autograd.grad(expected,logits)[0]
        self.assertTrue(torch.equal(a,b))

    def test_05_cpu_checkpoint_resume(self):
        torch.manual_seed(13); m=NativeHeads('A-64-noaux',42).train(); opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
        inp=features(32,batch=2); target=torch.rand(2,1,32,32)
        def step():
            opt.zero_grad(set_to_none=True); loss=objective(m(inp,(32,32)),target)[0]; loss.backward();opt.step();return loss.detach()
        step(); buffer=io.BytesIO();torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state()},buffer)
        expected_loss=step();expected={k:v.detach().clone() for k,v in m.state_dict().items()}
        buffer.seek(0);state=torch.load(buffer,map_location='cpu',weights_only=False);m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
        actual_loss=step();self.assertTrue(torch.equal(expected_loss,actual_loss))
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in expected.items()))

    def test_05b_preflight_and_protocol_static(self):
        import yaml
        ast.parse((ROOT/'src/preflight_gpu.py').read_text())
        cfg=[yaml.safe_load(p.read_text()) for p in sorted((ROOT/'config').glob('*.yaml'))]
        self.assertEqual(len(cfg),3)
        for c in cfg:
            self.assertEqual(c['effective_batch'],64);self.assertIsNone(c['micro_batch']);self.assertEqual(c['input_size'],256)
        self.assertEqual({c['head_channels'] for c in cfg},{64,512})
        self.assertEqual(sum(c['auxiliary_head'] for c in cfg),1)
        self.assertTrue(all(c['aux_loss_weight']==.4 for c in cfg))

    def test_06_metrics_and_split_hashes(self):
        v=aggregate_counts([2,0],[1,0],[1,2],[4,3]);self.assertAlmostEqual(v['macro']['dice'],1/3);self.assertAlmostEqual(v['pooled']['dice'],.5)
        partition=json.loads((PROJECT/'stage45_upernet_decoder/identity_partition.json').read_text())
        manifest=json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
        test_manifest=json.loads(Path('/home/waas/mambavision_stage22/local_audit_records/test_assets/test_manifest.json').read_text())
        ids=partition['splits'];self.assertEqual([len(ids[k]) for k in ('train','val','test')],[1886,762,867])
        for k in ids:self.assertEqual(len(ids[k]),len(set(ids[k])))
        lookup={(x['split'],x['id']):x for x in manifest}; test_lookup={x['id']:x for x in test_manifest}
        self.assertEqual(len(lookup),len(manifest));self.assertEqual(len(test_lookup),len(test_manifest))
        image_hashes={k:{lookup[k,s]['decoded_image_sha256'] for s in ids[k]} for k in ('train','val')}
        test_dir=Path('/home/waas/mambavision_stage22/heldout_data/test')
        def check_dev(pair):
            split,sid=pair; rec=lookup[split,sid]; root=PROJECT/'data/isic2018'/split
            image=root/'images'/f'{sid}.png'; mask=root/'masks'/f'{sid}.png'
            assert sha256(image)==rec['image_sha256'], str(image)
            assert sha256(mask)==rec['mask_sha256'], str(mask)
            return 1
        def check_test(sid):
            rec=test_lookup[sid]; image=test_dir/'images'/f'{sid}.png'; mask=test_dir/'masks'/f'{sid}.png'
            assert sha256(image)==rec['image_256_sha256'], str(image)
            assert sha256(mask)==rec['mask_256_sha256'], str(mask)
            with Image.open(image) as im: return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(check_dev,[(k,s) for k in ('train','val') for s in ids[k]])),2648)
            image_hashes['test']=set(pool.map(check_test,ids['test']))
        for a,b in (('train','val'),('train','test'),('val','test')):
            self.assertFalse(image_hashes[a]&image_hashes[b],f'exact resized-image overlap: {a} {b}')
        audit=Path('/home/waas/mambavision_stage22/local_audit_records/test_assets')
        for name in ('overlap_audit.json','development_overlap_audit.json','test_manifest.json'):
            self.assertEqual(sha256(audit/name),partition['audit_sha256'][name])
        self.assertFalse(partition['patient_disjointness_verified'])
        print('SPLITS',partition['counts'],'train/val IDs are split-local; exact resized image hash disjoint',flush=True)

if __name__=='__main__':unittest.main(verbosity=2)
