"""Stage34: prove intervention scope and original-author initialization before training."""
from pathlib import Path
import ast,copy,gc,json,hashlib
import torch
from torch import nn
from train import ROOT,PROJECT,System,base,dump,digest,setup
from author_modules import _import,OLD,NEW
def main():
    setup()
    torch.set_num_threads(4)
    manifest=json.loads((ROOT/'ARCHIVED_SOURCE_HASHES.json').read_text())
    assert all(digest(Path(p))==sha for p,sha in manifest.items())
    assert (ROOT/'identity_partition.json').read_bytes()==(PROJECT/'stage33_author_modules/identity_partition.json').read_bytes()
    original=_import(PROJECT/'stage33_author_modules/author_modules.py','stage33_original_wiring_for_audit')
    author_cga=_import(NEW/'maklachur__MambaLiteUNet/models/MambaLiteUNet.py','stage34_native_cga_policy')
    author_emcad=_import(OLD/'SLDGroup__EMCAD/lib/decoders.py','stage34_native_emcad_policy')
    reports={}
    mapping={'swin_cga_author_init':'swin_cga','emcad64_exp2':'emcad64'}
    for arm,previous in mapping.items():
        reference=base.System(finetune=True,stochastic_depth=False)
        old_decoder=original.AuthorModuleDecoder(previous,reference.probe.decoder,seed=42)
        model=System(arm)
        base.assert_tree_exact(reference.encoder.state_dict(),model.encoder.state_dict())
        base.assert_tree_exact(reference.probe.adapter.state_dict(),model.probe.adapter.state_dict())
        optimizer=base.optimizer(model)
        optimized=[p for g in optimizer.param_groups for p in g['params']]
        assert len(optimized)==len({id(p) for p in optimized})
        assert {id(p) for p in optimized}=={id(p) for p in model.parameters() if p.requires_grad}
        gate_prefix='probe.decoder.author_gates.'
        if previous=='swin_cga':
            base.assert_tree_exact(old_decoder.baseline.state_dict(),model.probe.decoder.baseline.state_dict())
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(42)
                expected=nn.ModuleDict({str(i):author_cga.CrossGatedAttentionGate(64,64,32) for i in (2,1,0)})
                expected.apply(lambda m:author_cga.MambaLiteUNet._init_weights(None,m))
            base.assert_tree_exact(expected.state_dict(),model.probe.decoder.author_gates.state_dict())
            changed=[k for k,v in old_decoder.state_dict().items() if not torch.equal(v,model.probe.decoder.state_dict()[k])]
            assert changed and all(k.startswith('author_gates.') for k in changed)
            assert sum(p.numel() for p in old_decoder.parameters())==sum(p.numel() for p in model.probe.decoder.parameters())
            item=dict(native_root_initialization_exact=True,changed_decoder_tensors=len(changed),
                      changes_restricted_to_fresh_gates=True,parameter_count_unchanged=True)
        else:
            base.assert_tree_exact(old_decoder.readout.state_dict(),model.probe.decoder.readout.state_dict())
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(42)
                expected=author_emcad.EMCAD(channels=[64]*4,kernel_sizes=[1,3,5],expansion_factor=2,
                   dw_parallel=True,add=True,lgag_ks=3,activation='relu6')
            base.assert_tree_exact(expected.state_dict(),model.probe.decoder.author_decoder.state_dict())
            old_n=sum(p.numel() for p in old_decoder.author_decoder.parameters())
            new_n=sum(p.numel() for p in model.probe.decoder.author_decoder.parameters())
            assert old_n==284043 and new_n==108939
            item=dict(native_binary_config_exact=True,expansion_factor=2,activation='relu6',
                      old_decoder_parameters=old_n,new_decoder_parameters=new_n)
        rng_before=torch.get_rng_state().clone()
        recreated=type(model.probe.decoder)(arm,reference.probe.decoder,seed=42)
        assert torch.equal(rng_before,torch.get_rng_state())
        base.assert_tree_exact(model.probe.decoder.state_dict(),recreated.state_dict())
        item.update(encoder_exact=True,adapters_exact=True,shared_swin_or_readout_exact=True,
                    all_trainable_parameters_in_optimizer=True,outer_torch_rng_unchanged=True,
                    frozen_encoder_early_trainable_late=all(not p.requires_grad for p in model.encoder.model.levels[0].parameters())
                        and all(p.requires_grad for p in model.encoder.model.levels[2].parameters()),
                    original_author_sha256=digest(NEW/'maklachur__MambaLiteUNet/models/MambaLiteUNet.py') if previous=='swin_cga'
                        else digest(OLD/'SLDGroup__EMCAD/lib/decoders.py'))
        assert item['frozen_encoder_early_trainable_late']
        reports[arm]=item
        print('INTEGRATION_OK',arm,json.dumps(item),flush=True)
        del reference,old_decoder,model,optimizer,optimized,expected,recreated
        gc.collect()
    result=dict(passed=True,checks=reports,original_files_unchanged=True,test_read=False,
                scope='CPU construction and parameter ownership; real-data forward/backward and resume checked separately')
    dump(ROOT/'INTEGRATION_VERIFIED.json',result)
if __name__=='__main__':main()
