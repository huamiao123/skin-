"""Non-mutating resume and capacity check on the replacement GPU.

Load the real checkpoint into a separate process, keep the real validation
cache resident, and execute one training update on the next epoch's first real
batch. No model, optimizer or RNG state from this dry run is written back.
"""
from segman_model import *
from train import encode_val, extract
from stage17_mambavision_decoder_hiram_seed42.models import FrozenEncoder
from stage3_confirmation.data import ImageData
from stage3_confirmation.models import segmentation_loss, tensor_hash


def main():
    setup()
    local=Path(__file__).resolve().parent
    out=local/'results'
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    state=torch.load(out/'latest.pth',map_location='cpu')
    cfg=state['config']
    assert cfg['seed']==42 and cfg['epochs']==300 and cfg['train_batch']==64
    assert cfg['gpu']==torch.cuda.get_device_name()
    assert cfg['precision']=='BF16 autocast with FP32 segmentation loss'
    assert state['updates']==state['epoch']*30 and state['epoch']<300
    torch.cuda.reset_peak_memory_stats()
    data=ImageData(42)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    encoder=FrozenEncoder().cuda().eval()
    enc_hash=tensor_hash(encoder)
    assert enc_hash==state['encoder_hash']
    model=MambaVisionSegMAN().cuda()
    model.load_state_dict(state['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300,eta_min=1e-5)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    assert tree_hash(model.state_dict())==tree_hash(state['model'])
    assert tree_hash(optimizer.state_dict())==tree_hash(state['optimizer'])
    assert tree_hash(scheduler.state_dict())==tree_hash(state['scheduler'])
    print('ENCODING REAL VALIDATION SET',flush=True)
    val_features,val_targets=encode_val(encoder,data,64)
    restore_rng(state['rng'])
    assert tree_hash(rng_state())==tree_hash(state['rng'])
    next_epoch=state['epoch']+1
    order,matrices=data.geometry(next_epoch)
    images,target=data.batch('train',order[:64],matrices)
    assert len(images)==64
    model.train()
    features=extract(encoder,images,64)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(features)
    loss=segmentation_loss(logits,target)
    assert list(logits.shape)==[64,1,256,256] and torch.isfinite(loss)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(norm)
    optimizer.step()
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        validation_logits=model([x[:64] for x in val_features])
    assert list(validation_logits.shape)==[64,1,256,256] and torch.isfinite(validation_logits).all()
    torch.cuda.synchronize()
    assert tensor_hash(encoder)==enc_hash and all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    free,total=torch.cuda.mem_get_info()
    result=dict(passed=True,gpu=torch.cuda.get_device_name(),total_gib=total/1024**3,
        checkpoint_epoch=state['epoch'],resume_epoch=next_epoch,optimizer_updates=state['updates'],
        batch=64,precision=cfg['precision'],loss=float(loss),gradient_norm=float(norm),
        checkpoint_models_optimizer_scheduler_rng_restored=True,encoder_frozen_and_unchanged=True,
        real_train_batch_forward_backward_optimizer_update=True,real_validation_cache_and_forward=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,free_after_test_gib=free/1024**3,
        note='Separate dry-run process; no checkpoint or experiment state was modified')
    dump(local/'resume_preflight_24g.json',result)
    print('RESUME_PREFLIGHT_PASSED',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
