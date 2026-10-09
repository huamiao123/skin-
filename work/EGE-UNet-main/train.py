import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.egeunet import EGEUNet, BGCTEGEUNet, EGEHRViTUNet, EGEWaveUNet, EGEPRHUNet, EGEWavePSUDRUNet

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def log_wave_stats(config, model, epoch, logger, csv_path):
    if not hasattr(model, 'wave_adapter'):
        return
    adapter = model.wave_adapter
    if getattr(adapter, 'fusion_type', None) != 'scalar':
        return
    alpha = torch.sigmoid(adapter.alpha_logit).detach().item()
    import csv as _csv
    import os as _os
    header = ['epoch', 'alpha']
    row = [epoch, f'{alpha:.6f}']
    new_file = not _os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        w = _csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)
    log_info = f'wave mode={getattr(adapter, "mode", "full")}, epoch {epoch}, alpha: {alpha:.4f}'
    print(log_info)
    logger.info(log_info)


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                worker_init_fn=seed_worker,
                                generator=make_data_generator(config.seed),
                                persistent_workers=False)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_batch_size = int(getattr(config, 'val_batch_size', 1))
    if val_batch_size < 1:
        raise ValueError('val_batch_size must be positive')
    print(f'validation batch size: {val_batch_size}; final evaluation batch size: 1')
    val_loader = DataLoader(val_dataset,
                                batch_size=val_batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=False,
                                worker_init_fn=seed_worker,
                                persistent_workers=False)





    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'egeunet':
        model = EGEUNet(num_classes=model_cfg['num_classes'],
                        input_channels=model_cfg['input_channels'],
                        c_list=model_cfg['c_list'],
                        bridge=model_cfg['bridge'],
                        gt_ds=model_cfg['gt_ds'],
                        )
    elif config.network == 'bgct_egeunet':
        model = BGCTEGEUNet(num_classes=model_cfg['num_classes'],
                        input_channels=model_cfg['input_channels'],
                        c_list=model_cfg['c_list'],
                        bridge=model_cfg['bridge'],
                        gt_ds=model_cfg['gt_ds'],
                        window_size=model_cfg.get('window_size', 8),
                        num_heads=model_cfg.get('num_heads', 4),
                        )
    elif config.network == 'ege_hrvit_unet':
        schedule = config.halting_schedule
        model = EGEHRViTUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'],
            bridge=model_cfg['bridge'],
            gt_ds=model_cfg['gt_ds'],
            adapter_dim=48,
            adapter_heads=4,
            adapter_window_size=8,
            adapter_total_depth=12,
            adapter_halt_after=3,
            adapter_mlp_ratio=2.0,
        )
        model.halting_schedule = schedule
    elif config.network == 'ege_hrvit_nohalting':
        model = EGEHRViTUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'],
            bridge=model_cfg['bridge'],
            gt_ds=model_cfg['gt_ds'],
            adapter_dim=48,
            adapter_heads=4,
            adapter_window_size=8,
            adapter_total_depth=12,
            adapter_halt_after=3,
            adapter_mlp_ratio=2.0,
            no_halting=True,
        )
    elif config.network == 'ege_prh':
        model = EGEPRHUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'],
            bridge=model_cfg['bridge'],
            gt_ds=model_cfg['gt_ds'],
            adapter_dim=48,
            adapter_heads=4,
            adapter_window_size=8,
            adapter_total_depth=12,
            adapter_shallow_blocks=3,
            adapter_final_keep_ratio=0.5,
            adapter_mlp_ratio=2.0,
        )
    elif config.network == 'ege_wave_psudr':
        model = EGEWavePSUDRUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'],
            bridge=model_cfg['bridge'],
            gt_ds=model_cfg['gt_ds'],
        )
    elif config.network == 'ege_wave_unet':
        model = EGEWaveUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'],
            bridge=model_cfg['bridge'],
            gt_ds=model_cfg['gt_ds'],
            wave_mode=getattr(config, 'wave_mode', 'full'),
            fusion_type=getattr(config, 'fusion_type', 'spatial_gate'),
        )
    elif config.network == 'ege_dual':
        from models.ege_dual import EGEDualUNet
        model = EGEDualUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'],
            bridge=model_cfg['bridge'],
            gt_ds=model_cfg['gt_ds'],
            t_embed=getattr(config, 't_embed', 48),
            t_depths=getattr(config, 't_depths', (2, 2, 2, 2)),
            t_head_dim=getattr(config, 't_head_dim', 16),
            t_sr_ratios=getattr(config, 't_sr_ratios', (4, 2, 1, 1)),
            t_mlp_ratio=getattr(config, 't_mlp_ratio', 4.0),
            t_drop_path_rate=getattr(config, 't_drop_path_rate', 0.1),
            fusion_type=getattr(config, 'fusion_type', 'scalar'),
        )
    else: raise Exception('network in not right!')
    model = model.cuda()





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    if getattr(config, 'diff_lr', False) and config.network == 'ege_dual':
        t_params = list(model.t_encoder.parameters())
        t_ids = {id(p) for p in t_params}
        other_params = [p for p in model.parameters() if id(p) not in t_ids]
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': config.lr},
            {'params': t_params, 'lr': config.t_lr},
        ], betas=config.betas, eps=config.eps,
            weight_decay=config.weight_decay, amsgrad=config.amsgrad)
        print(f'[diff-lr] CNN+fusion: {sum(p.numel() for p in other_params):,} params '
              f'lr={config.lr}; transformer branch: {sum(p.numel() for p in t_params):,} '
              f'params lr={config.t_lr}')
        logger.info(f'[diff-lr] group1 {sum(p.numel() for p in other_params):,} '
                    f'lr={config.lr}; group2 {sum(p.numel() for p in t_params):,} '
                    f'lr={config.t_lr}')
    else:
        optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=getattr(config, 'amp', False))





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    resume_micro_step = 0





    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        restore_rng_state(checkpoint.get('rng_state'))
        resume_micro_step = int(checkpoint.get('micro_step', 0))
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    step = resume_micro_step
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            scaler=scaler,
        )

        loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )

        if config.network == 'ege_wave_unet':
            log_wave_stats(config, model, epoch, logger,
                           os.path.join(config.work_dir, 'alpha.csv'))

        if config.network == 'ege_dual' and hasattr(model, '_gamma_stats') and model._gamma_stats:
            import csv as _csv
            validation_gamma = getattr(model, '_last_validation_gamma', None)
            if validation_gamma is not None:
                means = validation_gamma.float().mean(dim=0).tolist()
                stats = {k: means[i] for i, k in enumerate(('g2', 'g3', 'g4', 'g5'))}
            else:
                stats = {k: v.detach().float().mean().item() for k, v in model._gamma_stats.items()}
            csv_path = os.path.join(config.work_dir, 'gamma.csv')
            new_file = not os.path.exists(csv_path)
            with open(csv_path, 'a') as f:
                w = _csv.writer(f)
                if new_file:
                    w.writerow(['epoch'] + list(stats.keys()))
                w.writerow([epoch] + [f'{v:.6f}' for v in stats.values()])
            log_info = ('train gamma: ' + ' '.join(f'{k}={v:.3f}' for k, v in stats.items()))
            print(log_info)
            logger.info(log_info)

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'rng_state': capture_rng_state(),
                'micro_step': step,
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        # Keep the final per-image evaluation and image export on batch size 1.
        final_loader = val_loader if val_batch_size == 1 else DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=False,
            worker_init_fn=seed_worker,
            persistent_workers=False,
        )
        loss = test_one_epoch(
                final_loader,
                model,
                criterion,
                logger,
                config,
                split_name='V_dev',
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(config)
