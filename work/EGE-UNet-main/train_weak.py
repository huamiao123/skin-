import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from models.ege_wave_weak import EGEWaveWeakUNet
from datasets.isic2018_weak import ISIC2018WeakDataset
from datasets.dataset import NPY_datasets
from losses.weak_losses import WeakLosses
from losses.total_loss import compute_total_loss
from engine import val_one_epoch, test_one_epoch, get_original_seg_loss_fn
from utils import set_seed, get_logger, log_config_info, get_optimizer, get_scheduler

import warnings
warnings.filterwarnings("ignore")


def train_one_epoch_weak(train_loader, model, optimizer, epoch, step, logger, config, writer):
    model.train()
    loss_list = []
    loss_partial_list = []
    loss_box_lf_list = []
    loss_point_local_list = []
    loss_cons_list = []

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    for iter, batch in enumerate(train_loader):
        step += iter

        optimizer.zero_grad(set_to_none=True)

        if config.weak_mode == 'baseline':
            image = batch["image"].cuda().float()
            pos = batch["pos_mask"].cuda().bool()
            neg = batch["neg_mask"].cuda().bool()

            with torch.cuda.amp.autocast(enabled=False):
                _, final_prob = model(image, return_aux=False)
                final_logits = torch.logit(final_prob.clamp(1e-7, 1 - 1e-7))
                loss_partial = WeakLosses.balanced_partial_bce(final_logits, pos, neg)
                loss = config.lambda_partial * loss_partial

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_list.append(loss.item())
            loss_partial_list.append(loss_partial.item())

            if iter % config.print_interval == 0:
                print(f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, '
                      f'partial: {np.mean(loss_partial_list):.4f}')

        elif config.weak_mode == 'full':
            xw = batch["image_weak"].cuda().float()
            xs = batch["image_strong"].cuda().float()
            pos = batch["pos_mask"].cuda().bool()
            neg = batch["neg_mask"].cuda().bool()
            unknown = batch["unknown_mask"].cuda().bool()

            with torch.cuda.amp.autocast(enabled=False):
                out_w = model(xw, return_aux=True)
                out_s = model(xs, return_aux=True)

                loss_partial = WeakLosses.balanced_partial_bce(out_w["final"], pos, neg)

                box_masks = batch["box_mask"].cuda()
                boxes = WeakLosses.box_mask_to_boxes(box_masks)
                loss_box_lf = WeakLosses.projection_loss(out_w["ll"], boxes)

                loss_point_local = WeakLosses.local_positive_loss(out_w["local"], pos)

                loss_cons = WeakLosses.unknown_consistency_loss(
                    out_w["final"], out_s["final"], unknown
                )

                loss = (
                    config.lambda_partial * loss_partial
                    + config.lambda_box_lf * loss_box_lf
                    + config.lambda_local_point * loss_point_local
                    + config.lambda_consistency * loss_cons
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_list.append(loss.item())
            loss_partial_list.append(loss_partial.item())
            loss_box_lf_list.append(loss_box_lf.item())
            loss_point_local_list.append(loss_point_local.item())
            loss_cons_list.append(loss_cons.item())

            if iter % config.print_interval == 0:
                log_str = (f'train: epoch {epoch}, iter:{iter}, total: {np.mean(loss_list):.4f}, '
                           f'partial: {np.mean(loss_partial_list):.4f}, '
                           f'box_lf: {np.mean(loss_box_lf_list):.4f}, '
                           f'point_local: {np.mean(loss_point_local_list):.4f}, '
                           f'cons: {np.mean(loss_cons_list):.4f}')
                print(log_str)

        writer.add_scalar('loss', np.mean(loss_list), global_step=step)

    now_lr = optimizer.state_dict()['param_groups'][0]['lr']
    return step


def val_one_epoch_weak(val_loader, model, criterion, epoch, logger, config):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(val_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            _, final_prob = model(img, return_aux=False)

            final_logits = torch.logit(final_prob.clamp(1e-7, 1 - 1e-7))
            bce = F.binary_cross_entropy_with_logits(final_logits, msk)
            smooth = 1.0
            dice = 1 - (2 * (final_prob * msk).sum() + smooth) / (final_prob.sum() + msk.sum() + smooth)
            loss = bce + dice

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            preds.append(final_prob.squeeze(1).cpu().detach().numpy())

    if epoch % config.val_interval == 0:
        preds_flat = np.concatenate([p.reshape(-1) for p in preds])
        gts_flat = np.concatenate([g.reshape(-1) for g in gts])

        y_pre = np.where(preds_flat >= config.threshold, 1, 0)
        y_true = np.where(gts_flat >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = (f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou:.4f}, '
                    f'dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, '
                    f'specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}')
        print(log_info)
        logger.info(log_info)
    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    for d in [checkpoint_dir, outputs]:
        os.makedirs(d, exist_ok=True)

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
    train_dataset = ISIC2018WeakDataset(
        config.data_path,
        config.weak_json_path,
        config,
        mode=config.weak_mode,
        is_train=True
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    print(f'Weak train samples: {len(train_dataset)}')

    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)
    print(f'Val samples: {len(val_dataset)}')

    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    model = EGEWaveWeakUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        c_list=model_cfg['c_list'],
        bridge=model_cfg['bridge'],
        gt_ds=model_cfg['gt_ds'],
    )
    model = model.cuda()
    print(f'Model: {config.network}, mode: {config.weak_mode}')

    print('#----------Preparing loss, opt, sch----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss, min_epoch = checkpoint['min_loss'], checkpoint['min_epoch']
        log_info = f'resuming model from {resume_model}. epoch: {start_epoch}, min_loss: {min_loss:.4f}'
        logger.info(log_info)

    step = 0
    print(f'#----------Training ({config.weak_mode} mode)----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()

        step = train_one_epoch_weak(
            train_loader, model, optimizer, epoch, step, logger, config, writer
        )

        scheduler.step()

        loss = val_one_epoch_weak(
            val_loader, model, criterion, epoch, logger, config
        )

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
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        from engine import test_one_epoch as orig_test_one_epoch
        orig_test_one_epoch(val_loader, model, criterion, logger, config)
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'full'])
    args = parser.parse_args()

    if args.mode == 'baseline':
        from configs.weak_config import WeakBaseConfig
        config = WeakBaseConfig()
    else:
        from configs.weak_config import WeakFullConfig
        config = WeakFullConfig()

    main(config)
