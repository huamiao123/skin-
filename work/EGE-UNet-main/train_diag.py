import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random as py_random

from models.ege_wave_weak import EGEWaveWeakUNet
from datasets.diag_dataset import DiagDataset
from datasets.dataset import NPY_datasets
from losses.diag_losses import compute_diag_loss
from utils import set_seed, get_logger, log_config_info, get_optimizer, get_scheduler

import warnings
warnings.filterwarnings("ignore")


def compute_diag_metrics(preds_flat, gts_flat, threshold=0.5):
    y_pre = np.where(preds_flat >= threshold, 1, 0)
    y_true = np.where(gts_flat >= 0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    acc = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sens = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    spec = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    pred_fg = y_pre.mean()
    gt_fg = y_true.mean()
    area_ratio = pred_fg / (gt_fg + 1e-6)
    fp_ratio = FP / (FP + TN + 1e-6)
    fn_ratio = FN / (FN + TP + 1e-6)

    return {
        "acc": acc, "sens": sens, "spec": spec, "dsc": dsc, "miou": miou,
        "pred_fg_ratio": pred_fg, "gt_fg_ratio": gt_fg,
        "area_ratio": area_ratio, "fp_ratio": fp_ratio, "fn_ratio": fn_ratio,
        "confusion": confusion,
    }


def train_one_epoch_diag(train_loader, model, optimizer, epoch, step, logger, config, writer):
    model.train()
    total_list = []
    loss_lists = {}

    for iter, batch in enumerate(train_loader):
        step += iter
        optimizer.zero_grad(set_to_none=True)

        batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=False):
            if config.enable_consistency:
                xw = batch_gpu["image_weak"].float()
                xs = batch_gpu["image_strong"].float()
                out_w = model(xw, return_aux=True)
                out_s = model(xs, return_aux=True)
                outputs = {
                    "seg_logits": out_w["final"],
                    "prob_weak": torch.sigmoid(out_w["final"]),
                    "prob_strong": torch.sigmoid(out_s["final"]),
                    "local_logits": out_w.get("local", None),
                    "ll_logits": out_w.get("ll", None),
                }
            else:
                image = batch_gpu["image"].float()
                out = model(image, return_aux=True)
                outputs = {
                    "seg_logits": out["final"],
                    "local_logits": out.get("local", None),
                    "ll_logits": out.get("ll", None),
                }

            losses = compute_diag_loss(outputs, batch_gpu, config)
            loss = losses["total"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_list.append(loss.item())
        for k, v in losses.items():
            if k not in loss_lists:
                loss_lists[k] = []
            loss_lists[k].append(v.item())

        if iter % config.print_interval == 0:
            parts = []
            for k in losses:
                parts.append(f"{k}: {np.mean(loss_lists[k]):.4f}")
            log_str = f'train: epoch {epoch}, iter:{iter}, ' + ', '.join(parts)
            print(log_str)

    return step


def val_one_epoch_diag(val_loader, model, epoch, logger, config):
    model.eval()
    preds, gts, loss_list = [], [], []
    best_dsc = getattr(val_one_epoch_diag, '_best_dsc', 0.0)
    best_metrics = getattr(val_one_epoch_diag, '_best_metrics', None)

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

    preds_flat = np.concatenate([p.reshape(-1) for p in preds])
    gts_flat = np.concatenate([g.reshape(-1) for g in gts])

    if epoch % config.val_interval == 0:
        m = compute_diag_metrics(preds_flat, gts_flat, config.threshold)
        log_info = (f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, '
                    f'miou: {m["miou"]:.4f}, dsc: {m["dsc"]:.4f}, '
                    f'sens: {m["sens"]:.4f}, spec: {m["spec"]:.4f}, '
                    f'area_ratio: {m["area_ratio"]:.3f}, '
                    f'fp_ratio: {m["fp_ratio"]:.4f}, fn_ratio: {m["fn_ratio"]:.4f}')
        print(log_info)
        logger.info(log_info)
        if m["dsc"] > best_dsc:
            val_one_epoch_diag._best_dsc = m["dsc"]
            val_one_epoch_diag._best_metrics = m
    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def main(config):
    val_one_epoch_diag._best_dsc = 0.0
    val_one_epoch_diag._best_metrics = None

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    for d in [checkpoint_dir, os.path.join(config.work_dir, 'outputs')]:
        os.makedirs(d, exist_ok=True)

    logger = get_logger('train', log_dir)
    writer = SummaryWriter(config.work_dir + 'summary')
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = DiagDataset(config.data_path, config.weak_json_path, config, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              pin_memory=True, num_workers=config.num_workers)
    print(f'Diag train samples: {len(train_dataset)}, exp: {config.exp_name}')

    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=config.num_workers, drop_last=True)
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

    print(f'Diag Settings: LocalPoint={config.enable_local_point_aux}, '
          f'Consistency={config.enable_consistency}, BoxLF={config.enable_box_lf}')

    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss, min_epoch = checkpoint['min_loss'], checkpoint['min_epoch']

    step = 0
    print(f'#----------Training ({config.exp_name})----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        step = train_one_epoch_diag(train_loader, model, optimizer, epoch, step, logger, config, writer)
        scheduler.step()
        loss = val_one_epoch_diag(val_loader, model, epoch, logger, config)

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save({
            'epoch': epoch,
            'min_loss': min_loss,
            'min_epoch': min_epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, 'latest.pth'))

    best_m = val_one_epoch_diag._best_metrics
    print(f'\n#========== {config.exp_name} FINAL SUMMARY ==========')
    print(f'Best DSC: {best_m["dsc"]:.4f}, mIoU: {best_m["miou"]:.4f}')
    print(f'Sens: {best_m["sens"]:.4f}, Spec: {best_m["spec"]:.4f}')
    print(f'Area Ratio: {best_m["area_ratio"]:.3f}')
    logger.info(f'{config.exp_name} FINAL: DSC={best_m["dsc"]:.4f} mIoU={best_m["miou"]:.4f} '
                f'Sens={best_m["sens"]:.4f} Spec={best_m["spec"]:.4f} AreaRatio={best_m["area_ratio"]:.3f}')

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        criterion = config.criterion

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for data in tqdm(val_loader, desc='Test'):
                img, msk = data
                img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
                _, final_prob = model(img, return_aux=False)
                preds.append(final_prob.squeeze(1).cpu().detach().numpy())
                gts.append(msk.squeeze(1).cpu().detach().numpy())

        preds_flat = np.concatenate([p.reshape(-1) for p in preds])
        gts_flat = np.concatenate([g.reshape(-1) for g in gts])
        m = compute_diag_metrics(preds_flat, gts_flat, config.threshold)

        print(f'\n#========== {config.exp_name} TEST ==========')
        print(f'DSC: {m["dsc"]:.4f}, mIoU: {m["miou"]:.4f}, Acc: {m["acc"]:.4f}')
        print(f'Sens: {m["sens"]:.4f}, Spec: {m["spec"]:.4f}')
        print(f'Pred FG: {m["pred_fg_ratio"]:.4f}, GT FG: {m["gt_fg_ratio"]:.4f}, Area Ratio: {m["area_ratio"]:.3f}')
        print(f'FP Ratio: {m["fp_ratio"]:.6f}, FN Ratio: {m["fn_ratio"]:.6f}')
        print(f'Confusion: {m["confusion"]}')

        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, choices=['B1', 'B2', 'B3'])
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    from configs.diag_config import DiagConfig
    config = DiagConfig()
    config.batch_size = args.batch_size

    if args.exp == 'B1':
        config.set_exp('B1', local_point=True, consistency=False, box_lf=False)
    elif args.exp == 'B2':
        config.set_exp('B2', local_point=False, consistency=True, box_lf=False)
        config.batch_size = 32
    elif args.exp == 'B3':
        config.set_exp('B3', local_point=False, consistency=False, box_lf=True)

    main(config)
