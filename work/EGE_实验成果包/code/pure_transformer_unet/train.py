import os
import sys
import random
import argparse
import importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from datetime import datetime

from models.pure_transformer_unet import PureTransformerUNet

EGE_ROOT = '/root/EGE-UNet-main'


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ege_dataset = _load_module('ege_dataset', os.path.join(EGE_ROOT, 'datasets', 'dataset.py'))
ege_utils = _load_module('ege_utils', os.path.join(EGE_ROOT, 'utils.py'))

NPY_datasets = ege_dataset.NPY_datasets
myNormalize = ege_utils.myNormalize
myToTensor = ege_utils.myToTensor
myRandomHorizontalFlip = ege_utils.myRandomHorizontalFlip
myRandomVerticalFlip = ege_utils.myRandomVerticalFlip
myRandomRotation = ege_utils.myRandomRotation
myResize = ege_utils.myResize
BceDiceLoss = ege_utils.BceDiceLoss
get_logger = ege_utils.get_logger

import warnings
warnings.filterwarnings("ignore")


class PTUConfig:
    data_path = '/root/EGE-UNet-main/data/isic2018/'
    input_size_h = 256
    input_size_w = 256
    batch_size = 64
    accum_steps = 1
    epochs = 300
    lr = 1e-3
    weight_decay = 1e-2
    seed = 42
    num_workers = 8
    threshold = 0.5
    print_interval = 20
    val_interval = 30
    save_interval = 100
    drop_path_rate = 0.1
    embed_dims = (64, 128, 256, 512)
    num_heads = (2, 4, 8, 16)
    sr_ratios = (4, 2, 1, 1)
    decoder_depths = (2, 2, 2)
    decoder_heads = (8, 4, 2)
    decoder_sr_ratios = (1, 2, 4)
    work_dir = 'results/ptu_isic18_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
    train_transformer = transforms.Compose([
        myNormalize('isic18', train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])
    test_transformer = transforms.Compose([
        myNormalize('isic18', train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(config, depths, base_embed=64, decoder_depths=None):
    embed_dims = tuple(base_embed * (2 ** i) for i in range(4))
    num_heads = tuple(d // 32 for d in embed_dims)
    if decoder_depths is None:
        decoder_depths = config.decoder_depths
    decoder_heads = tuple(embed_dims[2 - j] // 32 for j in range(3))
    return PureTransformerUNet(
        in_chans=3, num_classes=1, patch_size=4,
        embed_dims=embed_dims, depths=depths,
        num_heads=num_heads, sr_ratios=config.sr_ratios,
        decoder_depths=decoder_depths,
        decoder_heads=decoder_heads,
        decoder_sr_ratios=config.decoder_sr_ratios,
        mlp_ratio=4.0, head_dim=32,
        drop=0.0, attn_drop=0.0, drop_path_rate=config.drop_path_rate,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth3', type=int, default=4, choices=[2, 4, 6, 8, 12])
    parser.add_argument('--ddec3', type=int, default=2, choices=[2, 4, 6, 12])
    parser.add_argument('--embed', type=int, default=64, choices=[48, 64, 96])
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--dual', action='store_true',
                        help='PTU-Dual: transformer main + parallel CNN aux branch')
    parser.add_argument('--t_lr', type=float, default=None,
                        help='lr for transformer main path in dual mode')
    parser.add_argument('--cnn_lr', type=float, default=1e-3,
                        help='lr for CNN aux branch in dual mode')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='explicit work_dir (used for resume)')
    parser.add_argument('--resume', action='store_true',
                        help='resume from latest.pth in work_dir')
    args = parser.parse_args()

    config = PTUConfig
    if args.lr is not None:
        config.lr = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    depths = [2, 2, args.depth3, 2]
    decoder_depths = [args.ddec3, 2, 2]
    if args.embed == 96 and config.batch_size == 64:
        config.batch_size = 32
        config.accum_steps = 2
    if args.dual and args.t_lr is None:
        args.t_lr = 1e-4
    tag = args.tag or (f'dual_d{args.depth3}_tlr{args.t_lr:.0e}_clr{args.cnn_lr:.0e}'
                       if args.dual
                       else f'e{args.embed}_d{args.depth3}_dd{args.ddec3}_lr{config.lr:.0e}')
    if args.work_dir is not None:
        config.work_dir = args.work_dir
    else:
        config.work_dir = f'results/ptu_isic18_{tag}_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    os.makedirs(config.work_dir, exist_ok=True)
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = get_logger('train', log_dir)

    set_seed(config.seed)
    torch.cuda.empty_cache()

    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=config.num_workers,
                            drop_last=True)

    if args.dual:
        from models.ptu_dual import PTUDualUNet
        embed_dims = tuple(config.embed_dims)
        num_heads = tuple(d // 32 for d in embed_dims)
        decoder_heads = tuple(embed_dims[2 - j] // 32 for j in range(3))
        model = PTUDualUNet(
            in_chans=3, num_classes=1, patch_size=4,
            embed_dims=embed_dims, depths=depths, num_heads=num_heads,
            sr_ratios=config.sr_ratios,
            decoder_depths=decoder_depths, decoder_heads=decoder_heads,
            decoder_sr_ratios=config.decoder_sr_ratios,
            mlp_ratio=4.0, head_dim=32,
            drop=0.0, attn_drop=0.0, drop_path_rate=config.drop_path_rate,
            cnn_channels=(32, 64, 128, 256),
        ).cuda()
    else:
        model = build_model(config, depths, base_embed=args.embed,
                            decoder_depths=decoder_depths).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'PTU ISIC18 | dual={args.dual} | depths={depths} | '
                f'params: {n_params:,} | batch {config.batch_size} | '
                f'Cosine {config.epochs}ep | seed {config.seed}')

    criterion = BceDiceLoss(wb=1, wd=1)
    if args.dual:
        cnn_ids = set()
        cnn_ids.update(id(p) for p in model.cnn_branch.parameters())
        cnn_ids.update(id(p) for p in model.fuse.parameters())
        cnn_params = [p for p in model.parameters() if id(p) in cnn_ids]
        t_params = [p for p in model.parameters() if id(p) not in cnn_ids]
        optimizer = torch.optim.AdamW([
            {'params': t_params, 'lr': args.t_lr},
            {'params': cnn_params, 'lr': args.cnn_lr},
        ], weight_decay=config.weight_decay)
        print(f'[diff-lr] transformer main: {sum(p.numel() for p in t_params):,} '
              f'lr={args.t_lr}; CNN aux: {sum(p.numel() for p in cnn_params):,} '
              f'lr={args.cnn_lr}')
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                      weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5)

    min_loss = 999
    min_epoch = 1
    start_epoch = 1

    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    if args.resume and os.path.exists(resume_model):
        ckpt = torch.load(resume_model, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        saved_epoch = ckpt['epoch']
        start_epoch = saved_epoch + 1
        min_loss, min_epoch = ckpt['min_loss'], ckpt['min_epoch']
        log_info = (f'resumed from {resume_model}: epoch {saved_epoch}, '
                    f'min_loss {min_loss:.4f}@ep{min_epoch}, continue from {start_epoch}')
        print(log_info)
        logger.info(log_info)

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        loss_list = []
        optimizer.zero_grad()
        for iter, (images, targets) in enumerate(train_loader):
            images = images.cuda(non_blocking=True).float()
            targets = targets.cuda(non_blocking=True).float()
            out = torch.sigmoid(model(images))
            loss = criterion(out, targets) / config.accum_steps
            loss.backward()
            if (iter + 1) % config.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            if torch.isnan(loss).any():
                logger.info(f'NaN loss at epoch {epoch} iter {iter}, aborting')
                print('NaN LOSS, ABORT')
                return
            loss_list.append(loss.item() * config.accum_steps)
            if iter % config.print_interval == 0:
                lr_now = optimizer.state_dict()['param_groups'][0]['lr']
                log_info = (f'train: epoch {epoch}, iter:{iter}, '
                            f'loss: {np.mean(loss_list):.4f}, lr: {lr_now}')
                print(log_info)
                logger.info(log_info)
        scheduler.step()

        model.eval()
        val_loss_list = []
        preds = []
        gts = []
        with torch.no_grad():
            for img, msk in val_loader:
                img = img.cuda(non_blocking=True).float()
                msk = msk.cuda(non_blocking=True).float()
                out = torch.sigmoid(model(img))
                val_loss_list.append(criterion(out, msk).item())
                gts.append(msk.squeeze(1).cpu().detach().numpy())
                preds.append(out.squeeze(1).cpu().detach().numpy())

        loss = np.mean(val_loss_list)
        if args.dual and model._gamma_stats:
            import csv as _csv
            stats = {k: v.detach().item() for k, v in model._gamma_stats.items()}
            csv_path = os.path.join(config.work_dir, 'gamma.csv')
            new_file = not os.path.exists(csv_path)
            with open(csv_path, 'a') as f:
                w = _csv.writer(f)
                if new_file:
                    w.writerow(['epoch'] + list(stats.keys()))
                w.writerow([epoch] + [f'{v:.6f}' for v in stats.values()])
            print('train gamma: ' + ' '.join(f'{k}={v:.3f}' for k, v in stats.items()))
        if epoch % config.val_interval == 0:
            preds = np.array(preds).reshape(-1)
            gts = np.array(gts).reshape(-1)
            y_pre = np.where(preds >= config.threshold, 1, 0)
            y_true = np.where(gts >= 0.5, 1, 0)
            confusion = confusion_matrix(y_true, y_pre)
            TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
            accuracy = float(TN + TP) / float(np.sum(confusion))
            sensitivity = float(TP) / float(TP + FN) if (TP + FN) != 0 else 0
            specificity = float(TN) / float(TN + FP) if (TN + FP) != 0 else 0
            f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
            miou = float(TP) / float(TP + FP + FN) if (TP + FP + FN) != 0 else 0
            log_info = (f'val epoch: {epoch}, loss: {loss:.4f}, miou: {miou}, '
                        f'f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, '
                        f'specificity: {specificity}, sensitivity: {sensitivity}')
            print(log_info)
            logger.info(log_info)
        else:
            log_info = f'val epoch: {epoch}, loss: {loss:.4f}'
            print(log_info)
            logger.info(log_info)

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

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        model.load_state_dict(torch.load(
            os.path.join(checkpoint_dir, 'best.pth'), map_location='cpu'))
        model.eval()
        preds = []
        gts = []
        with torch.no_grad():
            for img, msk in val_loader:
                img = img.cuda(non_blocking=True).float()
                msk = msk.cuda(non_blocking=True).float()
                out = torch.sigmoid(model(img))
                gts.append(msk.squeeze(1).cpu().detach().numpy())
                preds.append(out.squeeze(1).cpu().detach().numpy())
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)
        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
        accuracy = float(TN + TP) / float(np.sum(confusion))
        sensitivity = float(TP) / float(TP + FN) if (TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if (TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if (TP + FP + FN) != 0 else 0
        log_info = (f'test of best model, best_val_loss: {min_loss:.4f}, '
                    f'miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, '
                    f'specificity: {specificity}, sensitivity: {sensitivity}, '
                    f'confusion_matrix: {confusion}')
        print(log_info)
        logger.info(log_info)
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )
        print('=== PTU TRAINING COMPLETE ===')


if __name__ == '__main__':
    main()