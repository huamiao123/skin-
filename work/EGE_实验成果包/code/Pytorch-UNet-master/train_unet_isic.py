import os
import sys
import random
import importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from datetime import datetime

from unet.unet_model import UNet

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


class UNetConfig:
    data_path = '/root/EGE-UNet-main/data/isic2018/'
    input_size_h = 256
    input_size_w = 256
    batch_size = 32
    accum_steps = 2
    epochs = 300
    lr = 1e-3
    weight_decay = 1e-2
    seed = 42
    num_workers = 8
    threshold = 0.5
    print_interval = 20
    val_interval = 30
    save_interval = 100
    work_dir = 'results/unet_isic18_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
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


def main():
    config = UNetConfig
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

    model = UNet(n_channels=3, n_classes=1, bilinear=False).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'UNet ISIC18 | params: {n_params:,} | input 256x256 | '
                f'batch {config.batch_size} x{config.accum_steps} | AdamW lr={config.lr} wd={config.weight_decay} '
                f'| Cosine 300ep | seed {config.seed}')

    criterion = BceDiceLoss(wb=1, wd=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5)

    min_loss = 999
    min_epoch = 1

    for epoch in range(1, config.epochs + 1):
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
        print('=== UNET TRAINING COMPLETE ===')


if __name__ == '__main__':
    main()