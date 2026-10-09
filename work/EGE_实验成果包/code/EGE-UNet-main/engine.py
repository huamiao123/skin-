import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
from losses.total_loss import compute_total_loss


_original_seg_loss_fn = None
_hrvit_loss_fn = None
_prh_loss_fn = None


def get_prh_loss_fn(config):
    global _prh_loss_fn
    if _prh_loss_fn is None:
        from losses.prh_loss import PRHLoss
        _prh_loss_fn = PRHLoss(
            edge_pos_weight=config.edge_pos_weight,
        )
    return _prh_loss_fn


def get_original_seg_loss_fn(config):
    global _original_seg_loss_fn
    if _original_seg_loss_fn is None:
        _original_seg_loss_fn = config.criterion
    return _original_seg_loss_fn


def get_hrvit_loss_fn(config):
    global _hrvit_loss_fn
    if _hrvit_loss_fn is None:
        from losses.router_loss import EGEHRViTLoss
        from utils import BceDiceLoss
        _hrvit_loss_fn = EGEHRViTLoss(
            lambda_router=config.lambda_router,
            lambda_aux=config.lambda_aux,
            original_seg_loss_fn=BceDiceLoss(wb=1, wd=1),
            token_size=config.token_size,
        )
    return _hrvit_loss_fn


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer):
    model.train()

    if hasattr(model, 'set_epoch'):
        model.set_epoch(epoch)

    loss_list = []
    seg_loss_fn = get_original_seg_loss_fn(config)

    is_hrvit = (config.network == 'ege_hrvit_unet')
    is_prh = (config.network == 'ege_prh')
    is_psudr = (config.network == 'ege_wave_psudr')

    use_amp = getattr(config, 'amp', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum_steps = getattr(config, 'gradient_accumulation_steps', 1)

    for iter, data in enumerate(train_loader):
        step += iter
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        with torch.cuda.amp.autocast(enabled=use_amp):
            model_output = model(images)

            if is_hrvit:
                loss_fn = get_hrvit_loss_fn(config)
                loss, loss_dict = loss_fn(model_output, targets, epoch=epoch)
            elif is_prh:
                loss_fn = get_prh_loss_fn(config)
                loss, loss_dict = loss_fn(model_output, targets)
            elif is_psudr:
                loss = criterion(model_output["deep_supervision"], model_output["final_output"], targets)
                loss_dict = None
            elif isinstance(model_output, dict):
                loss, loss_dict = compute_total_loss(
                    model_output=model_output,
                    target=targets,
                    epoch=epoch,
                    original_seg_loss_fn=seg_loss_fn,
                    lambda_edge=config.lambda_edge,
                    warmup_epochs=config.edge_warmup_epochs
                )
            else:
                gt_pre, out = model_output
                loss = criterion(gt_pre, out, targets)

            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (iter + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_list.append(loss.item() * accum_steps)

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss.item() * accum_steps, global_step=step)

        if iter % config.print_interval == 0:
            if is_prh:
                log_info = (f'train: epoch {epoch}, iter:{iter}, total: {loss_dict["total_loss"]:.4f}, '
                            f'seg: {loss_dict["seg_loss"]:.4f}, edge: {loss_dict["edge_loss"]:.4f}, '
                            f'rho: {loss_dict["retention_ratio"]:.2f}, gamma: {loss_dict["gamma"]:.3f}, '
                            f'edge_pos_p: {loss_dict["edge_pos_prob"]:.3f}, edge_neg_p: {loss_dict["edge_neg_prob"]:.3f}, '
                            f'lr: {now_lr}')
            elif is_psudr:
                s = model_output["psudr_stats"]
                log_info = (f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, '
                            f'b2: {s["beta_gab2"].item():.3f}, b1: {s["beta_gab1"].item():.3f}, '
                            f'w2: [{s["scale_w_gab2"][0].item():.2f},{s["scale_w_gab2"][1].item():.2f},{s["scale_w_gab2"][2].item():.2f}], '
                            f'w1: [{s["scale_w_gab1"][0].item():.2f},{s["scale_w_gab1"][1].item():.2f},{s["scale_w_gab1"][2].item():.2f}], '
                            f'unc2: {s["uncert_gab2"].item():.3f}, unc1: {s["uncert_gab1"].item():.3f}, lr: {now_lr}')
            else:
                log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion,
                    epoch,
                    logger,
                    config):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    seg_loss_fn = get_original_seg_loss_fn(config)
    is_hrvit = (config.network == 'ege_hrvit_unet')
    is_prh = (config.network == 'ege_prh')
    is_psudr = (config.network == 'ege_wave_psudr')
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            model_output = model(img)

            if is_hrvit:
                loss_fn = get_hrvit_loss_fn(config)
                loss, _ = loss_fn(model_output, msk, epoch=epoch)
                out = model_output["final_output"]
            elif is_prh:
                loss_fn = get_prh_loss_fn(config)
                loss, _ = loss_fn(model_output, msk)
                out = model_output["final_output"]
            elif is_psudr:
                loss = criterion(model_output["deep_supervision"], model_output["final_output"], msk)
                out = model_output["final_output"]
            elif isinstance(model_output, dict):
                loss, _ = compute_total_loss(
                    model_output=model_output,
                    target=msk,
                    epoch=epoch,
                    original_seg_loss_fn=seg_loss_fn,
                    lambda_edge=config.lambda_edge,
                    warmup_epochs=config.edge_warmup_epochs
                )
                out = model_output["final_output"]
            else:
                gt_pre, out = model_output
                loss = criterion(gt_pre, out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    seg_loss_fn = get_original_seg_loss_fn(config)
    is_hrvit = (config.network == 'ege_hrvit_unet')
    is_prh = (config.network == 'ege_prh')
    is_psudr = (config.network == 'ege_wave_psudr')
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            model_output = model(img)

            if is_hrvit:
                loss_fn = get_hrvit_loss_fn(config)
                loss, _ = loss_fn(model_output, msk, epoch=0)
                out = model_output["final_output"]
            elif is_prh:
                loss_fn = get_prh_loss_fn(config)
                loss, _ = loss_fn(model_output, msk)
                out = model_output["final_output"]
            elif is_psudr:
                loss = criterion(model_output["deep_supervision"], model_output["final_output"], msk)
                out = model_output["final_output"]
            elif isinstance(model_output, dict):
                loss, _ = compute_total_loss(
                    model_output=model_output,
                    target=msk,
                    epoch=0,
                    original_seg_loss_fn=seg_loss_fn,
                    lambda_edge=config.lambda_edge,
                    warmup_epochs=config.edge_warmup_epochs
                )
                out = model_output["final_output"]
            else:
                gt_pre, out = model_output
                loss = criterion(gt_pre, out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
