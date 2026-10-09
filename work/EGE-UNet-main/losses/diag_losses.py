import torch
import torch.nn.functional as F


def partial_bce(logits, pos_mask, neg_mask, eps=1e-6):
    p = torch.sigmoid(logits)
    pos_loss = -torch.log(p.clamp_min(eps))
    neg_loss = -torch.log((1 - p).clamp_min(eps))
    lp = (pos_loss * pos_mask.float()).sum() / (pos_mask.float().sum() + eps)
    ln = (neg_loss * neg_mask.float()).sum() / (neg_mask.float().sum() + eps)
    return lp + ln


def positive_point_loss(local_logits, point_mask, eps=1e-6):
    p = torch.sigmoid(local_logits)
    loss = -torch.log(p.clamp_min(eps))
    if point_mask.sum() == 0:
        return torch.tensor(0.0, device=local_logits.device)
    return (loss * point_mask.float()).sum() / (point_mask.float().sum() + eps)


def consistency_loss(p_w, p_s, unknown_mask, eps=1e-6):
    diff = (p_w - p_s) ** 2
    if unknown_mask.sum() == 0:
        return torch.tensor(0.0, device=p_w.device)
    return (diff * unknown_mask.float()).sum() / (unknown_mask.float().sum() + eps)


def box_projection_loss(ll_logits, boxes, eps=1e-6):
    prob = torch.sigmoid(ll_logits)
    if prob.dim() == 4 and prob.shape[1] == 1:
        prob = prob.squeeze(1)

    B, H, W = prob.shape
    proj_x = prob.amax(dim=1)
    proj_y = prob.amax(dim=2)

    target_x = torch.zeros(B, W, device=prob.device)
    target_y = torch.zeros(B, H, device=prob.device)

    for b, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = [int(round(v)) for v in box]
        xmin = max(0, min(xmin, W - 1))
        xmax = max(0, min(xmax, W - 1))
        ymin = max(0, min(ymin, H - 1))
        ymax = max(0, min(ymax, H - 1))
        target_x[b, xmin:xmax + 1] = 1.0
        target_y[b, ymin:ymax + 1] = 1.0

    lx = F.binary_cross_entropy(proj_x, target_x)
    ly = F.binary_cross_entropy(proj_y, target_y)
    return lx + ly


def compute_diag_loss(outputs, batch, cfg):
    losses = {}

    losses["partial"] = partial_bce(
        outputs["seg_logits"],
        batch["point_mask"],
        batch["outside_box_mask"],
    )

    total = cfg.lambda_partial * losses["partial"]

    if cfg.enable_local_point_aux:
        losses["point"] = positive_point_loss(
            outputs["local_logits"],
            batch["point_mask"],
        )
        total = total + cfg.lambda_point * losses["point"]

    if cfg.enable_consistency:
        losses["cons"] = consistency_loss(
            outputs["prob_weak"],
            outputs["prob_strong"],
            batch["unknown_mask"],
        )
        total = total + cfg.lambda_consistency * losses["cons"]

    if cfg.enable_box_lf:
        box_tensor = batch["box_coords"]
        if box_tensor.dim() == 2:
            boxes_list = [box_tensor[i].tolist() for i in range(box_tensor.shape[0])]
        else:
            boxes_list = [box_tensor.tolist()]
        losses["box_lf"] = box_projection_loss(
            outputs["ll_logits"],
            boxes_list,
        )
        total = total + cfg.lambda_box_lf * losses["box_lf"]

    losses["total"] = total
    return losses
