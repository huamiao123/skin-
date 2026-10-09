import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeakLosses:
    @staticmethod
    def box_mask_to_boxes(box_masks):
        boxes = []
        for i in range(box_masks.shape[0]):
            bm = box_masks[i, 0].cpu().numpy()
            ys, xs = np.where(bm)
            if len(xs) == 0:
                boxes.append([0, 0, 1, 1])
            else:
                boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
        return boxes

    @staticmethod
    def balanced_partial_bce(logits, pos_mask, neg_mask):
        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]

        if pos_logits.numel() > 0:
            loss_pos = F.binary_cross_entropy_with_logits(
                pos_logits,
                torch.ones_like(pos_logits)
            )
        else:
            loss_pos = torch.tensor(0.0, device=logits.device)

        if neg_logits.numel() > 0:
            loss_neg = F.binary_cross_entropy_with_logits(
                neg_logits,
                torch.zeros_like(neg_logits)
            )
        else:
            loss_neg = torch.tensor(0.0, device=logits.device)

        return 0.5 * loss_pos + 0.5 * loss_neg

    @staticmethod
    def box_projection_targets(boxes, h, w, device):
        B = len(boxes)
        tx = torch.zeros((B, w), device=device)
        ty = torch.zeros((B, h), device=device)

        for b, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = [int(round(v)) for v in box]
            xmin = max(0, min(xmin, w - 1))
            xmax = max(0, min(xmax, w - 1))
            ymin = max(0, min(ymin, h - 1))
            ymax = max(0, min(ymax, h - 1))
            tx[b, xmin:xmax + 1] = 1.0
            ty[b, ymin:ymax + 1] = 1.0

        return tx, ty

    @staticmethod
    def projection_loss(ll_logits, boxes):
        prob = torch.sigmoid(ll_logits).squeeze(1)
        proj_x = prob.amax(dim=1)
        proj_y = prob.amax(dim=2)

        B, H, W = prob.shape
        tx, ty = WeakLosses.box_projection_targets(boxes, H, W, prob.device)

        lx = F.binary_cross_entropy(proj_x, tx)
        ly = F.binary_cross_entropy(proj_y, ty)

        return 0.5 * (lx + ly)

    @staticmethod
    def local_positive_loss(local_logits, pos_mask):
        vals = local_logits[pos_mask]

        if vals.numel() == 0:
            return torch.tensor(0.0, device=local_logits.device)

        return F.binary_cross_entropy_with_logits(
            vals,
            torch.ones_like(vals)
        )

    @staticmethod
    def unknown_consistency_loss(logits_w, logits_s, unknown_mask):
        pw = torch.sigmoid(logits_w).detach()
        ps = torch.sigmoid(logits_s)

        if unknown_mask.sum() == 0:
            return torch.tensor(0.0, device=logits_s.device)

        return F.mse_loss(ps[unknown_mask], pw[unknown_mask])
