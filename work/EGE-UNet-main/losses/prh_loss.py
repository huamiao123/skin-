import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import GT_BceDiceLoss, BceDiceLoss


def build_edge_token_gt(mask, token_h=32, token_w=32):
    mask = mask.float()

    ratio = F.adaptive_avg_pool2d(
        mask,
        output_size=(token_h, token_w)
    )

    eps = 1e-6

    edge_gt = (
        (ratio > eps) &
        (ratio < 1.0 - eps)
    ).float()

    edge_gt = edge_gt.flatten(1)

    return edge_gt


class PRHLoss(nn.Module):
    def __init__(self, edge_pos_weight=1.0, lambda_edge=1.0):
        super().__init__()
        self.seg_loss = GT_BceDiceLoss(wb=1, wd=1)
        self.lambda_edge = lambda_edge
        self.edge_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(edge_pos_weight))
        )

    def forward(self, model_output, target):
        gt_pre = model_output["deep_supervision"]
        out = model_output["final_output"]

        seg_loss = self.seg_loss(gt_pre, out, target)

        edge_logits = model_output["edge_logits"]

        edge_gt = build_edge_token_gt(
            target,
            token_h=32,
            token_w=32,
        )

        edge_loss = self.edge_criterion(edge_logits, edge_gt)

        total = seg_loss + self.lambda_edge * edge_loss

        loss_dict = {
            "seg_loss": seg_loss.item(),
            "edge_loss": edge_loss.item(),
            "total_loss": total.item(),
        }

        with torch.no_grad():
            edge_prob = model_output["edge_prob"]
            pos_mask = edge_gt > 0.5
            neg_mask = edge_gt < 0.5
            if pos_mask.sum() > 0:
                loss_dict["edge_pos_prob"] = edge_prob[pos_mask].mean().item()
            else:
                loss_dict["edge_pos_prob"] = 0.0
            if neg_mask.sum() > 0:
                loss_dict["edge_neg_prob"] = edge_prob[neg_mask].mean().item()
            else:
                loss_dict["edge_neg_prob"] = 0.0

        loss_dict["retention_ratio"] = model_output["retention_ratio"]
        loss_dict["gamma"] = model_output["gamma"].item()

        return total, loss_dict
