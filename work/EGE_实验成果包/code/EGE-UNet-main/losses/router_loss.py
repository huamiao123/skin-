import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import BceDiceLoss
from .boundary_loss import dice_loss as dice_loss_fn


class RouterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, router_logits, target, token_size=8):
        B = router_logits.shape[0]
        device = router_logits.device

        with torch.no_grad():
            target_float = target.float()
            patches = F.unfold(target_float, kernel_size=token_size, stride=token_size)
            patch_min = patches.min(dim=1).values
            patch_max = patches.max(dim=1).values
            edge_label = (patch_min != patch_max).float()

        pos_count = edge_label.sum(dim=1).clamp(min=1)
        neg_count = (1.0 - edge_label).sum(dim=1).clamp(min=1)
        pos_weight = neg_count / pos_count

        expanded_pos_weight = pos_weight.unsqueeze(1).expand(-1, edge_label.shape[1])

        bce = F.binary_cross_entropy_with_logits(
            router_logits,
            edge_label,
            weight=expanded_pos_weight,
            reduction='mean',
        )

        pred = (torch.sigmoid(router_logits) >= 0.5).float()
        tp = (pred * edge_label).sum(dim=1)
        fp = (pred * (1 - edge_label)).sum(dim=1)
        fn = ((1 - pred) * edge_label).sum(dim=1)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        stats = {
            "router_loss": bce,
            "router_precision": precision.mean(),
            "router_recall": recall.mean(),
        }

        return bce, stats


class AuxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, aux_logits, target):
        pred = torch.sigmoid(aux_logits)
        target_resized = F.interpolate(target, size=pred.shape[2:], mode='nearest')
        bce_loss = F.binary_cross_entropy(pred, target_resized.float(), reduction='mean')
        d_loss = dice_loss_fn(pred, target_resized)
        return (bce_loss + d_loss) / 2.0


class EGEHRViTLoss(nn.Module):
    def __init__(self, lambda_router=0.2, lambda_aux=0.2,
                 original_seg_loss_fn=None, token_size=8):
        super().__init__()
        self.lambda_router = lambda_router
        self.lambda_aux = lambda_aux
        self.original_seg_loss = original_seg_loss_fn or BceDiceLoss(wb=1, wd=1)
        self.router_loss = RouterLoss()
        self.aux_loss = AuxLoss()
        self.token_size = token_size

    def forward(self, model_output, target, epoch=0):
        gt_pre = model_output["deep_supervision"]
        out = model_output["final_output"]
        adapter_stats = model_output["adapter_stats"]
        aux_logits = model_output["aux_logits"]

        seg_loss = self.original_seg_loss(out, target)

        if gt_pre is not None:
            gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
            deep_loss = (self.original_seg_loss(gt_pre5, target) * 0.1 +
                         self.original_seg_loss(gt_pre4, target) * 0.2 +
                         self.original_seg_loss(gt_pre3, target) * 0.3 +
                         self.original_seg_loss(gt_pre2, target) * 0.4 +
                         self.original_seg_loss(gt_pre1, target) * 0.5)
        else:
            deep_loss = 0.0

        total_seg_loss = seg_loss + deep_loss

        router_logits = adapter_stats.get("router_logits", None)
        router_loss = 0.0
        router_stats = {}
        if router_logits is not None:
            rl, router_stats = self.router_loss(router_logits, target, self.token_size)
            router_loss = rl

        aux_loss = 0.0
        if aux_logits is not None:
            aux_loss = self.aux_loss(aux_logits, target)

        total_loss = total_seg_loss
        if router_logits is not None:
            total_loss = total_loss + self.lambda_router * router_loss
        if aux_logits is not None:
            total_loss = total_loss + self.lambda_aux * aux_loss

        loss_dict = {
            "seg_loss": total_seg_loss.item() if isinstance(total_seg_loss, torch.Tensor) else total_seg_loss,
            "router_loss": router_loss.item() if isinstance(router_loss, torch.Tensor) else router_loss,
            "aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
            "total_loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            **{k: v.item() if isinstance(v, torch.Tensor) else v
               for k, v in router_stats.items()},
        }
        retention_ratio = adapter_stats.get("retention_ratio", None)
        if retention_ratio is not None:
            loss_dict["retention_ratio"] = retention_ratio.item() if isinstance(retention_ratio, torch.Tensor) else retention_ratio

        return total_loss, loss_dict
