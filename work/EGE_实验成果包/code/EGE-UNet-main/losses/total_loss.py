import torch
import torch.nn.functional as F

from losses.boundary_loss import mask_to_boundary, dice_loss


def compute_total_loss(
    model_output,
    target,
    epoch,
    original_seg_loss_fn,
    lambda_edge=0.2,
    warmup_epochs=20
):
    deep_outputs = (
        model_output["deep_supervision"]
    )

    final_output = (
        model_output["final_output"]
    )

    edge_logits = (
        model_output["edge_logits"]
    )

    seg_loss = original_seg_loss_fn(
        deep_outputs,
        final_output,
        target
    )

    edge_gt_full = mask_to_boundary(
        target,
        kernel_size=5
    )

    edge_gt_32 = F.adaptive_max_pool2d(
        edge_gt_full,
        output_size=edge_logits.shape[-2:]
    )

    edge_bce = (
        F.binary_cross_entropy_with_logits(
            edge_logits,
            edge_gt_32
        )
    )

    edge_prob = torch.sigmoid(
        edge_logits
    )

    edge_dice = dice_loss(
        edge_prob,
        edge_gt_32
    )

    edge_loss = (
        edge_bce
        +
        edge_dice
    )

    lambda_edge_current = lambda_edge * min(
        1.0,
        epoch / warmup_epochs
    )

    total_loss = (
        seg_loss
        +
        lambda_edge_current * edge_loss
    )

    loss_dict = {
        "total": total_loss.detach(),
        "seg": seg_loss.detach(),
        "edge": edge_loss.detach(),
        "lambda_edge": lambda_edge_current
    }

    return total_loss, loss_dict
