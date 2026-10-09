import torch
import torch.nn.functional as F


def build_edge_token_label(mask, token_size=8):
    B = mask.shape[0]
    mask = mask.float()

    patches = F.unfold(
        mask,
        kernel_size=token_size,
        stride=token_size
    )

    patch_min = patches.min(dim=1).values
    patch_max = patches.max(dim=1).values

    edge_label = (patch_min != patch_max).float()
    return edge_label


def build_keep_mask(router_prob, threshold=0.5, min_keep_ratio=0.15):
    B, N = router_prob.shape

    keep_mask = (router_prob >= threshold).float()

    min_keep = max(1, int(N * min_keep_ratio))
    keep_counts = keep_mask.sum(dim=1)

    for b in range(B):
        if keep_counts[b] < min_keep:
            _, top_idx = router_prob[b].topk(min_keep)
            keep_mask[b, :] = 0
            keep_mask[b, top_idx] = 1

    return keep_mask


def spatial_expand(keep_mask, H, W, kernel_size=3):
    B, N = keep_mask.shape
    assert N == H * W

    keep_map = keep_mask.view(B, 1, H, W)

    pad = kernel_size // 2
    expanded = F.max_pool2d(keep_map, kernel_size=kernel_size, stride=1, padding=pad)

    return expanded.view(B, N)


def gather_active_tokens(x, keep_mask):
    B, N, C = x.shape
    device = x.device

    active_idx_list = []
    x_active_list = []
    max_active = 0

    for b in range(B):
        idx = keep_mask[b].nonzero(as_tuple=False).squeeze(-1)
        active_idx_list.append(idx)
        x_active_list.append(x[b, idx])
        max_active = max(max_active, len(idx))

    if max_active == 0:
        max_active = 1

    x_active = torch.zeros(B, max_active, C, device=device)
    padding_mask = torch.zeros(B, max_active, device=device)

    for b in range(B):
        n_active = len(active_idx_list[b])
        if n_active > 0:
            x_active[b, :n_active] = x_active_list[b]
            padding_mask[b, :n_active] = 1

    return x_active, active_idx_list, padding_mask


def reconstruct_tokens(memory, x_active, active_idx, padding_mask):
    B, N, C = memory.shape
    device = memory.device

    recon = memory.clone()

    for b in range(B):
        idx = active_idx[b]
        n_active = int(padding_mask[b].sum().item())
        if n_active > 0:
            recon[b, idx] = x_active[b, :n_active]

    return recon
