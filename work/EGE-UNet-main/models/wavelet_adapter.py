import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    def forward(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5

        return ll, lh, hl, hh


class HaarIDWT2D(nn.Module):
    def forward(self, ll, lh, hl, hh):
        a = (ll - lh - hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll + lh + hl + hh) * 0.5

        B, C, H, W = a.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=a.device, dtype=a.dtype)
        out[:, :, 0::2, 0::2] = a
        out[:, :, 0::2, 1::2] = b
        out[:, :, 1::2, 0::2] = c
        out[:, :, 1::2, 1::2] = d

        return out


class ConvPosEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pos = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)

    def forward(self, x):
        return x + self.pos(x)


class LFTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        q = self.norm1(x)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SpatialWaveGate(nn.Module):
    def __init__(self, channels, hidden_channels=None, gate_init=0.1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(channels // 2, 8)

        self.proj = nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                                padding=1, groups=hidden_channels, bias=True)
        self.act = nn.GELU()
        self.out_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)

        nn.init.zeros_(self.out_conv.weight)
        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
        init_bias = math.log(gate_init / (1.0 - gate_init))
        nn.init.constant_(self.out_conv.bias, init_bias)

    def forward(self, x, x_wave):
        g = torch.cat([x, x_wave], dim=1)
        g = self.proj(g)
        g = self.act(g)
        g = self.dwconv(g)
        g = self.act(g)
        gate = torch.sigmoid(self.out_conv(g))
        delta = x_wave - x
        out = x + gate * delta
        return out, gate


class WaveletGlobalAdapter(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, depth=1,
                 alpha_init=0.1, fusion_type='scalar', mode='full'):
        super().__init__()
        assert mode in {'full', 'll_only', 'no_ll_transformer', 'hf_only'}, \
            f'unknown wave mode: {mode}'
        self.dim = dim
        self.fusion_type = fusion_type
        self.mode = mode
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()
        self.pos = ConvPosEncoding(dim)

        self.blocks = nn.ModuleList([
            LFTransformerBlock(dim=dim, num_heads=num_heads,
                              mlp_ratio=mlp_ratio, dropout=0.0)
            for _ in range(depth)
        ])

        if fusion_type == 'scalar':
            init_logit = math.log(alpha_init / (1.0 - alpha_init))
            self.alpha_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        elif fusion_type == 'spatial_gate':
            self.spatial_gate = SpatialWaveGate(
                channels=dim,
                hidden_channels=max(dim // 2, 8),
                gate_init=alpha_init,
            )

    def _subband_energy_stats(self, ll, lh, hl, hh):
        with torch.no_grad():
            e_ll = ll.detach().pow(2).mean().item()
            e_lh = lh.detach().pow(2).mean().item()
            e_hl = hl.detach().pow(2).mean().item()
            e_hh = hh.detach().pow(2).mean().item()
            e_total = e_ll + e_lh + e_hl + e_hh + 1e-12
            r_hf = (e_lh + e_hl + e_hh) / e_total
        return {
            'e_ll': e_ll,
            'e_lh': e_lh,
            'e_hl': e_hl,
            'e_hh': e_hh,
            'r_hf': r_hf,
        }

    def forward(self, x, return_gate=False):
        orig_h, orig_w = x.shape[-2:]

        pad_h = orig_h % 2
        pad_w = orig_w % 2
        if pad_h or pad_w:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_pad = x

        ll, lh, hl, hh = self.dwt(x_pad)

        if self.mode == 'full':
            ll_t = ll
            lh_out, hl_out, hh_out = lh, hl, hh
        elif self.mode == 'll_only':
            ll_t = ll
            lh_out = torch.zeros_like(lh)
            hl_out = torch.zeros_like(hl)
            hh_out = torch.zeros_like(hh)
        elif self.mode == 'no_ll_transformer':
            ll_t = ll
            lh_out, hl_out, hh_out = lh, hl, hh
        elif self.mode == 'hf_only':
            ll_t = torch.zeros_like(ll)
            lh_out, hl_out, hh_out = lh, hl, hh

        if self.mode != 'no_ll_transformer':
            ll_t = self.pos(ll_t)

            B, C, H, W = ll_t.shape
            z = ll_t.flatten(2).transpose(1, 2)
            for block in self.blocks:
                z = block(z)
            ll_t = z.transpose(1, 2).reshape(B, C, H, W)

        x_rec = self.idwt(ll_t, lh_out, hl_out, hh_out)
        x_rec = x_rec[..., :orig_h, :orig_w]

        energy = self._subband_energy_stats(ll, lh, hl, hh)

        if self.fusion_type == 'scalar':
            alpha = torch.sigmoid(self.alpha_logit)
            delta = x_rec - x
            r_delta = delta.norm(p=2) / (x.norm(p=2) + 1e-8)
            out = x + alpha * delta
            stats = {'alpha': alpha, 'r_delta': r_delta, **energy}
            if return_gate:
                gate = torch.full((B, 1, orig_h, orig_w), alpha.item(),
                                  device=x.device, dtype=x.dtype)
                return out, gate
            return out, stats
        elif self.fusion_type == 'spatial_gate':
            out, gate = self.spatial_gate(x, x_rec)
            delta = x_rec - x
            r_delta = delta.norm(p=2) / (x.norm(p=2) + 1e-8)
            with torch.no_grad():
                gv = gate.detach().flatten()
                gate_mean = gv.mean()
                gate_std = gv.std()
                gate_p10 = torch.quantile(gv, 0.1)
                gate_p50 = torch.quantile(gv, 0.5)
                gate_p90 = torch.quantile(gv, 0.9)
            stats = {
                'r_delta': r_delta,
                'gate_mean': gate_mean,
                'gate_std': gate_std,
                'gate_p10': gate_p10,
                'gate_p50': gate_p50,
                'gate_p90': gate_p90,
                **energy,
            }
            if return_gate:
                return out, gate
            return out, stats
