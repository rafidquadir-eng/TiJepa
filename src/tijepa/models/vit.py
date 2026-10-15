"""Vision transformer over spatio-temporal tubelets, for TiJepa-0.

Shapes follow PLAN.md A2: 96x96 px, 16 frames, tubelet = 2 frames x 8x8 px
-> grid (T/2, H/8, W/8) = (8, 12, 12) = 1152 tokens.

The encoder accepts an optional boolean keep-mask so the context encoder can run on
visible tokens only (JEPA-style) without wasting compute on masked positions.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sincos_3d(dim: int, grid: tuple[int, int, int], device=None) -> torch.Tensor:
    """Fixed 3D sine-cosine positional embedding, (1, T*H*W, dim).

    dim is split across the three axes; any remainder is given to the spatial axes.
    """
    t, h, w = grid
    per = dim // 3
    per -= per % 2
    dims = [per, per, dim - 2 * per]

    def axis_emb(length: int, d: int) -> torch.Tensor:
        if d <= 0:
            return torch.zeros(length, 0, device=device)
        pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        omega = torch.exp(
            torch.arange(0, d, 2, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / max(d, 1))
        )[None, :]
        ang = pos * omega
        return torch.cat([ang.sin(), ang.cos()], dim=1)[:, :d]

    et, eh, ew = (axis_emb(n, d) for n, d in zip((t, h, w), dims, strict=False))
    out = torch.cat(
        [
            et[:, None, None, :].expand(t, h, w, dims[0]),
            eh[None, :, None, :].expand(t, h, w, dims[1]),
            ew[None, None, :, :].expand(t, h, w, dims[2]),
        ],
        dim=-1,
    )
    return out.reshape(1, t * h * w, dim)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block using fused scaled-dot-product attention."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


class TubeletEmbed(nn.Module):
    """Conv3d tubelet embedding: (B,C,T,H,W) -> (B, N, dim)."""

    def __init__(self, in_ch: int = 1, dim: int = 384, tubelet: tuple[int, int, int] = (2, 8, 8)):  # noqa: E501
        super().__init__()
        self.tubelet = tubelet
        self.proj = nn.Conv3d(in_ch, dim, kernel_size=tubelet, stride=tubelet)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        z = self.proj(x)
        grid = (z.shape[2], z.shape[3], z.shape[4])
        return z.flatten(2).transpose(1, 2), grid


class VideoViT(nn.Module):
    """ViT encoder over tubelets. dim/depth/heads default to ViT-S."""

    def __init__(
        self,
        in_ch: int = 1,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        tubelet: tuple[int, int, int] = (2, 8, 8),
        clip: tuple[int, int, int] = (16, 96, 96),
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.tubelet = tubelet
        self.grid = (clip[0] // tubelet[0], clip[1] // tubelet[1], clip[2] // tubelet[2])
        self.n_tokens = self.grid[0] * self.grid[1] * self.grid[2]
        self.patch_embed = TubeletEmbed(in_ch, dim, tubelet)
        self.register_buffer("pos", sincos_3d(dim, self.grid), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B,C,T,H,W). keep: (B,N) bool, True = feed to the encoder.

        With `keep`, every row must keep the same number of tokens (JEPA samples one
        mask per batch), so the gathered tensor stays rectangular.
        """
        z, grid = self.patch_embed(x)
        if grid != self.grid:
            z = z + sincos_3d(self.dim, grid, device=z.device)
        else:
            z = z + self.pos
        if keep is not None:
            b, n, d = z.shape  # noqa: E501
            k = int(keep[0].sum().item())
            idx = keep.nonzero(as_tuple=False)[:, 1].reshape(b, k)
            z = torch.gather(z, 1, idx[:, :, None].expand(b, k, d))
        for blk in self.blocks:
            z = blk(z)
        return self.norm(z)


class Predictor(nn.Module):
    """Narrow transformer predicting target latents at masked positions."""

    def __init__(self, dim: int = 384, pred_dim: int = 192, depth: int = 6, heads: int = 6):
        super().__init__()
        self.embed = nn.Linear(dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_proj = nn.Linear(dim, pred_dim)
        self.blocks = nn.ModuleList([Block(pred_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.out = nn.Linear(pred_dim, dim)

    def forward(
        self,
        ctx: torch.Tensor,
        pos_ctx: torch.Tensor,
        pos_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """ctx: (B,Nc,dim) context latents; pos_*: positional embeddings (B,N*,dim)."""
        c = self.embed(ctx) + self.pos_proj(pos_ctx)
        t = self.mask_token.expand(ctx.shape[0], pos_tgt.shape[1], -1) + self.pos_proj(pos_tgt)
        z = torch.cat([c, t], dim=1)
        for blk in self.blocks:
            z = blk(z)
        return self.out(self.norm(z[:, c.shape[1]:]))


def vit_small(**kw) -> VideoViT:
    return VideoViT(dim=384, depth=12, heads=6, **kw)
