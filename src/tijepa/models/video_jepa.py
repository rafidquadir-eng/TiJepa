"""Video JEPA (PLAN.md A2, tiers T1/T2): latent-space masked prediction.

Context encoder sees visible tubelets; an EMA target encoder sees the full clip; a
narrow predictor maps context latents to the target encoder's latents at masked
positions. Loss is L1 in latent space (no pixel reconstruction).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .vit import Predictor, VideoViT, sincos_3d


def multiblock_tube_mask(
    grid: tuple[int, int, int],
    n_blocks: int = 8,
    scale: tuple[float, float] = (0.15, 0.3),
    aspect: tuple[float, float] = (0.6, 1.7),
    min_keep: int = 16,
    generator: torch.Generator | None = None,
    device=None,
) -> torch.Tensor:
    """V-JEPA-style multi-block mask, tubed over time.

    Spatial blocks are sampled and masked across every temporal position, so the model
    cannot solve the task by copying the same cell from an adjacent frame.
    Returns a (N,) bool tensor: True = masked (a prediction target).
    """
    t, h, w = grid
    m = torch.zeros(h, w, dtype=torch.bool, device=device)
    for _ in range(n_blocks):
        s = torch.empty(1, device=device).uniform_(
            *scale, generator=generator).item()
        ar = torch.empty(1, device=device).uniform_(*aspect, generator=generator).item()
        bh = max(1, min(h, int(round((s * h * w * ar) ** 0.5))))
        bw = max(1, min(w, int(round((s * h * w / ar) ** 0.5))))
        top = int(torch.randint(0, h - bh + 1, (1,), generator=generator, device=device))
        left = int(torch.randint(0, w - bw + 1, (1,), generator=generator, device=device))
        m[top : top + bh, left : left + bw] = True
    # guarantee some visible context
    if (~m).sum() < min_keep:
        flat = m.flatten()
        masked_idx = flat.nonzero(as_tuple=False).flatten()
        perm = masked_idx[torch.randperm(masked_idx.numel(), generator=generator, device=device)]  # noqa: E501
        flat[perm[: min_keep - int((~m).sum())]] = False
        m = flat.reshape(h, w)
    return m[None, :, :].expand(t, h, w).reshape(-1).clone()


class VideoJEPA(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        pred_dim: int = 192,
        pred_depth: int = 6,
        tubelet: tuple[int, int, int] = (2, 8, 8),
        clip: tuple[int, int, int] = (16, 96, 96),
        ema: tuple[float, float] = (0.996, 1.0),
    ):
        super().__init__()
        self.encoder = VideoViT(in_ch, dim, depth, heads, tubelet, clip)
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(dim, pred_dim, pred_depth, heads)
        self.grid = self.encoder.grid
        self.n_tokens = self.encoder.n_tokens
        self.ema_range = ema
        self.register_buffer(
            "pos_full", sincos_3d(dim, self.grid), persistent=False
        )

    @torch.no_grad()
    def update_target(self, progress: float) -> float:
        lo, hi = self.ema_range
        m = lo + (hi - lo) * min(max(progress, 0.0), 1.0)
        for pt, pc in zip(self.target.parameters(), self.encoder.parameters(), strict=False):
            pt.mul_(m).add_(pc.detach(), alpha=1 - m)
        for bt, bc in zip(self.target.buffers(), self.encoder.buffers(), strict=False):
            bt.copy_(bc)
        return m

    def forward(self, clips: torch.Tensor, mask: torch.Tensor) -> dict:
        """clips: (B,C,T,H,W) float. mask: (N,) bool, True = target position."""
        b = clips.shape[0]
        keep = (~mask)[None, :].expand(b, -1)
        ctx = self.encoder(clips, keep=keep)

        with torch.no_grad():
            tgt_all = self.target(clips)
            tgt_all = torch.nn.functional.layer_norm(tgt_all, (tgt_all.shape[-1],))
            idx_t = mask.nonzero(as_tuple=False).flatten()
            tgt = tgt_all[:, idx_t, :]

        pos = self.pos_full.expand(b, -1, -1)
        pred = self.predictor(
            ctx,
            pos[:, (~mask).nonzero(as_tuple=False).flatten(), :],
            pos[:, mask.nonzero(as_tuple=False).flatten(), :],
        )
        loss = torch.nn.functional.l1_loss(pred, tgt)
        return {"loss": loss, "pred": pred, "target": tgt, "context": ctx}

    @torch.no_grad()
    def embed(self, clips: torch.Tensor, pooled: bool = True) -> torch.Tensor:
        """Representation for probing: target-encoder tokens, mean-pooled by default."""
        z = self.target(clips)
        return z.mean(dim=1) if pooled else z


def collapse_diagnostics(z: torch.Tensor) -> dict:
    """Embedding std per dimension and effective rank (PLAN.md A3)."""
    z = z.float()
    zc = z - z.mean(0, keepdim=True)
    std = zc.std(0)
    cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov.double()).clamp_min(0)
    p = ev / ev.sum().clamp_min(1e-12)
    entropy = -(p * (p + 1e-12).log()).sum()
    return {
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "effective_rank": float(entropy.exp()),
        "n_dims": int(z.shape[1]),
    }
