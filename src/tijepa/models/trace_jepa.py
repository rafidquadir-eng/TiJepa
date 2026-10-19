"""Trace-JEPA (PLAN.md B2): masked-patch latent prediction over 1D NF-kB traces.

Same JEPA principle as Part A: a context encoder sees visible patches, an EMA target
encoder sees the whole trace, and a narrow predictor maps context latents to target
latents at masked patch positions. L1 loss in latent space.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from .vit import Block, sincos_3d


class TraceEncoder(nn.Module):
    def __init__(self, length: int = 128, patch: int = 4, dim: int = 128,
                 depth: int = 4, heads: int = 4):
        super().__init__()
        assert length % patch == 0
        self.patch, self.n_patches, self.dim = patch, length // patch, dim
        self.embed = nn.Linear(patch, dim)
        self.register_buffer(
            "pos", sincos_3d(dim, (self.n_patches, 1, 1)), persistent=False
        )
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], self.n_patches, self.patch)

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        z = self.embed(self.patchify(x)) + self.pos
        if keep is not None:
            b, n, d = z.shape
            k = int(keep[0].sum().item())
            idx = keep.nonzero(as_tuple=False)[:, 1].reshape(b, k)
            z = torch.gather(z, 1, idx[:, :, None].expand(b, k, d))
        for blk in self.blocks:
            z = blk(z)
        return self.norm(z)


class TraceJEPA(nn.Module):
    def __init__(self, length: int = 128, patch: int = 4, dim: int = 128,
                 depth: int = 4, heads: int = 4, pred_dim: int = 64, pred_depth: int = 2,
                 ema: tuple[float, float] = (0.996, 1.0)):
        super().__init__()
        self.encoder = TraceEncoder(length, patch, dim, depth, heads)
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.n_patches = self.encoder.n_patches
        self.pred_embed = nn.Linear(dim, pred_dim)
        self.pos_proj = nn.Linear(dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pred_blocks = nn.ModuleList([Block(pred_dim, 4) for _ in range(pred_depth)])
        self.pred_norm = nn.LayerNorm(pred_dim)
        self.pred_out = nn.Linear(pred_dim, dim)
        self.ema_range = ema

    @torch.no_grad()
    def update_target(self, progress: float) -> float:
        lo, hi = self.ema_range
        m = lo + (hi - lo) * min(max(progress, 0.0), 1.0)
        for pt, pc in zip(self.target.parameters(), self.encoder.parameters(), strict=False):
            pt.mul_(m).add_(pc.detach(), alpha=1 - m)
        return m

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict:
        b = x.shape[0]
        keep = (~mask)[None, :].expand(b, -1)
        ctx = self.encoder(x, keep=keep)
        with torch.no_grad():
            t = self.target(x)
            t = torch.nn.functional.layer_norm(t, (t.shape[-1],))
            tgt = t[:, mask.nonzero(as_tuple=False).flatten(), :]
        pos = self.encoder.pos.expand(b, -1, -1)
        c = self.pred_embed(ctx) + self.pos_proj(
            pos[:, (~mask).nonzero(as_tuple=False).flatten(), :])
        q = self.mask_token.expand(b, int(mask.sum()), -1) + self.pos_proj(
            pos[:, mask.nonzero(as_tuple=False).flatten(), :])
        z = torch.cat([c, q], dim=1)
        for blk in self.pred_blocks:
            z = blk(z)
        pred = self.pred_out(self.pred_norm(z[:, c.shape[1]:]))
        return {"loss": torch.nn.functional.l1_loss(pred, tgt)}

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.target(x).mean(dim=1)


def random_patch_mask(n_patches: int, ratio: float = 0.5,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    k = max(1, min(n_patches - 1, int(round(n_patches * ratio))))
    perm = torch.randperm(n_patches, generator=generator)
    m = torch.zeros(n_patches, dtype=torch.bool)
    m[perm[:k]] = True
    return m


def trace_features(win: np.ndarray, frame_min: float = 6.0) -> tuple[np.ndarray, list[str]]:
    """Hand-crafted trace features (PLAN.md B2), computed on a baseline-corrected window."""
    x = np.nan_to_num(win, nan=0.0)
    n, L = x.shape
    t = np.arange(L) * frame_min
    feats, names = [], []

    def add(v, name):
        feats.append(np.asarray(v, dtype=float).reshape(n, -1))
        names.extend([name] if feats[-1].shape[1] == 1 else
                     [f"{name}_{i}" for i in range(feats[-1].shape[1])])

    add(x[:, :3].mean(1), "baseline")
    add(x.max(1), "peak_amplitude")
    add(t[np.argmax(x, axis=1)], "peak_time_min")
    add(np.trapezoid(x, dx=frame_min, axis=1), "auc")
    add(np.trapezoid(x[:, : L // 2], dx=frame_min, axis=1), "auc_first_half")
    add(np.trapezoid(x[:, L // 2:], dx=frame_min, axis=1), "auc_second_half")
    add(x.mean(1), "mean")
    add(x.std(1), "std")

    # number of peaks: local maxima above a per-cell threshold
    thr = x.mean(1, keepdims=True) + 0.5 * x.std(1, keepdims=True)
    above = x > thr
    n_peaks = (above[:, 1:-1] & (x[:, 1:-1] >= x[:, :-2]) & (x[:, 1:-1] > x[:, 2:])).sum(1)
    add(n_peaks, "n_peaks")

    # oscillation: power spectrum peak and its period
    xc = x - x.mean(1, keepdims=True)
    ps = np.abs(np.fft.rfft(xc, axis=1)) ** 2
    freqs = np.fft.rfftfreq(L, d=frame_min)
    if ps.shape[1] > 1:
        k = 1 + np.argmax(ps[:, 1:], axis=1)
        add(ps[np.arange(n), k], "fft_peak_power")
        with np.errstate(divide="ignore"):
            add(np.where(freqs[k] > 0, 1.0 / np.maximum(freqs[k], 1e-12), 0.0),
                "fft_peak_period_min")
        add(ps[:, 1:].sum(1), "fft_total_power")

    # decay half-life after the peak
    half = np.zeros(n)
    for i in range(n):
        pk = int(np.argmax(x[i]))
        target = x[i, pk] / 2.0
        after = np.flatnonzero(x[i, pk:] <= target)
        half[i] = after[0] * frame_min if after.size else (L - pk) * frame_min
    add(half, "decay_half_life_min")

    # late/early ratio: sustained vs transient response
    early = np.trapezoid(x[:, : max(1, L // 4)], dx=frame_min, axis=1)
    late = np.trapezoid(x[:, -max(1, L // 4):], dx=frame_min, axis=1)
    add(late / np.where(np.abs(early) < 1e-9, 1e-9, early), "late_over_early")
    return np.concatenate(feats, axis=1), names
