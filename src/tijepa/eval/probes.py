"""Part A evaluation: the six encoders of PLAN.md A4 and the frozen-probe protocol.

E1 hand-crafted morphology (+ motility where a dataset is temporal)
E2 ImageNet ViT-S frame-mean
E3 DINOv2-S/14 frame-mean
E4 TiJepa-0
E5 TiJepa-0 on frame-shuffled clips
E6 TiJepa-0 random-init (architecture control)

All encoders are frozen; the probe is identical for every encoder.
"""
from __future__ import annotations

import numpy as np
import torch


# ----------------------------------------------------------------- E1 hand-crafted
def morphology_features(crop: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Shape + intensity + Haralick texture for one 2D crop."""
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops

    img = crop.astype(np.float32)
    rng = img.max() - img.min()
    norm = (img - img.min()) / (rng + 1e-6)
    names = ["area", "perimeter", "eccentricity", "solidity", "major_axis",
             "minor_axis", "mean_intensity", "std_intensity"]
    vals = [0.0] * len(names)
    try:
        thr = threshold_otsu(norm)
        mask = norm > thr
        cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
        lab = label(mask)
        if lab[cy, cx] == 0:
            props = sorted(regionprops(lab), key=lambda r: -r.area)
        else:
            props = [r for r in regionprops(lab) if r.label == lab[cy, cx]]
        if props:
            r = props[0]
            vals = [float(r.area), float(r.perimeter), float(r.eccentricity),
                    float(r.solidity), float(r.major_axis_length),
                    float(r.minor_axis_length), float(norm[mask].mean()),
                    float(norm[mask].std())]
    except (ValueError, IndexError):
        pass

    q = np.clip((norm * 31).astype(np.uint8), 0, 31)
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi / 2], levels=32,
                        symmetric=True, normed=True)
    tex_names = ["contrast", "homogeneity", "energy", "correlation", "dissimilarity",
                 "ASM"]
    tex = []
    for p in tex_names:
        v = graycoprops(glcm, p)
        tex.append(float(np.nanmean(v)))
    return np.asarray(vals + tex, dtype=float), names + [f"haralick_{t}" for t in tex_names]


def motility_features(ys: np.ndarray, xs: np.ndarray, frame_interval_s: float
                      ) -> tuple[np.ndarray, list[str]]:
    """Mean speed, persistence, MSD exponent, turning angles, fraction moving."""
    d = np.sqrt(np.diff(ys) ** 2 + np.diff(xs) ** 2)
    total = float(d.sum())
    net = float(np.hypot(ys[-1] - ys[0], xs[-1] - xs[0]))
    speed = d / frame_interval_s
    # MSD exponent from a log-log fit
    lags = np.arange(1, max(2, len(ys) // 2))
    msd = np.array([np.mean((ys[l:] - ys[:-l]) ** 2 + (xs[l:] - xs[:-l]) ** 2)
                    for l in lags])
    ok = (msd > 0) & np.isfinite(msd)
    alpha = float(np.polyfit(np.log(lags[ok]), np.log(msd[ok]), 1)[0]) if ok.sum() > 1 else 0.0
    vy, vx = np.diff(ys), np.diff(xs)
    ang = np.arctan2(vy, vx)
    turn = np.diff(ang)
    turn = (turn + np.pi) % (2 * np.pi) - np.pi
    names = ["mean_speed", "max_speed", "persistence", "msd_alpha",
             "turn_abs_mean", "turn_std", "frac_moving", "total_path", "net_disp"]
    vals = [float(speed.mean()), float(speed.max()),
            float(net / total) if total > 0 else 0.0, alpha,
            float(np.abs(turn).mean()) if turn.size else 0.0,
            float(turn.std()) if turn.size else 0.0,
            float((d > np.median(d) * 0.5).mean()) if d.size else 0.0,
            total, net]
    return np.asarray(vals, dtype=float), names


# --------------------------------------------------------------- timm frame encoders
def load_timm_encoder(name: str, device: str = "cuda"):
    """Frozen timm encoder; returns (model, preprocess_fn, licence_string)."""
    import timm

    model = timm.create_model(name, pretrained=True, num_classes=0)
    model.eval().to(device)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)
    size = cfg["input_size"][-1]

    def prep(x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,H,W) in standardised units -> (B,3,size,size) normalised."""
        x = x - x.amin(dim=(1, 2, 3), keepdim=True)
        x = x / (x.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
        x = x.repeat(1, 3, 1, 1)
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                            align_corners=False)
        return (x - mean) / std

    return model, prep, cfg.get("license", "see manifests/models.yaml")


@torch.no_grad()
def embed_frames_mean(model, prep, clips: torch.Tensor, device: str = "cuda",
                      batch: int = 256, amp=torch.bfloat16) -> np.ndarray:
    """Frame-mean embedding of clips (B,1,T,H,W) with a 2D frame encoder."""
    out = []
    B, _, T, H, W = clips.shape
    flat = clips.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)
    for i in range(0, flat.shape[0], batch):
        x = prep(flat[i:i + batch].to(device))
        with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
            f = model(x)
        out.append(f.float().cpu())
    f = torch.cat(out).reshape(B, T, -1).mean(dim=1)
    return f.numpy()


@torch.no_grad()
def embed_tijepa(model, clips: torch.Tensor, device: str = "cuda", batch: int = 128,
                 shuffle_frames: bool = False, seed: int = 0,
                 amp=torch.bfloat16) -> np.ndarray:
    out = []
    g = torch.Generator().manual_seed(seed)
    for i in range(0, clips.shape[0], batch):
        x = clips[i:i + batch].to(device)
        if shuffle_frames:
            perm = torch.randperm(x.shape[2], generator=g).to(device)
            x = x[:, :, perm]
        with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
            z = model.embed(x)
        out.append(z.float().cpu())
    return torch.cat(out).numpy()
