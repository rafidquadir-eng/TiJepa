"""Frozen encoders applied to static single-channel images (v3 coverage-map datasets).

Conventions are those of the BBBC054 benchmark (scripts/62_bbbc054_v1.py) so results are
comparable across datasets: each image is min-max scaled to [0, 1]; image models get it
replicated to 3 channels, resized to their native input and normalised with their own
mean/std; video models get it repeated over 16 frames; TiJepa models see a 96 x 96 input,
z-scored per clip. Every learned encoder carries its checkpoint record (provenance).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tijepa.eval import provenance as PV
from tijepa.models.video_jepa import VideoJEPA

REPO = Path(__file__).resolve().parents[3]
NF, CROP = 16, 96
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TIMM = {
    "E2_imagenet_vit_s": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "E3_dinov2_vit_s": "vit_small_patch14_dinov2.lvd142m",
}
JEPA_RUNS = (
    ("tijepa1", "tijepa1_T1_seed0.pt"),
    ("tijepa1_spatial", "tijepa1_spatial_T1_seed0.pt"),
    ("tijepa0", "tijepa0_T1_seed0.pt"),
)
RANDOM_SEEDS = (5000, 5001, 5002)


def minmax(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    lo = X.reshape(len(X), -1).min(1)[:, None, None]
    hi = X.reshape(len(X), -1).max(1)[:, None, None]
    return (X - lo) / (hi - lo + 1e-6)


def resize(X: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(X.astype(np.float32))[:, None]
    return torch.nn.functional.interpolate(
        t, size=(size, size), mode="bilinear", align_corners=False
    )[:, 0].numpy()


@torch.no_grad()
def embed_timm(name: str, X: np.ndarray, batch: int = 256) -> np.ndarray:
    import timm

    model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(DEV)
    cfg = model.pretrained_cfg
    size = cfg.get("input_size", (3, 224, 224))[1]
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
    outs = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i : i + batch])[:, None].repeat(1, 3, 1, 1)
        x = torch.nn.functional.interpolate(
            x, size=(size, size), mode="bilinear", align_corners=False
        )
        outs.append(model(((x - mean) / std).to(DEV)).float().cpu().numpy())
    return np.concatenate(outs)


def load_jepa(run: str, fn: str) -> tuple[VideoJEPA, dict]:
    path = REPO / "checkpoints" / fn
    sd = torch.load(path, map_location="cpu", weights_only=False)
    rec = PV.checkpoint_record(path, sd)
    if run != "tijepa0" and not rec["final"]:
        raise SystemExit(f"{fn} is not at its planned final step")
    m = VideoJEPA(clip=(NF, CROP, CROP))
    m.load_state_dict(sd["model"])
    return m.eval().to(DEV), rec


@torch.no_grad()
def embed_jepa(model: VideoJEPA, X: np.ndarray, batch: int = 128) -> np.ndarray:
    X = resize(X, CROP) if X.shape[-1] != CROP else X
    outs = []
    for i in range(0, len(X), batch):
        t = torch.from_numpy(X[i : i + batch])[:, None, None].repeat(1, 1, NF, 1, 1)
        m = t.mean(dim=(1, 2, 3, 4), keepdim=True)
        s = t.std(dim=(1, 2, 3, 4), keepdim=True)
        c = ((t - m) / (s + 1e-6)).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            outs.append(model.embed(c).float().cpu().numpy())
    return np.concatenate(outs)


def all_encoders(include=("E2", "E3", "E7", "E4", "E4s", "E0", "E6")):
    """Yield (key, label-provenance record or None, embed function X -> Z).

    X is (N, H, W) float in [0, 1]. Each embed function is created lazily so only one
    model is on the GPU at a time.
    """
    if "E2" in include:
        yield "E2_imagenet_vit_s", None, lambda X: embed_timm(TIMM["E2_imagenet_vit_s"], X)
    if "E3" in include:
        yield "E3_dinov2_vit_s", None, lambda X: embed_timm(TIMM["E3_dinov2_vit_s"], X)
    if "E7" in include:
        from tijepa.eval import vjepa2 as VJ2

        yield VJ2.KEY, None, lambda X: VJ2.embed_frames(X, DEV)
    tags = {"tijepa1": "E4", "tijepa1_spatial": "E4s", "tijepa0": "E0"}
    for run, fn in JEPA_RUNS:
        if tags[run] not in include:
            continue
        m, rec = load_jepa(run, fn)
        yield PV.key_for(rec), rec, (lambda X, m=m: embed_jepa(m, X))
        del m
        torch.cuda.empty_cache()
    if "E6" in include:
        for s in RANDOM_SEEDS:
            torch.manual_seed(s)
            m = VideoJEPA(clip=(NF, CROP, CROP)).eval().to(DEV)
            yield (
                f"E6_random_init_draw{s - RANDOM_SEEDS[0]}",
                PV.random_init_record(s),
                (lambda X, m=m: embed_jepa(m, X)),
            )
