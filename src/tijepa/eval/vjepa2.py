"""Frozen V-JEPA 2 ViT-L as a pretrained video baseline (E7; PLAN_v3 H1.3, finding R7).

Weights: Hugging Face `facebook/vjepa2-vitl-fpc64-256`, pinned revision below, MIT licence
(recorded in manifests/models.yaml). Inputs follow the model's own preprocessing config:
grayscale replicated to 3 channels, resized to 256 x 256, ImageNet mean/std, 16 frames
(static images: the frame repeated). The representation is the mean of the encoder's
last-layer patch tokens, the analogue of TiJepa's mean-pooled target-encoder tokens.
"""
from __future__ import annotations

import numpy as np
import torch

REPO_ID = "facebook/vjepa2-vitl-fpc64-256"
REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
SAFETENSORS_SHA256 = "25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b"
LICENCE = "MIT (model card and repository tag, checked 2026-10-19)"
SIZE, N_FRAMES = 256, 16
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
KEY = "E7_vjepa2_vitl"

_MODEL = None


def record() -> dict:
    return {"source": f"huggingface:{REPO_ID}", "revision": REVISION,
            "safetensors_sha256": SAFETENSORS_SHA256, "licence": LICENCE,
            "input": f"grayscale x3, {SIZE}x{SIZE}, ImageNet mean/std, {N_FRAMES} frames",
            "pooling": "mean of last-layer patch tokens"}


def load(device: str = "cuda"):
    global _MODEL
    if _MODEL is None:
        from transformers import VJEPA2Model
        _MODEL = VJEPA2Model.from_pretrained(REPO_ID, revision=REVISION,
                                             dtype=torch.float32)
        _MODEL = _MODEL.eval().to(device)
    return _MODEL


def _prep(x: torch.Tensor) -> torch.Tensor:
    """(B,T,H,W) float -> (B,T,3,SIZE,SIZE) normalised."""
    b, t, h, w = x.shape
    x = x.reshape(b * t, 1, h, w)
    x = torch.nn.functional.interpolate(x, size=(SIZE, SIZE), mode="bilinear",
                                        align_corners=False).repeat(1, 3, 1, 1)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).reshape(b, t, 3, SIZE, SIZE)


@torch.no_grad()
def embed_frames(X: np.ndarray, device: str = "cuda", batch: int = 16) -> np.ndarray:
    """Static images (N,H,W), each repeated to 16 frames, as for every video encoder."""
    model = load(device)
    out = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(np.asarray(X[i:i + batch], np.float32)).to(device)
        x = x[:, None].expand(-1, N_FRAMES, -1, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            h = model(pixel_values_videos=_prep(x)).last_hidden_state
        out.append(h.float().mean(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_clips(clips: np.ndarray, device: str = "cuda", batch: int = 16) -> np.ndarray:
    """Video clips (N,T,H,W) with T == 16."""
    model = load(device)
    out = []
    for i in range(0, len(clips), batch):
        x = torch.from_numpy(np.asarray(clips[i:i + batch], np.float32)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            h = model(pixel_values_videos=_prep(x)).last_hidden_state
        out.append(h.float().mean(1).cpu().numpy())
    return np.concatenate(out)
