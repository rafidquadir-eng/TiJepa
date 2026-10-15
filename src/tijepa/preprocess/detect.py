"""Classical brightfield/phase cell detection, tracking and clip extraction.

PLAN.md A1 permits "a classical phase/brightfield pipeline if Cellpose is unavailable".
We use the classical route for the pretraining corpora by choice and record it as a
deviation: the labelled evaluation sets do not need segmentation (BBBC054 ships
annotated cell coordinates and S-BIAD2196 ships instance masks with consistent IDs),
so Cellpose would only have added GPU contention with training for no evaluation gain.

Detection: temporal-median background, absolute deviation, robust (MAD) threshold,
area filter, centroid extraction. Linking: trackpy.
"""
from __future__ import annotations

import numpy as np

TARGET_UM_PER_PX = 0.65


def temporal_background(frames: np.ndarray, n_sample: int = 32) -> np.ndarray:
    idx = np.linspace(0, frames.shape[0] - 1, min(n_sample, frames.shape[0])).astype(int)
    return np.median(frames[idx].astype(np.float32), axis=0)


def detect_frame(
    frame: np.ndarray,
    bg: np.ndarray,
    k_mad: float = 4.0,
    min_area_px: int = 25,
    max_area_px: int = 4000,
) -> np.ndarray:
    """Return (n, 2) array of (y, x) centroids for one frame."""
    from skimage.measure import label, regionprops

    d = np.abs(frame.astype(np.float32) - bg)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) + 1e-6
    mask = d > med + k_mad * 1.4826 * mad
    if mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    lab = label(mask)
    cents = [
        r.centroid for r in regionprops(lab)
        if min_area_px <= r.area <= max_area_px
    ]
    return np.asarray(cents, dtype=np.float32).reshape(-1, 2)


def detect_movie(
    frames: np.ndarray,
    stride: int = 1,
    k_mad: float = 4.0,
    min_area_px: int = 25,
    max_area_px: int = 4000,
) -> object:
    """Detect in every `stride`-th frame; returns a DataFrame for trackpy."""
    import pandas as pd

    bg = temporal_background(frames)
    rows = []
    for t in range(0, frames.shape[0], stride):
        c = detect_frame(frames[t], bg, k_mad, min_area_px, max_area_px)
        for y, x in c:
            rows.append({"frame": t, "y": float(y), "x": float(x)})
    return pd.DataFrame(rows, columns=["frame", "y", "x"])


def link_tracks(det, search_range_px: float = 12.0, memory: int = 2, min_len: int = 16):
    """Link detections into tracks; returns a DataFrame with a `particle` column."""
    import trackpy as tp

    if len(det) == 0:
        return det.assign(particle=[])
    tp.quiet()
    linked = tp.link_df(det, search_range=search_range_px, memory=memory)
    counts = linked.groupby("particle")["frame"].count()
    keep = counts[counts >= min_len].index
    return linked[linked["particle"].isin(keep)].copy()


def extract_clips(
    frames: np.ndarray,
    linked,
    crop: int = 96,
    n_frames: int = 16,
    strides: tuple[int, ...] = (1,),
    max_clips: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Cell-centred clips, (n, n_frames, crop, crop) uint8, plus per-clip metadata.

    A clip follows its track: each frame is centred on that track's position in that
    frame, so the cell stays centred and the model cannot use drift as a shortcut.
    """
    rng = rng or np.random.default_rng(0)
    half = crop // 2
    H, W = frames.shape[1], frames.shape[2]
    clips, meta = [], []
    if len(linked) == 0:
        return np.zeros((0, n_frames, crop, crop), np.uint8), meta

    for pid, g in linked.groupby("particle"):
        g = g.sort_values("frame")
        fr = g["frame"].to_numpy()
        ys = g["y"].to_numpy()
        xs = g["x"].to_numpy()
        pos = {int(f): (y, x) for f, y, x in zip(fr, ys, xs, strict=False)}
        for stride in strides:
            span = (n_frames - 1) * stride
            if fr[-1] - fr[0] < span:
                continue
            starts = [f for f in fr if all((f + i * stride) in pos for i in range(n_frames))]
            if not starts:
                continue
            t0 = int(rng.choice(starts))
            buf = np.zeros((n_frames, crop, crop), np.uint8)
            for i in range(n_frames):
                t = t0 + i * stride
                cy, cx = pos[t]
                y0 = int(round(cy)) - half
                x0 = int(round(cx)) - half
                ys0, ys1 = max(0, y0), min(H, y0 + crop)
                xs0, xs1 = max(0, x0), min(W, x0 + crop)
                if ys1 - ys0 < crop * 0.6 or xs1 - xs0 < crop * 0.6:
                    buf = None
                    break
                patch = frames[t, ys0:ys1, xs0:xs1]
                buf[i, (ys0 - y0):(ys0 - y0) + patch.shape[0],
                    (xs0 - x0):(xs0 - x0) + patch.shape[1]] = patch
            if buf is None:
                continue
            clips.append(buf)
            meta.append({"particle": int(pid), "t0": int(t0), "stride": int(stride),
                         "y": float(pos[t0][0]), "x": float(pos[t0][1]),
                         "track_len": int(len(fr))})
            if max_clips and len(clips) >= max_clips:
                return np.stack(clips), meta
    if not clips:
        return np.zeros((0, n_frames, crop, crop), np.uint8), meta
    return np.stack(clips), meta


def resample_factor(um_per_px: float) -> float:
    """Scale factor to reach TARGET_UM_PER_PX (>1 means upsampling)."""
    return um_per_px / TARGET_UM_PER_PX


def maybe_resample(frames: np.ndarray, um_per_px: float, tol: float = 0.05) -> np.ndarray:
    """Resample to the common pixel size when the factor differs by more than tol."""
    f = resample_factor(um_per_px)
    if abs(f - 1.0) <= tol:
        return frames
    from skimage.transform import rescale

    out = rescale(frames, (1.0, f, f), order=1, preserve_range=True,
                  anti_aliasing=(f < 1.0), channel_axis=None)
    return out.astype(frames.dtype)
