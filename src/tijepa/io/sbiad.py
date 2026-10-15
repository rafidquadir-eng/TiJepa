"""S-BIAD2196 "CellTrack": human neutrophil brightfield movies with tracked masks.

Licence CC BY 4.0 (verified from the BioStudies record). Pixel size 0.641 x 0.641 um,
time interval 8 s (verified from the record's per-file metadata).

Layout (OME-Zarr v2 inside a zip, one level of nesting):
    <stem>.ome.zarr/0                        raw, (t, y, x) float64
    <stem>.ome.zarr/labels/<name>/0          instance IDs, (t, y, x) int32,
                                             IDs consistent across time

The label group name is NOT consistent across the deposit: four of the five movies use
"Tracked Cells" and 01_A3 uses "TrackedCells", so it is discovered at read time rather
than hard-coded.
This dataset is used for EVALUATION ONLY in this run and is never pretrained on.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

UM_PER_PX = 0.641
FRAME_INTERVAL_S = 8.0


def _open(path: Path):
    import zarr

    store = zarr.storage.ZipStore(str(path), mode="r")
    with zipfile.ZipFile(path) as zf:
        root = zf.namelist()[0].split("/")[0]
    return zarr.open_group(store, mode="r", path=root, zarr_format=2), root


def label_group_name(g) -> str:
    """The deposit spells this inconsistently ("Tracked Cells" vs "TrackedCells")."""
    names = list(g["labels"].keys())
    for n in names:
        if n.replace(" ", "").lower() == "trackedcells":
            return n
    if not names:
        raise KeyError("no label group found")
    return names[0]


def load_movie(path: Path, max_frames: int | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw uint8 (T,Y,X), labels int32 (T,Y,X))."""
    g, _ = _open(path)
    raw = g["0"]
    lab = g["labels"][label_group_name(g)]["0"]
    t = raw.shape[0] if max_frames is None else min(max_frames, raw.shape[0])
    r = np.asarray(raw[:t])
    lo, hi = np.percentile(r, [0.5, 99.5])
    r8 = np.clip((r - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return r8, np.asarray(lab[:t], dtype=np.int32)


def tracks_from_labels(labels: np.ndarray, min_len: int = 16) -> dict[int, np.ndarray]:
    """Instance IDs -> (n_frames_present, 3) arrays of (t, y, x) centroids."""
    out: dict[int, list] = {}
    for t in range(labels.shape[0]):
        frame = labels[t]
        ids = np.unique(frame)
        for i in ids:
            if i == 0:
                continue
            ys, xs = np.nonzero(frame == i)
            if ys.size == 0:
                continue
            out.setdefault(int(i), []).append((t, float(ys.mean()), float(xs.mean())))
    return {k: np.asarray(v) for k, v in out.items() if len(v) >= min_len}
