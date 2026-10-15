"""BBBC054: LPS-exposed mouse microglia, per-cell phenotype annotations.

Licence CC BY 3.0 Unported (commercial use allowed), verified on the BBBC054 page.
Imaging (from the page): Leica DMI 6000, 20x objective, Andor AND-DG-152V-C1E-FI
camera, Metamorph Premiere, TIFF.

Ground truth CSV columns (verified on the page): index, axis-0 (FOV/image), axis-1 (x),
axis-2 (y), label in {round, amoeboid, stratified}.

FOV -> LPS exposure time: "Each field of view correspond to an exposure time of LPS
with 30 minute intervals between 0 hours and 30 hours" (verified on the page), so
FOV index i maps to i * 0.5 h. The FOV ordering is assumed to follow the axis-0 index;
this is recorded as an assumption in RUN_REPORT.md because the page does not give an
explicit per-file time table.
"""
from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np

LABELS = ("round", "amoeboid", "stratified")
MINUTES_PER_FOV = 30.0


def load_annotations(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            keys = {k.strip().lower(): k for k in r}

            def get(*names):
                for n in names:
                    if n in keys:
                        return r[keys[n]]
                return None

            lab = (get("label") or "").strip().strip('"')
            if lab not in LABELS:
                continue
            try:
                rows.append({
                    "index": int(float(get("index") or 0)),
                    "fov": int(float(get("axis-0", "axis_0"))),
                    "x": float(get("axis-1", "axis_1")),
                    "y": float(get("axis-2", "axis_2")),
                    "label": lab,
                })
            except (TypeError, ValueError):
                continue
    return rows


def list_zip_images(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return sorted(m for m in zf.namelist()
                      if m.lower().endswith((".tif", ".tiff"))
                      and not m.startswith("__MACOSX"))


def read_zip_image(zip_path: Path, member: str) -> np.ndarray:
    import io

    import tifffile

    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(member)
    return tifffile.imread(io.BytesIO(data))


def crop_at(img: np.ndarray, y: float, x: float, size: int = 96) -> np.ndarray | None:
    """Centred crop; returns None if it would fall mostly outside the frame."""
    if img.ndim == 3:
        img = img[..., 0] if img.shape[-1] <= 4 else img[0]
    h, w = img.shape[-2:]
    half = size // 2
    y0, x0 = int(round(y)) - half, int(round(x)) - half
    ys0, ys1 = max(0, y0), min(h, y0 + size)
    xs0, xs1 = max(0, x0), min(w, x0 + size)
    if (ys1 - ys0) < size * 0.6 or (xs1 - xs0) < size * 0.6:
        return None
    out = np.zeros((size, size), dtype=img.dtype)
    out[(ys0 - y0):(ys0 - y0) + (ys1 - ys0), (xs0 - x0):(xs0 - x0) + (xs1 - xs0)] = \
        img[ys0:ys1, xs0:xs1]
    return out


def exposure_hours(fov: int, fov_order: list[int] | None = None) -> float:
    """LPS exposure time in hours for a FOV index."""
    if fov_order is not None:
        return fov_order.index(fov) * MINUTES_PER_FOV / 60.0
    return fov * MINUTES_PER_FOV / 60.0
