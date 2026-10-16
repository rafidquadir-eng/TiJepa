"""BBBC054: LPS-exposed mouse microglia, per-cell phenotype annotations.

Licence CC BY 3.0 Unported (commercial use allowed), verified on the BBBC054 page.
Imaging (from the page): Leica DMI 6000, 20x objective, Andor AND-DG-152V-C1E-FI
camera, Metamorph Premiere, TIFF.

Ground truth CSV columns: index, axis-0 (FOV/image), axis-1, axis-2, label in
{round, amoeboid, stratified}.

**Axis convention.** The BBBC054 page says "axis-1 indicates the x-coordinate, and
axis-2 indicates the y-coordinate". The data contradict that, and the column names are
numpy-style (axis-0 = image, axis-1 = rows, axis-2 = columns). In Replicate 1 the images
are 1080 rows x 1280 columns, and across all 47,043 annotations axis-1 spans -4..1080
while axis-2 spans -6..1284. axis-1 therefore indexes ROWS (y) and axis-2 indexes
COLUMNS (x). We follow the data, not the page; using the page's wording transposes every
crop and drives every probe to chance.

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

# The three classes actually annotated in Replicate 1, with their counts:
# round (43,813), ramified (11,143), amoeboid (3,230).
#
# v0 had ("round", "amoeboid", "stratified"). "stratified" does not occur anywhere in
# the annotation file, so every `ramified` row -- 11,143 of 58,186, about one annotation
# in five -- was silently dropped by the `lab not in LABELS` filter below, and v0 ran a
# two-class task while believing the third class was simply absent from this replicate.
# That is the true content of review finding D5b: not a missing class, a wrong name.
LABELS = ("round", "amoeboid", "ramified")
MINUTES_PER_FOV = 30.0


def load_annotations(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            keys = {k.strip().lower(): k for k in r}

            def get(*names, r=r, keys=keys):
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
                    # axis-1 = row = y, axis-2 = column = x (see module docstring)
                    "y": float(get("axis-1", "axis_1")),
                    "x": float(get("axis-2", "axis_2")),
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
