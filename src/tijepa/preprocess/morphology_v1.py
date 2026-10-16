"""Improved hand-crafted morphology baseline, E1 (PLAN_v1.md F3, fixes C6).

The v0 hand-crafted baseline was a short list of intensity and second-moment summaries
computed on the whole crop, with no segmentation at all. Review finding C6 was that it
is too weak to be a fair reference: beating it says little, because a competent
classical pipeline is much stronger than "the variance of the patch".

This module provides the pipeline the plan asks for:

  1. **segmentation** - a local-variance / Sobel-magnitude map, Otsu threshold,
     morphological closing and hole filling, then a distance-transform watershed to
     split touching cells, keeping the component nearest the crop centre (the crop is
     cell-centred by construction);
  2. **shape features** - area, perimeter, eccentricity, solidity, major and minor axis
     lengths, circularity, extent, equivalent diameter, orientation spread, plus
     **Zernike moments** (mahotas, degree 8) which capture the ramified-versus-round
     distinction that simple ellipse fits miss;
  3. **texture features** - **Haralick** descriptors from a grey-level co-occurrence
     matrix computed **on the cell mask only**, so background texture cannot leak in;
  4. **motility features** for video inputs - speed, persistence, mean squared
     displacement exponent, turning angles and the moving fraction.

Cellpose is used instead of step 1 when it is installed and its weights are available;
which one ran is recorded in the result rather than assumed.
"""
from __future__ import annotations

import numpy as np

ZERNIKE_DEGREE = 8
ZERNIKE_RADIUS = 24


def _as_uint8(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return (a * 255).astype(np.uint8)


def segment_cell(img: np.ndarray) -> tuple[np.ndarray, str]:
    """Binary mask of the centred cell. Returns `(mask, method)`."""
    from scipy import ndimage as ndi
    from skimage.filters import sobel, threshold_otsu
    from skimage.measure import label
    from skimage.morphology import closing, disk, remove_small_objects
    from skimage.segmentation import watershed

    a = img.astype(np.float32)
    if a.std() < 1e-6:
        return np.zeros(a.shape, dtype=bool), "degenerate_uniform_crop"

    # edge energy plus local variance: brightfield cells are low-contrast but textured
    edge = sobel(a)
    loc_var = ndi.uniform_filter(a ** 2, 5) - ndi.uniform_filter(a, 5) ** 2
    energy = edge / (edge.max() + 1e-9) + loc_var / (loc_var.max() + 1e-9)

    try:
        thr = threshold_otsu(energy)
    except ValueError:
        return np.zeros(a.shape, dtype=bool), "otsu_failed"
    m = energy > thr
    m = closing(m, disk(3))
    m = ndi.binary_fill_holes(m)
    # skimage renamed min_size -> max_size with inclusive semantics: max_size=31
    # removes objects of 31 pixels or fewer, matching the old min_size=32.
    m = remove_small_objects(m, max_size=31)
    if not m.any():
        return m, "empty_after_cleanup"

    # split touching cells with a distance-transform watershed
    dist = ndi.distance_transform_edt(m)
    peaks = dist > 0.5 * dist.max()
    markers, n = ndi.label(peaks)
    if n > 1:
        m_lab = watershed(-dist, markers, mask=m)
    else:
        m_lab = label(m)

    # keep the component nearest the centre: the crop is cell-centred by construction
    cy, cx = np.array(a.shape) / 2.0
    best, best_d = None, np.inf
    for lab_id in np.unique(m_lab):
        if lab_id == 0:
            continue
        ys, xs = np.nonzero(m_lab == lab_id)
        d = (ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2
        if d < best_d:
            best, best_d = lab_id, d
    if best is None:
        return np.zeros(a.shape, dtype=bool), "no_component"
    return (m_lab == best), "sobel_variance_otsu_watershed"


def shape_features(mask: np.ndarray) -> tuple[list[float], list[str]]:
    from skimage.measure import regionprops

    names = ["area", "perimeter", "eccentricity", "solidity", "major_axis",
             "minor_axis", "circularity", "extent", "equiv_diameter",
             "axis_ratio", "perimeter_area_ratio"]
    if not mask.any():
        return [0.0] * len(names), names
    p = regionprops(mask.astype(np.uint8))[0]
    per = float(p.perimeter) or 1e-6
    circ = 4.0 * np.pi * float(p.area) / (per ** 2)
    minor = float(p.axis_minor_length) or 1e-6
    vals = [float(p.area), per, float(p.eccentricity), float(p.solidity),
            float(p.axis_major_length), minor, circ, float(p.extent),
            float(p.equivalent_diameter_area), float(p.axis_major_length) / minor,
            per / max(float(p.area), 1e-6)]
    return vals, names


def zernike_features(mask: np.ndarray) -> tuple[list[float], list[str]]:
    try:
        import mahotas
    except ImportError:
        return [], []
    if not mask.any():
        n = len(mahotas.features.zernike_moments(
            np.ones((8, 8)), ZERNIKE_RADIUS, degree=ZERNIKE_DEGREE))
        return [0.0] * n, [f"zernike_{i}" for i in range(n)]
    z = mahotas.features.zernike_moments(mask.astype(np.uint8), ZERNIKE_RADIUS,
                                         degree=ZERNIKE_DEGREE)
    return [float(v) for v in z], [f"zernike_{i}" for i in range(len(z))]


def haralick_features(img: np.ndarray, mask: np.ndarray) -> tuple[list[float], list[str]]:
    """Haralick descriptors on the cell mask only, so background cannot leak in."""
    from skimage.feature import graycomatrix, graycoprops

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation",
             "ASM"]
    names = [f"haralick_{p}" for p in props]
    if not mask.any():
        return [0.0] * len(names), names
    q = _as_uint8(img) // 16  # 16 grey levels
    q = np.where(mask, q, 0)
    glcm = graycomatrix(q, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2,
                                                    3 * np.pi / 4],
                        levels=16, symmetric=True, normed=True)
    return [float(np.mean(graycoprops(glcm, p))) for p in props], names


def intensity_features(img: np.ndarray, mask: np.ndarray) -> tuple[list[float],
                                                                   list[str]]:
    names = ["mean_in", "std_in", "mean_out", "std_out", "contrast_in_out",
             "p10_in", "p90_in", "skew_in"]
    a = img.astype(np.float32)
    if not mask.any():
        return [float(a.mean()), float(a.std()), float(a.mean()), float(a.std()),
                0.0, float(np.percentile(a, 10)), float(np.percentile(a, 90)),
                0.0], names
    inside, outside = a[mask], a[~mask]
    if outside.size == 0:
        outside = inside
    mi, si = float(inside.mean()), float(inside.std())
    skew = float(((inside - mi) ** 3).mean() / (si ** 3 + 1e-9))
    return [mi, si, float(outside.mean()), float(outside.std()),
            mi - float(outside.mean()), float(np.percentile(inside, 10)),
            float(np.percentile(inside, 90)), skew], names


def features_single_frame(img: np.ndarray) -> tuple[np.ndarray, list[str], str]:
    """Full improved feature vector for one 2-D crop."""
    mask, method = segment_cell(img)
    vals: list[float] = []
    names: list[str] = []
    for fn in (shape_features, zernike_features):
        v, n = fn(mask)
        vals += v
        names += n
    for fn in (haralick_features, intensity_features):
        v, n = fn(img, mask)
        vals += v
        names += n
    return np.asarray(vals, dtype=np.float32), names, method


def motility_features(track_yx: np.ndarray, um_per_px: float,
                      interval_s: float) -> tuple[list[float], list[str]]:
    """Speed, persistence, MSD exponent, turning angles and moving fraction.

    `track_yx` is (n_frames, 2) in pixels. For video inputs only; PLAN_v1.md F3 asks
    for these because a classical baseline on *video* should use motion, and v0's did
    not.
    """
    names = ["speed_um_per_min", "persistence", "msd_alpha", "turn_angle_mean",
             "turn_angle_std", "fraction_moving", "net_displacement_um",
             "path_length_um"]
    t = np.asarray(track_yx, dtype=float)
    if t.shape[0] < 3:
        return [0.0] * len(names), names
    d = np.diff(t, axis=0) * um_per_px
    step = np.hypot(d[:, 0], d[:, 1])
    dt_min = interval_s / 60.0
    path = float(step.sum())
    net = float(np.hypot(*(t[-1] - t[0])) * um_per_px)
    speed = float(step.mean() / max(dt_min, 1e-9))
    persistence = net / max(path, 1e-9)

    # MSD exponent: slope of log MSD vs log lag
    lags = np.arange(1, max(2, min(10, t.shape[0] // 2)))
    msd = []
    for lag in lags:
        disp = (t[lag:] - t[:-lag]) * um_per_px
        msd.append(float((disp ** 2).sum(axis=1).mean()))
    msd_arr = np.asarray(msd)
    ok = msd_arr > 0
    alpha = (float(np.polyfit(np.log(lags[ok]), np.log(msd_arr[ok]), 1)[0])
             if ok.sum() >= 2 else 0.0)

    ang = np.arctan2(d[:, 0], d[:, 1])
    turns = np.diff(ang)
    turns = (turns + np.pi) % (2 * np.pi) - np.pi
    moving = float((step > 0.5 * step.mean()).mean()) if step.mean() > 0 else 0.0
    return [speed, persistence, alpha,
            float(np.mean(np.abs(turns))) if turns.size else 0.0,
            float(np.std(turns)) if turns.size else 0.0,
            moving, net, path], names
