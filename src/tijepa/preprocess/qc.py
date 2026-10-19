"""Frame-level quality control for the pretraining corpus (PLAN_v1.md F1.4.4).

The v0 corpus applied no QC at all (review finding D4d): out-of-focus wells, saturated
frames and acquisitions that drift in illumination all became training clips.

The plan's first choice is a classifier trained on S-BIAD2133, which carries frame-level
anomaly labels. That deposit could not be retrieved in full (its file listing is served
5 rows at a time and the images themselves are not reachable in bulk), so this module
implements the plan's documented fallback: three interpretable per-frame measurements
whose thresholds are set from the **empirical distribution of this corpus**, not from
constants invented here.

  * `focus` - variance of the Laplacian. Low values mean a blurred or empty field.
  * `saturation` - fraction of pixels at the extremes of the dtype range.
  * `drift` - absolute change in frame mean relative to the movie's median frame mean,
    which catches illumination steps and bleaching.

Thresholds are quantiles of the pooled distribution, so the fraction removed is known
and reported by construction rather than being an accident of hand-picked cut-offs. A
movie is dropped outright when most of its frames fail, because a handful of good
frames from a bad acquisition is not worth the leakage of its artefacts into training.

This is labelled **heuristic** everywhere it is reported, per the plan.
"""
from __future__ import annotations

import numpy as np

#: quantiles used to set thresholds from the empirical distribution
FOCUS_Q = 0.05        # drop the least-focused 5% of frames
SATURATION_Q = 0.98   # drop frames in the worst 2% for saturation
DRIFT_Q = 0.98        # drop frames in the worst 2% for illumination drift
#: a movie whose failing fraction exceeds this is dropped entirely
MOVIE_FAIL_FRACTION = 0.5


def _laplacian_var(img: np.ndarray) -> float:
    """Variance of a 4-neighbour Laplacian, computed without scipy."""
    a = img.astype(np.float32)
    lap = (-4.0 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1]
           + a[1:-1, :-2] + a[1:-1, 2:])
    return float(lap.var())


def frame_metrics(frames: np.ndarray, sample: int = 24) -> dict:
    """Per-frame QC measurements for one movie, on up to `sample` frames.

    Sampling is uniform over the movie so that a slow drift is still visible; the
    measurements are returned per sampled frame and summarised per movie.
    """
    t = frames.shape[0]
    idx = np.unique(np.linspace(0, t - 1, min(sample, t)).astype(int))
    hi = 255.0 if frames.dtype == np.uint8 else float(frames.max() or 1)
    focus, satur, means = [], [], []
    for i in idx:
        f = frames[i]
        focus.append(_laplacian_var(f))
        satur.append(float(((f <= 0) | (f >= hi)).mean()))
        means.append(float(f.mean()))
    means_arr = np.asarray(means)
    med = float(np.median(means_arr)) if means_arr.size else 0.0
    drift = np.abs(means_arr - med) / (abs(med) + 1e-6)
    return {
        "frame_index": idx.tolist(),
        "focus": [float(x) for x in focus],
        "saturation": satur,
        "drift": [float(x) for x in drift],
        "focus_median": float(np.median(focus)) if focus else 0.0,
        "saturation_max": float(np.max(satur)) if satur else 0.0,
        "drift_max": float(np.max(drift)) if drift.size else 0.0,
    }


def fit_thresholds(all_metrics: list[dict]) -> dict:
    """Set thresholds from the pooled empirical distribution over all movies."""
    focus = np.concatenate([np.asarray(m["focus"]) for m in all_metrics
                            if m.get("focus")]) if all_metrics else np.zeros(1)
    satur = np.concatenate([np.asarray(m["saturation"]) for m in all_metrics
                            if m.get("saturation")]) if all_metrics else np.zeros(1)
    drift = np.concatenate([np.asarray(m["drift"]) for m in all_metrics
                            if m.get("drift")]) if all_metrics else np.zeros(1)
    return {
        "method": "heuristic (S-BIAD2133 classifier not trainable; see RUN_REPORT_v1)",
        "focus_min": float(np.quantile(focus, FOCUS_Q)),
        "saturation_max": float(np.quantile(satur, SATURATION_Q)),
        "drift_max": float(np.quantile(drift, DRIFT_Q)),
        "quantiles": {"focus": FOCUS_Q, "saturation": SATURATION_Q, "drift": DRIFT_Q},
        "movie_fail_fraction": MOVIE_FAIL_FRACTION,
        "n_frames_pooled": int(focus.size),
        "distribution": {
            "focus": {"p1": float(np.quantile(focus, 0.01)),
                      "p50": float(np.quantile(focus, 0.5)),
                      "p99": float(np.quantile(focus, 0.99))},
            "saturation": {"p50": float(np.quantile(satur, 0.5)),
                           "p99": float(np.quantile(satur, 0.99))},
            "drift": {"p50": float(np.quantile(drift, 0.5)),
                      "p99": float(np.quantile(drift, 0.99))},
        },
    }


def judge(metrics: dict, thresholds: dict) -> dict:
    """Apply thresholds to one movie's metrics."""
    focus = np.asarray(metrics.get("focus", []), dtype=float)
    satur = np.asarray(metrics.get("saturation", []), dtype=float)
    drift = np.asarray(metrics.get("drift", []), dtype=float)
    if focus.size == 0:
        return {"pass": False, "reason": "no frames measured", "fail_fraction": 1.0}
    bad = ((focus < thresholds["focus_min"])
           | (satur > thresholds["saturation_max"])
           | (drift > thresholds["drift_max"]))
    frac = float(bad.mean())
    reasons = []
    if (focus < thresholds["focus_min"]).any():
        reasons.append("out of focus")
    if (satur > thresholds["saturation_max"]).any():
        reasons.append("saturated")
    if (drift > thresholds["drift_max"]).any():
        reasons.append("illumination drift")
    return {
        "pass": bool(frac <= thresholds["movie_fail_fraction"]),
        "fail_fraction": frac,
        "reason": ", ".join(reasons) if reasons else "",
    }


def apply_static_augmentation(clips: np.ndarray, rng: np.random.Generator,
                              p: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Replace a fraction `p` of clips by one frame repeated (PLAN_v1.md F1.4.7).

    BBBC054, and any other still-image evaluation, presents the encoder with a clip
    whose frames are all identical. In v0 that input was out of distribution, because
    every training clip moved (review finding C5). Making static clips part of training
    at a known rate removes the mismatch.

    Returns the modified clips and a boolean mask of which were made static.
    """
    if len(clips) == 0:
        return clips, np.zeros(0, dtype=bool)
    mask = rng.random(len(clips)) < p
    out = clips.copy()
    if mask.any():
        idx = np.nonzero(mask)[0]
        # repeat the clip's middle frame, which is the one best centred on the cell
        mid = clips.shape[1] // 2
        out[idx] = clips[idx, mid][:, None, :, :].repeat(clips.shape[1], axis=1)
    return out, mask
