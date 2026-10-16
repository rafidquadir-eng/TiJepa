"""F3.2: BBBC054 morphology-class probe, redesigned (fixes D5a-d, B5, C1, C6).

This is a **morphology-class** task and is never described as an activation-state task
(review finding B5): the labels are microglial shape categories, and shape is not
activation.

What v1 changes:

  * **All three annotated classes are used.** Review finding D5b, and PLAN_v1.md F3.2.1,
    assumed the annotated replicate carries only two of the three classes. It does not:
    Replicate 1 annotates `round` (43,813), `ramified` (11,143) and `amoeboid` (3,230).
    The three-class task is therefore the primary one, with the two-class
    round-versus-amoeboid contrast reported alongside it for comparability with v0.
  * **The FOV-time confound is attacked three ways** (D5c), because in this dataset one
    field of view *is* one exposure time, so any FOV-grouped split leaks time:
      (a) a **time-only baseline** that classifies morphology from exposure time alone -
          whatever it scores is the part of any encoder's score that needs no image;
      (b) **within-time-bin** evaluation: 6 contiguous bins of 10 FOVs, balanced
          accuracy computed inside each bin and averaged, so time cannot explain it;
      (c) the **FOV-identity probe** from v0, predicting which FOV a crop came from.
  * **Exposure-time regression is demoted to exploratory** and uses contiguous block CV
    (leave 6 consecutive FOVs out), because FOV equals time and a shuffled split would
    let the model interpolate between neighbouring times.
  * **Scale is handled** (D5d). The BBBC054 page states a Leica DMI 6000 with a 20x
    objective and an Andor AND-DG-152V-C1E-FI camera. The Andor Neo 5.5 sCMOS
    specification gives a 2560 x 2160 array at 6.5 um pixel pitch; these images are
    1080 x 1280, exactly half in each axis, i.e. 2x2 binning, so the sample-plane pixel
    size is 2 x 6.5 / 20 = **0.65 um/px** - which is already the common target, so the
    resample factor is 1.0. The camera *model code* could not be confirmed by name, so a
    **scale-sensitivity analysis** over factors {0.5, 0.75, 1.0, 1.5} is run as well, as
    F1.5 requires when the pitch cannot be fully verified.
  * **Seeds vary real things** (C1): per seed, a different stratified cell subsample, a
    different `StratifiedGroupKFold` fold assignment, and for E6 a different random-init
    draw. v0's seeds changed nothing and its sd was decorative.
  * **E1 is a proper classical baseline** (C6): segmentation, shape, Zernike moments and
    Haralick texture on the cell mask, probed with both logistic regression and a random
    forest, better reported.

v2 changes (PLAN_v2.md G2.2; findings N6, N15, N1):

  * the full sample: the per-(FOV, class) cap is water-filled so the total reaches
    `--n-cells` (about 9,000 per seed); v1's numbers came from a ~880-cell smoke run;
  * exposure-time regression: robust scaling, features clipped to the 0.1-99.9%
    quantiles of the training fold, non-finite features dropped, and an assertion that
    every MAE is finite and below 10x the mean predictor (v1's E1 MAE was 1e8 h);
  * leave-one-FOV-out with a +/-3-FOV buffer replaces block CV, whose held-out blocks
    regress to a training mean on the other side of the time axis (negative rho for
    every encoder); the within-block rho is reported as well;
  * the confound is described on the full annotation file (class proportions per
    exposure time, Cramer's V), and the time-only baseline is reported for what it is;
  * every learned encoder records the checkpoint it came from (file, run, step,
    SHA-256) and its results key is derived from that record (`tijepa.eval.provenance`).

E5 (frame shuffle) stays **N/A** here: BBBC054 is static, so there is no temporal order
to destroy. It applies only to the tracking and forecasting evaluations.
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression, RidgeCV  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.preprocessing import RobustScaler, StandardScaler  # noqa: E402

from tijepa.eval import provenance as PV  # noqa: E402
from tijepa.io import bbbc054 as B  # noqa: E402
from tijepa.models.video_jepa import VideoJEPA  # noqa: E402
from tijepa.preprocess import morphology_v1 as M1  # noqa: E402

OUT = REPO / "results" / "v2" / "partA"
CROP, NF = 96, 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOFO_BUFFER = 3
SEEDS = (0, 1, 2)
PROBE_PCA = 64
N_TIME_BINS = 6
SCALE_FACTORS = (0.5, 0.75, 1.0, 1.5)

# Derived pixel size; see the module docstring for the derivation and its caveat.
UM_PER_PX = 0.65
UM_PER_PX_PROVENANCE = (
    "Leica DMI 6000 with a 20x objective and an Andor AND-DG-152V-C1E-FI camera "
    "(BBBC054 page). The Andor Neo 5.5 sCMOS specification gives 2560 x 2160 at 6.5 um "
    "pixel pitch; these images are 1080 x 1280 = exactly half in each axis, i.e. 2x2 "
    "binning, so the pixel size is 2 x 6.5 / 20 = 0.65 um/px, equal to the common "
    "target (resample factor 1.0). The camera model code could not be confirmed by "
    "name, so a scale-sensitivity analysis is also reported.")


def build_dataset(n_max: int, seed: int) -> dict:
    """Stratified subsample of annotated crops; the draw depends on `seed` (F3.5)."""
    ann = B.load_annotations(REPO / "data" / "raw" / "bbbc054"
                             / "Replicate1annotation.csv")
    zip_path = REPO / "data" / "raw" / "bbbc054" / "Replicate_1.zip"
    members = B.list_zip_images(zip_path)
    by_idx = {}
    for m in members:
        num = "".join(ch for ch in Path(m).stem.split("_")[-1] if ch.isdigit())
        if num:
            by_idx[int(num)] = m

    rng = np.random.default_rng(1000 + seed)
    by_key: dict[tuple, list] = {}
    for a in ann:
        by_key.setdefault((a["fov"], a["label"]), []).append(a)
    # water-fill: the smallest per-(FOV, class) cap whose total reaches n_max, so rare
    # classes are taken whole and common ones capped (v1 used n_max // n_keys, which
    # undershoots once amoeboid cells run out)
    sizes = np.array([len(v) for v in by_key.values()])
    per_key = 1
    while per_key < sizes.max() and np.minimum(sizes, per_key).sum() < n_max:
        per_key += 1
    chosen = []
    for _k, v in by_key.items():
        idx = rng.permutation(len(v))[:per_key]
        chosen.extend(v[i] for i in idx)

    fov_order = sorted(by_idx)
    crops, labels, fovs, hours = [], [], [], []
    cache: dict[int, np.ndarray] = {}
    for a in sorted(chosen, key=lambda r: r["fov"]):
        f = a["fov"]
        if f not in by_idx:
            continue
        if f not in cache:
            cache.clear()
            cache[f] = B.read_zip_image(zip_path, by_idx[f])
        c = B.crop_at(cache[f], y=a["y"], x=a["x"], size=CROP)
        if c is None:
            continue
        c = c.astype(np.float32)
        c = (c - c.min()) / (c.max() - c.min() + 1e-6)
        crops.append(c)
        labels.append(a["label"])
        fovs.append(f)
        hours.append(B.exposure_hours(f, fov_order))
    X = np.stack(crops)
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    print(f"  seed {seed}: {len(X)} crops, {len(set(fovs))} FOVs, "
          f"classes {dict(zip(uniq, counts, strict=False))}")
    return {"X": X, "labels": labels, "fovs": np.asarray(fovs),
            "hours": np.asarray(hours), "fov_order": fov_order, "per_key_cap": per_key,
            "class_counts": {str(u): int(c) for u, c in zip(uniq, counts, strict=False)}}


def static_clips(X: np.ndarray, scale: float = 1.0) -> torch.Tensor:
    """(N,H,W) -> (N,1,NF,H,W), each frame repeated; optionally rescaled first."""
    if abs(scale - 1.0) > 1e-6:
        from skimage.transform import rescale
        out = []
        for img in X:
            r = rescale(img, scale, order=1, preserve_range=True,
                        anti_aliasing=scale < 1.0)
            # centre-crop or pad back to CROP so the encoder input shape is fixed
            h, w = r.shape
            if h >= CROP:
                y0 = (h - CROP) // 2
                r = r[y0:y0 + CROP, :]
            if r.shape[1] >= CROP:
                x0 = (r.shape[1] - CROP) // 2
                r = r[:, x0:x0 + CROP]
            pad_y, pad_x = CROP - r.shape[0], CROP - r.shape[1]
            if pad_y > 0 or pad_x > 0:
                r = np.pad(r, ((0, max(0, pad_y)), (0, max(0, pad_x))), mode="edge")
            out.append(r.astype(np.float32))
        X = np.stack(out)
    t = torch.from_numpy(X)[:, None, None].repeat(1, 1, NF, 1, 1)
    m = t.mean(dim=(1, 2, 3, 4), keepdim=True)
    s = t.std(dim=(1, 2, 3, 4), keepdim=True)
    return (t - m) / (s + 1e-6)


@torch.no_grad()
def embed_jepa(model: VideoJEPA, clips: torch.Tensor, batch: int = 128) -> np.ndarray:
    outs = []
    for i in range(0, len(clips), batch):
        c = clips[i:i + batch].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            outs.append(model.embed(c).float().cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def embed_timm(name: str, X: np.ndarray, batch: int = 128) -> np.ndarray | None:
    try:
        import timm
    except ImportError:
        return None
    try:
        model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(DEV)
    except Exception as exc:
        print(f"    {name}: unavailable ({type(exc).__name__})")
        return None
    cfg = model.pretrained_cfg
    size = cfg.get("input_size", (3, 224, 224))[1]
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
    outs = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch])[:, None].repeat(1, 3, 1, 1)
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                            align_corners=False)
        x = ((x - mean) / std).to(DEV)
        outs.append(model(x).float().cpu().numpy())
    return np.concatenate(outs)


def probe_pipeline(seed: int, n_features: int, clf):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    steps = [StandardScaler()]
    if n_features > PROBE_PCA:
        steps += [PCA(n_components=PROBE_PCA, random_state=seed), StandardScaler()]
    steps.append(clf)
    return make_pipeline(*steps)


def grouped_predict(x, y, groups, seed: int, kind: str = "logreg"):
    """StratifiedGroupKFold predictions; fold assignment depends on `seed` (F3.5)."""
    pred = np.empty(len(y), dtype=object)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(x, y, groups):
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        clf = (LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                  random_state=seed) if kind == "logreg"
               else RandomForestClassifier(n_estimators=300, random_state=seed,
                                           class_weight="balanced", n_jobs=-1))
        p = probe_pipeline(seed, x.shape[1], clf)
        p.fit(x[tr], y[tr])
        pred[te] = p.predict(x[te])
    return np.asarray(pred)


def within_bin_score(y, pred, hours, n_bins: int = N_TIME_BINS) -> dict:
    """Balanced accuracy computed inside contiguous time bins, then averaged.

    If an encoder's apparent skill is really time information, it vanishes here,
    because within a bin every crop has nearly the same exposure time.
    """
    edges = np.quantile(hours, np.linspace(0, 1, n_bins + 1))
    scores, sizes = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (hours >= lo) & (hours <= hi if i == n_bins - 1 else hours < hi)
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        scores.append(float(balanced_accuracy_score(y[m], pred[m])))
        sizes.append(int(m.sum()))
    return {"n_bins_used": len(scores), "per_bin": scores, "per_bin_n": sizes,
            "mean": float(np.mean(scores)) if scores else None,
            "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else None}


class QuantileClip:
    """Clip each feature to its 0.1-99.9% quantiles on the training fold (G2.2.2)."""

    def __init__(self, lo: float = 0.001, hi: float = 0.999):
        self.lo, self.hi = lo, hi

    def fit(self, x, y=None):
        self.lo_, self.hi_ = np.quantile(x, [self.lo, self.hi], axis=0)
        return self

    def transform(self, x):
        return np.clip(x, self.lo_, self.hi_)

    def fit_transform(self, x, y=None):
        return self.fit(x).transform(x)

    def get_params(self, deep=True):
        return {"lo": self.lo, "hi": self.hi}

    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self


def regression_pipeline(seed: int, n_features: int):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    steps = [QuantileClip(), RobustScaler()]
    if n_features > PROBE_PCA:
        steps += [PCA(n_components=PROBE_PCA, random_state=seed), StandardScaler()]
    steps.append(RidgeCV(alphas=np.logspace(-2, 4, 13)))
    return make_pipeline(*steps)


class RegressionBlowUp(AssertionError):
    pass


def check_mae(mae: float, mae_mean: float, what: str) -> None:
    if not np.isfinite(mae) or mae >= 10 * mae_mean:
        raise RegressionBlowUp(f"{what}: MAE {mae} is not finite or >= 10x the mean "
                               f"predictor ({mae_mean}); features reached the regressor "
                               "unscaled or with outliers")


def lofo_buffered_regression(x, y, fovs, seed: int, buffer: int = LOFO_BUFFER,
                             block: int = 6) -> dict:
    """Exposure time from embeddings, leave-one-FOV-out with a +/-buffer exclusion.

    Each test FOV is bracketed by training FOVs on both sides of the time axis, so its
    prediction cannot be dragged toward a training mean that lies entirely on one side -
    the artefact that made every encoder's block-CV rho negative in v1.
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.array(sorted(set(fovs.tolist())))
    pos = {f: i for i, f in enumerate(order)}
    fpos = np.array([pos[f] for f in fovs])
    pred = np.full(len(y), np.nan)
    pred_mean = np.full(len(y), np.nan)  # the same folds, predicting the training mean
    n_dropped = []
    for i in range(len(order)):
        te = fpos == i
        tr = np.abs(fpos - i) > buffer
        pred_mean[te] = y[tr].mean()
        finite = np.isfinite(x[tr]).all(axis=0)
        n_dropped.append(int((~finite).sum()))
        m = regression_pipeline(seed, int(finite.sum()))
        m.fit(x[tr][:, finite], y[tr])
        pred[te] = m.predict(np.nan_to_num(x[te][:, finite]))
    from scipy import stats as st
    mae = float(np.mean(np.abs(y - pred)))
    mae_mean = float(np.mean(np.abs(y - y.mean())))
    check_mae(mae, mae_mean, "LOFO exposure-time regression")
    mae_cv_mean = float(np.mean(np.abs(y - pred_mean)))
    within = []
    for b in range(0, len(order), block):
        m = (fpos >= b) & (fpos < b + block)
        if m.sum() > 10 and len(np.unique(y[m])) > 2:
            within.append(float(st.spearmanr(y[m], pred[m]).statistic))
    return {
        "status": "ok",
        "cv": f"leave-one-FOV-out with a +/-{buffer}-FOV buffer removed from training, so "
              "each test FOV is bracketed by training FOVs on both sides of the time axis",
        "spearman_rho": float(st.spearmanr(y, pred).statistic),
        "spearman_rho_cv_mean_predictor": float(st.spearmanr(y, pred_mean).statistic),
        "mae_cv_mean_predictor_hours": mae_cv_mean,
        "skill_vs_cv_mean_predictor": 1.0 - mae / mae_cv_mean,
        "reading": "any leave-one-out scheme shifts the training mean away from the held-out "
                   "FOV, so a predictor with no time information is anti-ranked (see "
                   "spearman_rho_cv_mean_predictor). Read rho and MAE against that "
                   "reference under the same folds; skill > 0 means better than it.",
        "within_block_spearman_rho": within,
        "within_block_rho_mean": float(np.mean(within)) if within else None,
        "block_size_fovs": block,
        "mae_hours": mae,
        "mae_mean_predictor_hours": mae_mean,
        "max_nonfinite_features_dropped": int(max(n_dropped)),
        "feature_handling": "0.1-99.9% quantile clipping and RobustScaler fitted on the "
                            "training fold; non-finite features dropped",
        "n": int(len(y)),
    }


def full_file_confound(ann: list[dict]) -> dict:
    """Class proportions per exposure time and Cramer's V on the FULL annotation file."""
    from scipy.stats import chi2_contingency
    fov_order = sorted({a["fov"] for a in ann})
    labels = sorted({a["label"] for a in ann})
    tab = np.zeros((len(fov_order), len(labels)), dtype=int)
    fi = {f: i for i, f in enumerate(fov_order)}
    li = {lab: j for j, lab in enumerate(labels)}
    for a in ann:
        tab[fi[a["fov"]], li[a["label"]]] += 1
    chi2, p, dof, _ = chi2_contingency(tab)
    n = tab.sum()
    v = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))
    props = tab / tab.sum(1, keepdims=True)
    return {
        "n_annotations": int(n), "classes": labels,
        "class_totals": {lab: int(tab[:, j].sum()) for j, lab in enumerate(labels)},
        "per_exposure_time": [
            {"fov": int(f), "hours": float(B.exposure_hours(f, fov_order)),
             "n": int(tab[i].sum()),
             "proportions": {lab: round(float(props[i, j]), 4)
                             for j, lab in enumerate(labels)}}
            for i, f in enumerate(fov_order)],
        "cramers_v_class_vs_time": v, "chi2": float(chi2), "dof": int(dof),
        "p_value": float(p),
        "interpretation": "Cramer's V measures how strongly morphology class depends on "
                          "exposure time in the full annotation file, i.e. the confound "
                          "before any subsampling",
    }


#: per-cell predictions, filled by `register`, keyed (task, seed) -> encoder -> pred
PREDICTIONS: dict[tuple, dict[str, np.ndarray]] = {}
TRUTH: dict[tuple, tuple] = {}
N_BOOT = 2000


def paired_fov_bootstrap(y, fovs, pred_a, pred_b, seed: int = 0,
                         n_boot: int = N_BOOT) -> dict:
    """Paired cluster bootstrap over FOVs of (BA_a - BA_b).

    BBBC054 has 60 FOVs, which is above the 10-group minimum, so a CI is permitted
    here (unlike Part B's 4 wells or tracking's 6 movies). Both encoders are scored on
    the SAME resample, which is what makes the difference paired.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(fovs)
    by_fov = {f: np.flatnonzero(fovs == f) for f in uniq}
    # Integer-code labels once so balanced accuracy can be computed with bincount:
    # sklearn's scorer is far too slow for 2,000 resamples x every encoder pair.
    classes, y_i = np.unique(y, return_inverse=True)
    code = {c: i for i, c in enumerate(classes)}
    a_i = np.array([code.get(v, -1) for v in pred_a])
    b_i = np.array([code.get(v, -1) for v in pred_b])
    n_cls = len(classes)

    def fast_ba(idx: np.ndarray, p_i: np.ndarray) -> float:
        yy = y_i[idx]
        n = np.bincount(yy, minlength=n_cls)
        hit = np.bincount(yy, weights=(p_i[idx] == yy).astype(float),
                          minlength=n_cls)
        present = n > 0
        return float((hit[present] / n[present]).mean())

    all_idx = np.arange(len(y))
    obs = fast_ba(all_idx, a_i) - fast_ba(all_idx, b_i)
    diffs = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by_fov[f] for f in drawn])
        if len(np.unique(y_i[idx])) < 2:
            continue
        diffs.append(fast_ba(idx, a_i) - fast_ba(idx, b_i))
    arr = np.asarray(diffs)
    if arr.size == 0:
        return {"status": "degenerate"}
    lo, hi = np.quantile(arr, [0.025, 0.975])
    # two-sided bootstrap p: how often the resampled difference crosses zero
    p = 2.0 * min(float((arr <= 0).mean()), float((arr >= 0).mean()))
    return {"status": "ok", "difference": float(obs),
            "ci95": [float(lo), float(hi)], "p_value": float(min(1.0, p)),
            "n_resamples": int(arr.size), "n_fovs": int(len(uniq)),
            "method": "paired cluster bootstrap over FOVs, both encoders scored on the "
                      "same resample"}


def bh_adjust(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.asarray(pvals, dtype=float)[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return [float(v) for v in out]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=9000)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--scale-sensitivity", action="store_true")
    ap.add_argument("--encoders", default="",
                    help="comma-separated subset of E1,E2,E3,E0,E4,E4s,E6 (default all)")
    ap.add_argument("--allow-unfinished", action="store_true",
                    help="accept a TiJepa-1 checkpoint that has not reached its planned "
                         "final step (smoke tests only; recorded in the output)")
    ap.add_argument("--out-name", default="bbbc054.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ckpts = {
        "tijepa0": REPO / "checkpoints" / "tijepa0_T1_seed0.pt",
        "tijepa1": Path(args.ckpt) if args.ckpt else
        REPO / "checkpoints" / "tijepa1_T1_seed0.pt",
        "tijepa1_spatial": REPO / "checkpoints" / "tijepa1_spatial_T1_seed0.pt",
    }
    wanted = set(args.encoders.split(",")) if args.encoders else None
    print({k: (v.name if v.exists() else "absent") for k, v in ckpts.items()})

    result: dict = {
        "phase": "G2.2 (F3.2 redesigned)",
        "task": "BBBC054 microglial MORPHOLOGY CLASS (not activation state)",
        "dataset_note": (
            "Replicate 1 annotates all THREE classes - round, ramified and amoeboid - "
            "the three-class task is primary; the two-class round-vs-amoeboid contrast "
            "is also reported."),
        "n_cells_requested_per_seed": args.n_cells,
        "allow_unfinished": args.allow_unfinished,
        "um_per_px": UM_PER_PX,
        "um_per_px_provenance": UM_PER_PX_PROVENANCE,
        "e5_frame_shuffle": "N/A: BBBC054 is static, so there is no temporal order to "
                            "destroy. E5 applies only to tracking and forecasting.",
        "seeds": list(SEEDS),
        "seed_varies": "a different stratified cell subsample, a different "
                       "StratifiedGroupKFold fold assignment, and for E6 a different "
                       "random-init draw (PLAN_v1.md F3.5)",
        "encoders": {},
        "time_only_baseline": {},
    }
    ann_all = B.load_annotations(REPO / "data" / "raw" / "bbbc054"
                                 / "Replicate1annotation.csv")
    result["confound_full_file"] = full_file_confound(ann_all)
    print(f"  full-file Cramer's V (class vs time): "
          f"{result['confound_full_file']['cramers_v_class_vs_time']:.3f}")

    per_seed_data = {s: build_dataset(args.n_cells, s) for s in SEEDS}
    result["n_cells_per_seed"] = {str(s): int(len(d["labels"]))
                                  for s, d in per_seed_data.items()}
    result["class_counts_per_seed"] = {str(s): d["class_counts"]
                                       for s, d in per_seed_data.items()}
    result["per_key_cap"] = {str(s): int(d["per_key_cap"]) for s, d in per_seed_data.items()}

    # ---------------- (a) the time-only baseline: morphology from exposure time alone
    tob = []
    for s in SEEDS:
        d = per_seed_data[s]
        x = d["hours"][:, None]
        for task, y in (("three_class", d["labels"]),
                        ("round_vs_amoeboid", d["labels"])):
            m = (np.isin(d["labels"], ["round", "amoeboid"]) if task.startswith("round")
                 else np.ones(len(d["labels"]), dtype=bool))
            pred = grouped_predict(x[m], y[m], d["fovs"][m], s, kind="rf")
            tob.append({"seed": s, "task": task,
                        "balanced_accuracy": float(
                            balanced_accuracy_score(y[m], pred))})
    for task in ("three_class", "round_vs_amoeboid"):
        v = [r["balanced_accuracy"] for r in tob if r["task"] == task]
        result["time_only_baseline"][task] = {
            "per_seed": v, "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)),
            "interpretation": (
                "balanced accuracy obtainable from exposure time with no image. The "
                "sample takes up to the same number of cells of each class from every "
                "FOV, so where no class runs out every FOV has equal class counts and "
                "time carries no class information BY CONSTRUCTION; any skill above "
                "chance here comes from FOVs where a class ran out. It is therefore not "
                "a measure of the confound in the data, which is described on the full "
                "file in confound_full_file."),
        }
        print(f"  time-only baseline {task}: BA {np.mean(v):.3f} "
              f"+/- {np.std(v, ddof=1):.3f}")

    # ---------------- encoders
    def register(name: str, embed_fn, note: str = "", checkpoint: dict | None = None,
                 per_seed_checkpoint: dict | None = None, probe: str = "logreg") -> None:
        # the key must agree with the checkpoint it came from (PLAN_v2 rule 1)
        PV.label_for_key(name, checkpoint or (per_seed_checkpoint or {}).get(SEEDS[0]))
        rec: dict = {"note": note, "per_seed": [], "tasks": {}, "checkpoint": checkpoint,
                     "probe": probe}
        if per_seed_checkpoint:
            rec["checkpoint_per_seed"] = {str(k): v for k, v in per_seed_checkpoint.items()}
        for s in SEEDS:
            d = per_seed_data[s]
            z = embed_fn(d, s)
            if z is None:
                result["encoders"][name] = {"status": "unavailable", "note": note}
                print(f"  {name}: unavailable")
                return
            z = np.asarray(z, dtype=np.float64)
            # drop non-finite feature columns; v1's nan_to_num turned inf into 1.8e308,
            # which is what produced an exposure-time MAE of 1e8 h
            finite = np.isfinite(z).all(axis=0)
            rec.setdefault("nonfinite_features_dropped", {})[str(s)] = int((~finite).sum())
            z = z[:, finite]
            for task in ("three_class", "round_vs_amoeboid"):
                m = (np.isin(d["labels"], ["round", "amoeboid"])
                     if task.startswith("round") else
                     np.ones(len(d["labels"]), dtype=bool))
                y = d["labels"][m]
                pred = grouped_predict(z[m], y, d["fovs"][m], s, kind=probe)
                rec["tasks"].setdefault(task, []).append({
                    "seed": s,
                    "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                    "macro_f1": float(f1_score(y, pred, average="macro")),
                    "within_time_bin": within_bin_score(y, pred, d["hours"][m]),
                })
                # keep predictions so encoder pairs can be compared on the SAME
                # bootstrap resamples (PLAN_v1.md Section 9)
                PREDICTIONS.setdefault((task, s), {})[name] = pred
                TRUTH[(task, s)] = (y, d["fovs"][m])
            # exposure-time regression (exploratory) and the FOV-identity probe
            rec["per_seed"].append({
                "seed": s,
                "exposure_time_lofo": lofo_buffered_regression(
                    z, d["hours"], d["fovs"], s),
                "fov_identity_balanced_accuracy": float(balanced_accuracy_score(
                    d["fovs"].astype(str),
                    grouped_predict(z, d["fovs"].astype(str),
                                    np.arange(len(z)) % 5, s))),
            })
        for task, rows in rec["tasks"].items():
            ba = np.array([r["balanced_accuracy"] for r in rows])
            wb = [r["within_time_bin"]["mean"] for r in rows
                  if r["within_time_bin"]["mean"] is not None]
            rec[f"{task}_mean"] = float(ba.mean())
            rec[f"{task}_sd"] = float(ba.std(ddof=1))
            rec[f"{task}_within_bin_mean"] = float(np.mean(wb)) if wb else None
            print(f"  {name:28s} {task:18s} BA {ba.mean():.3f} +/- {ba.std(ddof=1):.3f}"
                  f"  within-bin {np.mean(wb):.3f}" if wb else "")
        fov = [r["fov_identity_balanced_accuracy"] for r in rec["per_seed"]]
        rec["fov_identity_mean"] = float(np.mean(fov))
        lofo = [r["exposure_time_lofo"] for r in rec["per_seed"]]
        rec["exposure_time_rho_mean"] = float(np.mean([r["spearman_rho"] for r in lofo]))
        rec["exposure_time_within_block_rho_mean"] = float(np.mean(
            [r["within_block_rho_mean"] for r in lofo
             if r["within_block_rho_mean"] is not None]))
        rec["exposure_time_mae_mean"] = float(np.mean([r["mae_hours"] for r in lofo]))
        rec["exposure_time_mae_mean_predictor"] = float(np.mean(
            [r["mae_mean_predictor_hours"] for r in lofo]))
        rec["exposure_time_rho_cv_mean_predictor"] = float(np.mean(
            [r["spearman_rho_cv_mean_predictor"] for r in lofo]))
        rec["exposure_time_skill_vs_cv_mean"] = float(np.mean(
            [r["skill_vs_cv_mean_predictor"] for r in lofo]))
        result["encoders"][name] = rec

    def want(tag: str) -> bool:
        return wanted is None or tag in wanted

    def load_jepa(run: str):
        path = ckpts[run]
        if not path.exists():
            print(f"  {run}: checkpoint absent, not evaluated")
            return None, None
        sd = torch.load(path, map_location="cpu", weights_only=False)
        rec = PV.checkpoint_record(path, sd)
        if run != "tijepa0" and not rec["final"] and not args.allow_unfinished:
            raise SystemExit(f"{path.name} is at step {rec['step']}, not its planned final "
                             f"step {rec['planned_final_step']}; evaluate final checkpoints "
                             "only (or pass --allow-unfinished for a smoke test)")
        m = VideoJEPA(clip=(NF, CROP, CROP))
        m.load_state_dict(sd["model"])
        return m.eval().to(DEV), rec

    # E1 classical baseline; features computed once per seed, probed two ways
    e1_cache: dict[int, np.ndarray] = {}

    def e1_embed(d: dict, seed: int) -> np.ndarray:
        # on-disk cache keyed by the exact crop set, so the CPU-heavy features can be
        # computed while the GPU is busy and reused by the final run
        import hashlib
        tag = hashlib.sha1(d["X"].tobytes()).hexdigest()[:16]
        cpath = REPO / "data" / "interim" / "bbbc054_e1_cache" / f"seed{seed}_{tag}.npz"
        if seed not in e1_cache and cpath.exists():
            z = np.load(cpath, allow_pickle=False)
            e1_cache[seed] = z["feats"]
            result["e1_segmentation_methods"] = json.loads(str(z["methods"]))
        if seed not in e1_cache:
            feats, methods = [], []
            for img in d["X"]:
                v, _n, meth = M1.features_single_frame(img)
                feats.append(v)
                methods.append(meth)
            uniq, cnt = np.unique(methods, return_counts=True)
            result["e1_segmentation_methods"] = {
                str(u): int(c) for u, c in zip(uniq, cnt, strict=False)}
            e1_cache[seed] = np.stack(feats)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cpath, feats=e1_cache[seed],
                     methods=json.dumps(result["e1_segmentation_methods"]))
        return e1_cache[seed]

    if want("E1"):
        register("E1_handcrafted_logreg", e1_embed,
                 "segmentation + shape + Zernike + Haralick on the mask; logistic-"
                 "regression probe (the probe every encoder gets)")
        register("E1_handcrafted_rf", e1_embed,
                 "the same features with a random-forest probe; both probes are "
                 "reported, neither is selected on the test folds", probe="rf")
    if want("E2"):
        register("E2_imagenet_vit_s",
                 lambda d, s: embed_timm("vit_small_patch16_224.augreg_in21k_ft_in1k",
                                         d["X"]))
    if want("E3"):
        register("E3_dinov2_vit_s",
                 lambda d, s: embed_timm("vit_small_patch14_dinov2.lvd142m", d["X"]))

    model = None
    for tag, run, note in (
            ("E4", "tijepa1", "mixed spatial + future masking"),
            ("E4s", "tijepa1_spatial", "matched-step spatial-masking-only ablation"),
            ("E0", "tijepa0", "historical reference: a different corpus, so not an "
                              "ablation of anything")):
        if not want(tag):
            continue
        m, rec = load_jepa(run)
        if m is None:
            continue
        register(PV.key_for(rec), lambda d, s, m=m: embed_jepa(m, static_clips(d["X"])),
                 f"{note}; checkpoint {rec['file']} at step {rec['step']}", checkpoint=rec)
        if run == "tijepa1":
            model = m
            result["scale_sensitivity_checkpoint"] = rec
        else:
            del m
            torch.cuda.empty_cache()

    # E6: three independent random-init draws, one per seed
    def e6_embed(d: dict, seed: int) -> np.ndarray:
        torch.manual_seed(5000 + seed)
        m = VideoJEPA(clip=(NF, CROP, CROP)).eval().to(DEV)
        return embed_jepa(m, static_clips(d["X"]))

    if want("E6"):
        register("E6_random_init", e6_embed,
                 "three INDEPENDENT random-init draws, one per seed, so the sd reflects "
                 "architecture variability and not just folds",
                 per_seed_checkpoint={s: PV.random_init_record(5000 + s) for s in SEEDS})

    # ---------------- scale sensitivity (TiJepa-1)
    if args.scale_sensitivity and model is not None:
        scales = {}
        d = per_seed_data[SEEDS[0]]
        for f in SCALE_FACTORS:
            z = embed_jepa(model, static_clips(d["X"], scale=f))
            m = np.ones(len(d["labels"]), dtype=bool)
            pred = grouped_predict(z[m], d["labels"][m], d["fovs"][m], SEEDS[0])
            scales[str(f)] = float(balanced_accuracy_score(d["labels"][m], pred))
            print(f"  scale {f}: three-class BA {scales[str(f)]:.3f}")
        result["scale_sensitivity"] = {
            "factors": list(SCALE_FACTORS), "three_class_balanced_accuracy": scales,
            "checkpoint": result["scale_sensitivity_checkpoint"],
            "note": "resample factor 1.0 is the derived 0.65 um/px; reported because "
                    "the camera model code could not be confirmed by name (F1.5)"}

    # ---------------- pairwise encoder comparisons with CIs and BH correction
    comps: dict[str, dict] = {}
    for task in ("three_class", "round_vs_amoeboid"):
        keys = [k for k in PREDICTIONS if k[0] == task]
        if not keys:
            continue
        encs = sorted({e for k in keys for e in PREDICTIONS[k]})
        raw: list[tuple[str, dict]] = []
        for a, b in itertools.combinations(encs, 2):
            per_seed = []
            for k in keys:
                if a not in PREDICTIONS[k] or b not in PREDICTIONS[k]:
                    continue
                y, fovs = TRUTH[k]
                per_seed.append(paired_fov_bootstrap(
                    y, fovs, PREDICTIONS[k][a], PREDICTIONS[k][b], seed=k[1]))
            ok = [r for r in per_seed if r.get("status") == "ok"]
            if not ok:
                continue
            # average the per-seed differences; the CI is the widest across seeds, so
            # seed-to-seed instability cannot be hidden by averaging intervals
            d = float(np.mean([r["difference"] for r in ok]))
            lo = float(min(r["ci95"][0] for r in ok))
            hi = float(max(r["ci95"][1] for r in ok))
            p = float(max(r["p_value"] for r in ok))
            raw.append((f"{a}_vs_{b}", {
                "encoder_a": a, "encoder_b": b, "task": task,
                "difference_mean_over_seeds": d, "ci95_widest_over_seeds": [lo, hi],
                "p_value_max_over_seeds": p, "per_seed": ok}))
        if raw:
            adj = bh_adjust([r[1]["p_value_max_over_seeds"] for r in raw])
            for (nm, rec), q in zip(raw, adj, strict=False):
                rec["p_value_bh"] = q
                comps[f"{task}:{nm}"] = rec
            print(f"  {len(raw)} pairwise comparisons for {task} "
                  f"(BH-corrected within the task)")
    result["pairwise_comparisons"] = comps

    problems = PV.check_results(result["encoders"])
    if problems:
        raise SystemExit("provenance check failed: " + "; ".join(problems))
    (OUT / args.out_name).write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"wrote {OUT / args.out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
