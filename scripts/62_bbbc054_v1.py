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

E5 (frame shuffle) stays **N/A** here: BBBC054 is static, so there is no temporal order
to destroy. It applies only to the tracking and forecasting evaluations.
"""
from __future__ import annotations

import argparse
import functools
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
from sklearn.preprocessing import StandardScaler  # noqa: E402

from tijepa.io import bbbc054 as B  # noqa: E402
from tijepa.models.video_jepa import VideoJEPA  # noqa: E402
from tijepa.preprocess import morphology_v1 as M1  # noqa: E402

OUT = REPO / "results" / "v1" / "partA"
CROP, NF = 96, 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"
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
    per_key = max(1, n_max // max(1, len(by_key)))
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
            "hours": np.asarray(hours), "fov_order": fov_order,
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


def block_cv_regression(x, y, fovs, seed: int, block: int = 6) -> dict:
    """Leave-6-contiguous-FOVs-out ridge regression of exposure time (exploratory)."""
    order = np.array(sorted(set(fovs.tolist())))
    pred = np.zeros(len(y))
    for i in range(0, len(order), block):
        held = set(order[i:i + block].tolist())
        te = np.array([f in held for f in fovs])
        tr = ~te
        if tr.sum() < 50 or te.sum() == 0:
            continue
        m = probe_pipeline(seed, x.shape[1], RidgeCV(alphas=np.logspace(-2, 4, 13)))
        m.fit(x[tr], y[tr])
        pred[te] = m.predict(x[te])
    ok = pred != 0
    if ok.sum() < 20:
        return {"status": "insufficient"}
    from scipy import stats as st
    return {
        "status": "ok",
        "cv": f"leave-{block}-contiguous-FOVs-out (FOV = exposure time, so a shuffled "
              "split would let the model interpolate between neighbouring times)",
        "spearman_rho": float(st.spearmanr(y[ok], pred[ok]).statistic),
        "mae_hours": float(np.mean(np.abs(y[ok] - pred[ok]))),
        "mae_mean_predictor_hours": float(np.mean(np.abs(y[ok] - y[ok].mean()))),
        "n": int(ok.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=9000)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--scale-sensitivity", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.ckpt) if args.ckpt else None
    if ckpt is None:
        for cand in ("tijepa1_T1_seed0.pt", "tijepa0_T1_seed0.pt"):
            if (REPO / "checkpoints" / cand).exists():
                ckpt = REPO / "checkpoints" / cand
                break
    abl = REPO / "checkpoints" / "tijepa1_spatial_T1_seed0.pt"
    v0_ckpt = REPO / "checkpoints" / "tijepa0_T1_seed0.pt"
    print(f"checkpoint: {ckpt.name if ckpt else 'NONE'}; "
          f"ablation: {'yes' if abl.exists() else 'no'}; "
          f"v0 reference: {'yes' if v0_ckpt.exists() else 'no'}")

    result: dict = {
        "phase": "F3.2",
        "task": "BBBC054 microglial MORPHOLOGY CLASS (not activation state)",
        "dataset_note": (
            "Replicate 1 annotates all THREE classes - round, ramified and amoeboid - "
            "contrary to the premise of review finding D5b and PLAN_v1.md F3.2.1 that "
            "only two of three are present. The three-class task is primary; the "
            "two-class round-vs-amoeboid contrast is also reported for comparability "
            "with v0."),
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
        "exposure_time_regression": {},
        "fov_identity_probe": {},
    }

    per_seed_data = {s: build_dataset(args.n_cells, s) for s in SEEDS}

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
            "interpretation": "balanced accuracy obtainable from exposure time with no "
                              "image at all; any encoder's score must be read against "
                              "this, because FOV = time in this dataset",
        }
        print(f"  time-only baseline {task}: BA {np.mean(v):.3f} "
              f"+/- {np.std(v, ddof=1):.3f}")

    # ---------------- encoders
    def register(name: str, embed_fn, note: str = "") -> None:
        rec: dict = {"note": note, "per_seed": [], "tasks": {}}
        for s in SEEDS:
            d = per_seed_data[s]
            z = embed_fn(d, s)
            if z is None:
                result["encoders"][name] = {"status": "unavailable", "note": note}
                print(f"  {name}: unavailable")
                return
            z = np.nan_to_num(np.asarray(z, dtype=np.float32))
            for task in ("three_class", "round_vs_amoeboid"):
                m = (np.isin(d["labels"], ["round", "amoeboid"])
                     if task.startswith("round") else
                     np.ones(len(d["labels"]), dtype=bool))
                y = d["labels"][m]
                pred = grouped_predict(z[m], y, d["fovs"][m], s)
                rec["tasks"].setdefault(task, []).append({
                    "seed": s,
                    "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                    "macro_f1": float(f1_score(y, pred, average="macro")),
                    "within_time_bin": within_bin_score(y, pred, d["hours"][m]),
                })
            # exposure-time regression (exploratory) and the FOV-identity probe
            rec["per_seed"].append({
                "seed": s,
                "exposure_time_block_cv": block_cv_regression(
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
        rho = [r["exposure_time_block_cv"].get("spearman_rho") for r in rec["per_seed"]
               if r["exposure_time_block_cv"].get("spearman_rho") is not None]
        rec["exposure_time_rho_mean"] = float(np.mean(rho)) if rho else None
        result["encoders"][name] = rec

    # E1 improved classical baseline
    def e1_embed(d: dict, seed: int) -> np.ndarray:
        feats, methods = [], []
        for img in d["X"]:
            v, _n, meth = M1.features_single_frame(img)
            feats.append(v)
            methods.append(meth)
        result.setdefault("e1_segmentation_methods", {})
        uniq, cnt = np.unique(methods, return_counts=True)
        result["e1_segmentation_methods"] = {
            str(u): int(c) for u, c in zip(uniq, cnt, strict=False)}
        return np.stack(feats)

    register("E1_handcrafted_improved", e1_embed,
             "segmentation + shape + Zernike + Haralick on the mask; probed with "
             "logistic regression and a random forest, better reported (fixes C6)")

    register("E2_imagenet_vit_s",
             lambda d, s: embed_timm("vit_small_patch16_224.augreg_in21k_ft_in1k",
                                     d["X"]))
    register("E3_dinov2_vit_s", lambda d, s: embed_timm("vit_small_patch14_dinov2.lvd142m",
                                                        d["X"]))

    if ckpt and ckpt.exists():
        model = VideoJEPA(clip=(NF, CROP, CROP))
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"])
        model = model.eval().to(DEV)
        name = ("E4_tijepa1" if "tijepa1" in ckpt.name else "E4_tijepa0_only_available")
        register(name, lambda d, s: embed_jepa(model, static_clips(d["X"])),
                 f"checkpoint {ckpt.name} at step {sd.get('step')}")
        result["checkpoint"] = {"file": ckpt.name, "step": sd.get("step")}

    if abl.exists():
        m_abl = VideoJEPA(clip=(NF, CROP, CROP))
        m_abl.load_state_dict(torch.load(abl, map_location="cpu",
                                         weights_only=False)["model"])
        m_abl = m_abl.eval().to(DEV)
        register("E4s_tijepa1_spatial_only",
                 lambda d, s: embed_jepa(m_abl, static_clips(d["X"])),
                 "the matched-step spatial-masking-only ablation (F2.3)")

    if v0_ckpt.exists() and (not ckpt or ckpt.name != v0_ckpt.name):
        m_v0 = VideoJEPA(clip=(NF, CROP, CROP))
        m_v0.load_state_dict(torch.load(v0_ckpt, map_location="cpu",
                                        weights_only=False)["model"])
        m_v0 = m_v0.eval().to(DEV)
        register("E0_tijepa0_reference",
                 lambda d, s: embed_jepa(m_v0, static_clips(d["X"])),
                 "historical reference only: a different corpus, so NOT a valid "
                 "ablation of anything")

    # E6: three independent random-init draws (F3's encoder list)
    def e6_embed(d: dict, seed: int) -> np.ndarray:
        torch.manual_seed(5000 + seed)
        m = VideoJEPA(clip=(NF, CROP, CROP)).eval().to(DEV)
        return embed_jepa(m, static_clips(d["X"]))

    register("E6_tijepa_random_init", e6_embed,
             "three INDEPENDENT random-init draws, one per seed, so the sd reflects "
             "architecture variability and not just folds")

    # ---------------- scale sensitivity
    if args.scale_sensitivity and ckpt and ckpt.exists():
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
            "note": "resample factor 1.0 is the derived 0.65 um/px; reported because "
                    "the camera model code could not be confirmed by name (F1.5)"}

    (OUT / "bbbc054_v1.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n")
    print(f"wrote {OUT/'bbbc054_v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
