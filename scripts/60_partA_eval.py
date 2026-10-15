"""M6 / Part A evaluation on BBBC054 (PLAN.md A4.2) with the six encoders of A4.

BBBC054 is a STATIC dataset: each annotated cell is a single frame, so a clip is built
by repeating that frame 16 times. Two consequences, both recorded in the paper:
  * E5 (frame-shuffled) is undefined here -- shuffling identical frames is a no-op --
    so it is reported as N/A rather than as a number.
  * The headline polarization benchmark of PLAN.md A4.1 (Dryad BMDM M0/M1/M2 movies)
    could not be downloaded (Anubis proof-of-work challenge), so no public dataset in
    this run provides VIDEO of innate immune cells with activation-state labels.

Tasks: (1) morphology class probe, FOV-grouped CV; (2) ridge regression of LPS
exposure time, FOV-grouped CV. Plus a FOV-identity confound probe.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.eval import probes as P  # noqa: E402
from tijepa.eval.stats import (  # noqa: E402
    classification_metrics,
    cluster_bootstrap_ci,
    paired_bootstrap,
    permutation_p,
)
from tijepa.io import bbbc054 as B  # noqa: E402
from tijepa.models.video_jepa import VideoJEPA  # noqa: E402

OUT = REPO / "results" / "partA"
SEEDS = [0, 1, 2]
CROP, NF = 96, 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataset(n_max: int, seed: int = 0):
    ann = B.load_annotations(REPO / "data" / "raw" / "bbbc054" / "Replicate1annotation.csv")
    zip_path = REPO / "data" / "raw" / "bbbc054" / "Replicate_1.zip"
    members = B.list_zip_images(zip_path)
    # map FOV index -> member via the number in the filename
    by_idx = {}
    for m in members:
        stem = Path(m).stem
        num = "".join(ch for ch in stem.split("_")[-1] if ch.isdigit())
        if num:
            by_idx[int(num)] = m

    rng = np.random.default_rng(seed)
    # stratify the subsample by (fov, label) so every FOV and class is represented
    by_key: dict[tuple, list] = {}
    for a in ann:
        by_key.setdefault((a["fov"], a["label"]), []).append(a)
    per_key = max(1, n_max // max(1, len(by_key)))
    chosen = []
    for _k, v in by_key.items():
        idx = rng.permutation(len(v))[:per_key]
        chosen.extend(v[i] for i in idx)
    rng.shuffle(chosen)
    print(f"BBBC054: {len(ann)} annotations -> {len(chosen)} sampled "
          f"({len(by_key)} fov x label strata)")

    crops, labels, fovs, hours, feats = [], [], [], [], []
    fov_order = sorted(by_idx)
    cache: dict[int, np.ndarray] = {}
    for a in sorted(chosen, key=lambda r: r["fov"]):
        f = a["fov"]
        if f not in by_idx:
            continue
        if f not in cache:
            cache.clear()
            cache[f] = B.read_zip_image(zip_path, by_idx[f])
        # page states axis-1 = x, axis-2 = y
        c = B.crop_at(cache[f], y=a["y"], x=a["x"], size=CROP)
        if c is None:
            continue
        c = c.astype(np.float32)
        c = (c - c.min()) / (c.max() - c.min() + 1e-6)
        crops.append(c)
        labels.append(a["label"])
        fovs.append(f)
        hours.append(B.exposure_hours(f, fov_order))
        mf, mnames = P.morphology_features(c)
        feats.append(mf)
    X = np.stack(crops)
    print(f"BBBC054: {len(X)} usable crops, {len(set(fovs))} FOVs, "
          f"labels {dict(zip(*np.unique(labels, return_counts=True), strict=False))}")
    return (X, np.asarray(labels), np.asarray(fovs), np.asarray(hours),
            np.stack(feats), mnames)


def static_clips(X: np.ndarray) -> torch.Tensor:
    """(N,H,W) -> (N,1,NF,H,W) by repeating each frame; standardised per clip."""
    t = torch.from_numpy(X)[:, None, None].repeat(1, 1, NF, 1, 1)
    m = t.mean(dim=(1, 2, 3, 4), keepdim=True)
    s = t.std(dim=(1, 2, 3, 4), keepdim=True)
    return (t - m) / (s + 1e-6)


def load_tijepa(ckpt: Path | None, random_init: bool) -> VideoJEPA:
    model = VideoJEPA(clip=(NF, CROP, CROP))
    if not random_init and ckpt and ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"  loaded {ckpt.name} (step {sd.get('step')})")
    return model.eval().to(DEV)


# Identical probe for every encoder (PLAN.md A4). A PCA step is included so the 384-d
# encoders and the 14-d hand-crafted baseline go through the same pipeline, and so the
# 200 group-permutation refits are tractable. Applied identically to all encoders and
# fitted inside each CV fold; recorded as a deviation in RUN_REPORT.md.
PROBE_PCA = 64


def _probe(seed: int, n_features: int, classifier):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [StandardScaler()]
    if n_features > PROBE_PCA:
        steps.append(PCA(n_components=PROBE_PCA, random_state=seed))
        steps.append(StandardScaler())
    steps.append(classifier)
    return make_pipeline(*steps)


def grouped_logreg(x, y, groups, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    classes = np.unique(y)
    pred = np.empty(len(y), dtype=y.dtype)
    score = np.zeros((len(y), len(classes)))
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups):
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        clf = _probe(seed, x.shape[1],
                     LogisticRegression(C=1.0, max_iter=2000,
                                        class_weight="balanced", random_state=seed))
        clf.fit(x[tr], y[tr])
        pred[te] = clf.predict(x[te])
        pr = clf.predict_proba(x[te])
        for j, c in enumerate(clf.classes_):
            score[te, list(classes).index(c)] = pr[:, j]
    return pred, score


def grouped_mlp(x, y, groups, seed: int):
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier

    pred = np.empty(len(y), dtype=y.dtype)
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups):
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        clf = _probe(seed, x.shape[1],
                     MLPClassifier(hidden_layer_sizes=(128,), max_iter=400,
                                   random_state=seed, early_stopping=True))
        clf.fit(x[tr], y[tr])
        pred[te] = clf.predict(x[te])
    return pred


def grouped_ridge(x, y, groups, seed: int):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold

    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups):
        m = _probe(seed, x.shape[1], RidgeCV(alphas=np.logspace(-2, 4, 13)))
        m.fit(x[tr], y[tr])
        pred[te] = m.predict(x[te])
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=9000)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    X, labels, fovs, hours, e1, e1names = build_dataset(args.n_cells)
    clips = static_clips(X)
    ckpt = Path(args.ckpt) if args.ckpt else next(
        iter(sorted((REPO / "checkpoints").glob("tijepa0_T1_seed*.pt"))), None)

    embeddings: dict[str, np.ndarray | None] = {"E1_handcrafted_morphology": e1}
    model_licences = {}

    for key, timm_name in (("E2_imagenet_vit_s", "vit_small_patch16_224.augreg_in21k_ft_in1k"),
                           ("E3_dinov2_vit_s", "vit_small_patch14_dinov2.lvd142m")):
        try:
            t0 = time.time()
            m, prep, lic = P.load_timm_encoder(timm_name, DEV)
            # frames are identical, so one frame equals the frame-mean (documented)
            one = clips[:, :, :1]
            embeddings[key] = P.embed_frames_mean(m, prep, one, DEV)
            model_licences[key] = {"timm_model": timm_name, "licence": lic}
            print(f"  {key}: {embeddings[key].shape} in {time.time()-t0:.0f}s")
            del m
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  {key} FAILED: {type(exc).__name__}: {exc}")
            embeddings[key] = None
            model_licences[key] = {"timm_model": timm_name,
                                   "status": f"unavailable: {type(exc).__name__}"}

    if ckpt and ckpt.exists():
        mt = load_tijepa(ckpt, random_init=False)
        embeddings["E4_tijepa0"] = P.embed_tijepa(mt, clips, DEV)
        del mt
        torch.cuda.empty_cache()
    else:
        embeddings["E4_tijepa0"] = None
        print("  E4 skipped: no checkpoint")
    embeddings["E5_tijepa0_frame_shuffled"] = None  # undefined on static clips
    mr = load_tijepa(None, random_init=True)
    embeddings["E6_tijepa0_random_init"] = P.embed_tijepa(mr, clips, DEV)
    del mr
    torch.cuda.empty_cache()

    results: dict = {
        "dataset": "BBBC054 Replicate 1",
        "licence": "CC BY 3.0 Unported",
        "n_cells": int(len(X)), "n_fovs": int(len(set(fovs))),
        "crop_px": CROP, "clip_frames": NF,
        "static_note": "each annotated cell is a single frame repeated to NF frames; "
                       "E5 (frame-shuffle) is undefined and reported as N/A",
        "labels_present": {k: int(v) for k, v in
                           zip(*np.unique(labels, return_counts=True), strict=False)},
        "labels_absent": [l for l in B.LABELS if l not in set(labels)],
        "pixel_size_note": "BBBC054 states a 20x objective but no um/px; crops are in "
                           "native pixels and were NOT resampled to 0.65 um/px. "
                           "Recorded as a deviation.",
        "e1_feature_names": e1names,
        "probe_note": f"identical probe for every encoder: StandardScaler -> PCA("
                      f"{PROBE_PCA}) when dim > {PROBE_PCA} -> StandardScaler -> "
                      f"classifier; PCA fitted inside each CV fold",
        "encoders": {}, "model_licences": model_licences,
        "tasks": {},
    }

    # ---- task 1: morphology class
    task1 = {}
    for key, emb in embeddings.items():
        if emb is None:
            task1[key] = {"status": "N/A" if key.startswith("E5") else "unavailable"}
            continue
        per_seed = []
        for seed in SEEDS:
            pred, score = grouped_logreg(emb, labels, fovs, seed)
            m = classification_metrics(labels, pred, score)
            m["seed"] = seed
            per_seed.append(m)
        ba = [p["balanced_accuracy"] for p in per_seed]
        pred0, _ = grouped_logreg(emb, labels, fovs, 0)
        ci = cluster_bootstrap_ci(labels, pred0, fovs, seed=0)
        perm = permutation_p(lambda yp, emb=emb: grouped_logreg(emb, yp, fovs, 0)[0],
                             labels, fovs, observed=ci["point"], n_perm=200, seed=0)
        mlp = grouped_mlp(emb, labels, fovs, 0)
        task1[key] = {
            "status": "ok", "dim": int(emb.shape[1]),
            "balanced_accuracy_mean": float(np.mean(ba)),
            "balanced_accuracy_sd": float(np.std(ba)),
            "auroc_mean": float(np.nanmean([p.get("auroc") or np.nan for p in per_seed])),
            "macro_f1_mean": float(np.mean([p["macro_f1"] for p in per_seed])),
            "per_seed": per_seed,
            "cluster_bootstrap_ci_seed0": ci,
            "fov_permutation_test_seed0": perm,
            "mlp_balanced_accuracy_seed0": float(
                classification_metrics(labels, mlp)["balanced_accuracy"]),
        }
        print(f"  T1 {key:32s} BA {np.mean(ba):.3f}+/-{np.std(ba):.3f} "
              f"CI[{ci['ci_low']:.3f},{ci['ci_high']:.3f}] p={perm['p_value']}")
    # paired comparisons against E1 (hand-crafted) and E6 (architecture control).
    # E4 vs E6 is the comparison that isolates the effect of pretraining from the
    # effect of the architecture, since E6 is the same network with random weights.
    preds = {k: grouped_logreg(e, labels, fovs, 0)[0]
             for k, e in embeddings.items() if e is not None}
    for ref in ("E1_handcrafted_morphology", "E6_tijepa0_random_init"):
        if ref not in preds:
            continue
        for key in preds:
            if key == ref:
                continue
            task1[key][f"paired_bootstrap_vs_{ref.split('_')[0]}"] = paired_bootstrap(
                labels, preds[key], preds[ref], fovs, seed=0)
    results["tasks"]["morphology_class"] = task1

    # ---- task 2: LPS exposure-time regression
    from scipy import stats as sps

    task2 = {}
    for key, emb in embeddings.items():
        if emb is None:
            task2[key] = {"status": "N/A" if key.startswith("E5") else "unavailable"}
            continue
        pred = grouped_ridge(emb, hours, fovs, 0)
        rho, p = sps.spearmanr(hours, pred)
        task2[key] = {
            "status": "ok",
            "spearman_rho": float(rho), "spearman_p": float(p),
            "mae_hours": float(np.mean(np.abs(pred - hours))),
            "mae_hours_baseline_mean_predictor": float(
                np.mean(np.abs(hours - hours.mean()))),
        }
        print(f"  T2 {key:32s} rho {rho:+.3f} MAE {task2[key]['mae_hours']:.2f}h "
              f"(mean-predictor {task2[key]['mae_hours_baseline_mean_predictor']:.2f}h)")
    results["tasks"]["lps_exposure_time_regression"] = task2

    # ---- confound probe: can the embedding predict FOV identity?
    conf = {}
    for key, emb in embeddings.items():
        if emb is None:
            continue
        pred, _ = grouped_logreg(emb, fovs.astype(str),
                                 np.arange(len(fovs)) % 5, 0)  # random folds on purpose
        conf[key] = {
            "fov_identity_balanced_accuracy": float(
                classification_metrics(fovs.astype(str), pred)["balanced_accuracy"]),
            "n_fovs": int(len(set(fovs))),
            "chance": float(1.0 / len(set(fovs))),
            "note": "higher means the embedding carries more FOV/batch identity "
                    "(lower is better)",
        }
    results["tasks"]["fov_confound_probe"] = conf

    np.savez_compressed(OUT / "bbbc054_embeddings.npz",
                        labels=labels, fovs=fovs, hours=hours,
                        **{k: v for k, v in embeddings.items() if v is not None})
    (OUT / "bbbc054_probes.json").write_text(json.dumps(results, indent=2,
                                                        default=float) + "\n")
    print(f"wrote {OUT/'bbbc054_probes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
