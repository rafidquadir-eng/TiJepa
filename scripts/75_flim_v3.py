"""H5.4 coverage map, metabolism: two-photon NAD(P)H-FLIM of macrophages (Zenodo 14330511,
CC BY 4.0; Musimbi et al. 2024, raw data of "Two-Photon NAD(P)H-FLIM reveals unperturbed
energy metabolism of Ascaris suum larvae, in contrast to host macrophages upon artemisinin
derivatives exposure").

Raw data, macrophage folders only (Fig1_M0-vs-M1, Fig2_M2-DMSO-ARS): each field of view
(FOV) is 5 repeated TCSPC acquisitions, each an OME-TIFF of 227 lifetime bins x 512 x 512
pixels (bin 0.054991 ns, 0.390625 um/px). Data-type guard: every file must have 227 pages
of 512 x 512 uint16.

Per FOV the 5 repeats are summed. The intensity image is the photon sum over bins. Cells
are segmented on the Gaussian-smoothed (sigma 2 px) intensity image by Otsu thresholding,
objects of 80-3,000 px kept, and a 64 x 64 crop (25 um) is cut around each centroid. The
decay of each cell is the sum over its mask. Two documented lifetime readouts, both on
the decay tail (from 0.5 ns after the peak, so the instrument response need not be
deconvolved):

* tau_m: first moment of the background-subtracted tail (intensity-weighted mean
  lifetime for a sum of exponentials, truncated by the 12.5 ns window);
* bound fraction: non-negative least squares of the tail on exp(-t/0.4 ns) (free NAD(P)H),
  exp(-t/3.4 ns) (enzyme-bound NAD(P)H) and a constant, amplitude fraction of the bound
  term. The two lifetimes are the commonly used literature values, fixed in advance.

Tests (frozen encoders see the **intensity** crop only, never the decay):
1. primary: predict tau_m and the bound fraction per cell (ridge, FOV-grouped 5-fold CV;
   R^2 against the fold mean and Spearman rho; cluster bootstrap over FOVs);
2. secondary: artesunate (IL4-DMSO-ARS1uM) vs vehicle (IL4-DMSO) from the intensity crop,
   FOV-grouped; the lifetime readouts themselves are the positive control for this task
   (does the drug change NAD(P)H lifetime in these data at all?). Two biological replicates
   per arm, so the arm comparison is descriptive.
The Fig1 arms are not used for classification: their labels ("Mac-DMSO",
"unstimMAC-DMSO") do not say which is M1, and they were imaged on different dates.
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ZIP = REPO / "data" / "raw" / "zenodo14330511" / "Musimbi-et-al_2024_rawdata.zip"
ZIP_MD5 = "e1892af8b2c51fed0d2d60a8ffb1287d"
CELLS = REPO / "data" / "interim" / "flim_cells.npz"
STAGE = REPO / "data" / "interim" / "stage_flim"
OUT = REPO / "results" / "v3" / "coverage" / "flim.json"
BIN_NS, N_BINS, PX = 0.054991, 227, 512
TAU_FREE, TAU_BOUND = 0.4, 3.4
CROP = 64
SEEDS = (0, 1, 2)


def fovs() -> dict[str, list[str]]:
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.endswith(".ome.tif")
                 and ("/Fig1_M0-vs-M1/" in n or "/Fig2_M2-DMSO-ARS/" in n)]
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(n.split("_C00_Time")[0], []).append(n)
    return {k: sorted(v) for k, v in sorted(out.items())}


def parse(fov: str) -> dict:
    base = fov.split("/")[-1]
    fig = fov.split("/")[-2]
    m = re.match(r"^[\d-]+_(\d{6})_([\d-]+(?:-[ab])?)_(.+)_([ivx]+)_DC-TCSPC$", base)
    date, sample, arm, pos = m.groups()
    return {"figure": fig, "date": date, "sample": f"{date}_{sample}", "arm": arm,
            "position": pos}


def lifetime(decay: np.ndarray) -> dict:
    t = np.arange(N_BINS) * BIN_NS
    pk = int(np.argmax(decay))
    st = pk + int(round(0.5 / BIN_NS))
    bg = float(np.median(decay[: max(1, pk - 10)])) if pk > 12 else float(decay[:3].mean())
    tail = np.clip(decay[st:] - bg, 0, None)
    tt = t[st:] - t[st]
    out = {"photons": float(decay.sum()), "peak_bin": pk}
    out["tau_m_ns"] = float((tt * tail).sum() / max(tail.sum(), 1e-9))
    from scipy.optimize import nnls
    A = np.stack([np.exp(-tt / TAU_FREE), np.exp(-tt / TAU_BOUND), np.ones_like(tt)], 1)
    coef, _ = nnls(A, decay[st:].astype(np.float64))
    out["bound_fraction"] = float(coef[1] / max(coef[0] + coef[1], 1e-9))
    return out


def fov_worker(args: tuple) -> dict:
    fov, files = args
    tmp = tempfile.mkdtemp(prefix="flim_", dir=str(STAGE))
    try:
        import tifffile
        from scipy.ndimage import gaussian_filter
        from skimage.filters import threshold_otsu
        from skimage.measure import label, regionprops
        decay = np.zeros((N_BINS, PX, PX), np.float32)
        with zipfile.ZipFile(ZIP) as z:
            for f in files:
                local = Path(tmp) / "x.tif"
                with z.open(f) as s, local.open("wb") as d:
                    shutil.copyfileobj(s, d, length=1 << 22)
                with tifffile.TiffFile(local) as tf:
                    if len(tf.pages) != N_BINS or tf.pages[0].shape != (PX, PX):
                        return {"fov": fov, "status": "data-type guard failed",
                                "detail": f"{len(tf.pages)} pages, {tf.pages[0].shape}"}
                    for i, p in enumerate(tf.pages):
                        decay[i] += p.asarray()
                local.unlink()
        inten = decay.sum(0)
        sm = gaussian_filter(inten, 2)
        lab = label(sm > threshold_otsu(sm))
        crops, feats, cents = [], [], []
        h = CROP // 2
        for r in regionprops(lab):
            if not 80 <= r.area <= 3000:
                continue
            cy, cx = (int(round(v)) for v in r.centroid)
            if cy - h < 0 or cx - h < 0 or cy + h > PX or cx + h > PX:
                continue
            ys, xs = r.coords[:, 0], r.coords[:, 1]
            cell_decay = decay[:, ys, xs].sum(1)
            lt = lifetime(cell_decay)
            if lt["photons"] < 2000:
                continue
            crops.append(inten[cy - h:cy + h, cx - h:cx + h].astype(np.float32))
            feats.append(lt)
            cents.append((cy, cx, int(r.area)))
        fov_lt = lifetime(decay[:, lab > 0].sum(1)) if (lab > 0).any() else None
        return {"fov": fov, "status": "ok", "n_files": len(files),
                "crops": np.stack(crops) if crops else np.zeros((0, CROP, CROP), np.float32),
                "cells": feats, "centroids": cents, "fov_lifetime": fov_lt}
    except Exception as exc:
        return {"fov": fov, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def extract(workers: int) -> None:
    import hashlib
    h = hashlib.md5()
    with open(ZIP, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != ZIP_MD5:
        raise SystemExit("FLIM archive MD5 does not match the Zenodo record")
    STAGE.mkdir(parents=True, exist_ok=True)
    fv = fovs()
    print(f"{len(fv)} macrophage FOVs")
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(fov_worker, list(fv.items())))
    X, rows, status = [], [], []
    for r in res:
        meta = parse(r["fov"])
        status.append({"fov": r["fov"].split("/")[-1], **meta, "status": r["status"],
                       "n_cells": len(r.get("cells", [])),
                       "fov_lifetime": r.get("fov_lifetime"),
                       **({"error": r["error"]} if "error" in r else {})})
        if r["status"] != "ok":
            continue
        X.append(r["crops"])
        for c, (cy, cx, a) in zip(r["cells"], r["centroids"], strict=False):
            rows.append({**meta, "fov": r["fov"].split("/")[-1], "y": cy, "x": cx,
                         "area_px": a, **c})
        print(f"  {r['fov'].split('/')[-1]}: {len(r['cells'])} cells")
    CELLS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CELLS, X=np.concatenate(X), rows=json.dumps(rows),
                        status=json.dumps(status))
    print(f"wrote {CELLS}: {len(rows)} cells")


def analyse(encoders: str) -> None:
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score

    from tijepa.eval import encoders as ENC
    from tijepa.eval import probe as P
    from tijepa.eval import provenance as PV
    from tijepa.preprocess import morphology_v1 as M1

    z = np.load(CELLS)
    X = z["X"]
    rows = json.loads(str(z["rows"]))
    status = json.loads(str(z["status"]))
    fov = np.array([r["fov"] for r in rows])
    arm = np.array([r["arm"] for r in rows])
    fig = np.array([r["figure"] for r in rows])
    targets = {"tau_m_ns": np.array([r["tau_m_ns"] for r in rows]),
               "bound_fraction": np.array([r["bound_fraction"] for r in rows])}
    Xn = ENC.minmax(X)
    res: dict = {
        "phase": "PLAN_v3 H5.4 (modality family: metabolism)",
        "dataset": {"zenodo": "14330511", "doi": "10.5281/zenodo.14330511",
                    "licence": "CC BY 4.0", "archive_md5": ZIP_MD5,
                    "folders_used": ["Fig1_M0-vs-M1", "Fig2_M2-DMSO-ARS"]},
        "fovs": status, "n_cells": len(rows), "n_fovs": int(len(np.unique(fov))),
        "lifetime_method": {"tau_m": "first moment of the background-subtracted decay tail "
                                     "from 0.5 ns after the peak",
                            "bound_fraction": f"NNLS on exp(-t/{TAU_FREE} ns), "
                                              f"exp(-t/{TAU_BOUND} ns) and a constant",
                            "bin_ns": BIN_NS, "n_bins": N_BINS},
        "target_summary": {k: {"mean": float(v.mean()), "sd": float(v.std()),
                               "p5_p95": [float(np.percentile(v, 5)),
                                          float(np.percentile(v, 95))]}
                           for k, v in targets.items()},
        "probe": "ridge (alpha 10) after PCA(64), FOV-grouped 5-fold CV, fold seeds 0-2; "
                 "R^2 against the observed values (a constant fold-mean predictor scores "
                 "<= 0); 95% cluster bootstrap over FOVs",
        "encoders": {}, "regression": {}, "classification": {},
    }
    feats: dict[str, np.ndarray] = {}
    yy, xx = np.mgrid[:CROP, :CROP]
    disk = (yy - 31.5) ** 2 + (xx - 31.5) ** 2 <= 10 ** 2
    v = X[:, disk]
    feats["intensity_baseline"] = np.stack([v.mean(1), np.median(v, 1), v.std(1),
                                            np.percentile(v, 90, axis=1),
                                            X.reshape(len(X), -1).mean(1)], 1)
    feats["E1_handcrafted"] = np.hstack([np.stack([M1.features_single_frame(im)[0]
                                                   for im in Xn]),
                                         feats["intensity_baseline"]])
    for key, rec, fn in ENC.all_encoders(tuple(encoders.split(","))):
        PV.label_for_key(key, rec)
        res["encoders"][key] = {"checkpoint": rec}
        feats[key] = fn(Xn)
        print(f"  embedded {key}")

    for tname, y in targets.items():
        res["regression"][tname] = {}
        for key, f in feats.items():
            per = []
            preds = {}
            for s in SEEDS:
                p = P.grouped_regress(f, y, fov, s)
                preds[s] = p
                per.append(float(r2_score(y, p)))
            ci = P.cluster_bootstrap(y, preds[0], fov, stat=r2_score)["ci95"]
            res["regression"][tname][key] = {
                "r2_mean": float(np.mean(per)), "r2_sd": float(np.std(per, ddof=1)),
                "r2_per_seed": per, "r2_ci95_seed0": ci,
                "spearman_seed0": float(spearmanr(y, preds[0]).statistic)}
            print(f"  {tname:15s} {key:24s} R2 {np.mean(per):+.3f} CI {ci}")

    m = (fig == "Fig2_M2-DMSO-ARS") & np.isin(arm, ["porMc_IL4-DMSO",
                                                    "porMc_IL4-DMSO-ARS1uM"])
    yc = np.where(arm[m] == "porMc_IL4-DMSO-ARS1uM", "ARS", "DMSO")
    lt_feats = np.stack([targets["tau_m_ns"], targets["bound_fraction"]], 1)
    cls = {"task": "artesunate 1 uM vs vehicle (IL4-DMSO), porcine IL-4 macrophages",
           "n_cells": int(m.sum()), "n_fovs": int(len(np.unique(fov[m]))),
           "replicates_per_arm": 2,
           "class_counts": {k: int((yc == k).sum()) for k in ("ARS", "DMSO")},
           "results": {}}
    for key, f in {"FLIM_lifetime_readouts (positive control)": lt_feats, **feats}.items():
        per, preds = [], {}
        for s in SEEDS:
            p = P.grouped_predict(f[m], yc, fov[m], s)
            preds[s] = p
            per.append(float(P.balanced_accuracy_score(yc, p)))
        cls["results"][key] = {"balanced_accuracy_mean": float(np.mean(per)),
                               "per_seed": per,
                               "ci95_seed0": P.cluster_bootstrap(yc, preds[0], fov[m])["ci95"]}
        print(f"  ARS vs DMSO {key:40s} BA {np.mean(per):.3f}")
    cls["arm_means"] = {k: {"tau_m_ns": float(targets["tau_m_ns"][m][yc == k].mean()),
                            "bound_fraction": float(targets["bound_fraction"][m][yc == k]
                                                    .mean())} for k in ("ARS", "DMSO")}
    res["classification"]["ars_vs_dmso"] = cls
    tm = targets["tau_m_ns"]
    res["data_type_guard"] = {
        "rule": "every file 227 pages of 512 x 512 uint16; median cell tau_m within the "
                "NAD(P)H range 0.3-3.5 ns",
        "median_tau_m_ns": float(np.median(tm)),
        "pass": bool(0.3 <= np.median(tm) <= 3.5
                     and all(s["status"] == "ok" for s in status))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--encoders", default="E2,E3,E7,E4,E4s,E0,E6")
    a = ap.parse_args()
    t0 = time.time()
    if a.extract:
        extract(a.workers)
    if a.analyse:
        analyse(a.encoders)
    print(f"{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
