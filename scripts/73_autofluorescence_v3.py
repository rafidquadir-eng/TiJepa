"""H5.4 coverage map, immunophenotype / cell identity: label-free two-photon
autofluorescence of mouse immune cells (Zenodo 15657646, CC BY 4.0; images from Lemire
et al.; deposit paper Kreiss et al., J Biophotonics 2026, doi 10.1002/jbio.70260).

Each 64 x 64 patch (25.6 um square, one centred cell) has three label-free channels: two
autofluorescence channels (NAD(P)H and FAD band-passes; the deposit does not document the
channel order, so they are called AF-0 and AF-1) and a forward-scattered Dodt gradient-
contrast channel (index 2, identified by its flat ~9,600-count background).

* Set A (primary, unconfounded): CD4+ T cells and neutrophils imaged **in the same field**
  (20 of 31 FOVs are mixtures), ground truth from an anti-CD3 APC channel recorded at a
  different excitation and not given to any model. Grouped by FOV (31 groups).
  A mixed-FOV-only analysis (20 FOVs) removes every FOV-level shortcut.
* Set B (secondary): six cell types each imaged in separate samples (85 FOVs). Grouped by
  FOV, but class and imaging session are confounded by design, which is stated.

Encoders: frozen E2, E3, E7, E4, E4s, E0 and three random-init draws (E6), each applied
per channel; feature sets "all three channels" (primary), "Dodt only" and "AF only".
Baselines: per-channel intensity statistics (a metabolic-intensity baseline) and E1
hand-crafted morphology + texture + intensity on the AF-sum image. Probe and statistics:
src/tijepa/eval/probe.py (FOV-grouped 5-fold CV, three fold seeds, cluster bootstrap over
FOVs, BH within task).

Data-type guard: every file must be a (3, 64, 64) uint16 TIFF and the patch totals must
equal the deposit's stated 5,078 (A) and 3,424 (B). Positive control: the intensity
baseline must classify Set A above chance (bootstrap CI lower bound > 0.5).
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.eval import encoders as ENC  # noqa: E402
from tijepa.eval import probe as P  # noqa: E402
from tijepa.eval import provenance as PV  # noqa: E402

RAW = REPO / "data" / "raw" / "zenodo15657646"
OUT = REPO / "results" / "v3" / "coverage" / "autofluorescence.json"
SEEDS = (0, 1, 2)
EXPECTED = {"A": 5078, "B": 3424}
CLASS_NAMES = {
    "A": {"1": "CD4 T cell (APC+)", "2": "neutrophil"},
    "B": {
        "1": "CD4 T cell",
        "2": "CD8 T cell",
        "3": "B cell",
        "4": "neutrophil",
        "5": "dendritic cell",
        "6": "macrophage",
    },
}
FILES = {
    "A": (
        "64x64_patches_Tcells_and_neutrophils_mixed_samples.zip",
        "setA_mixed",
        "ee985f3fcfff79f48edbb9002fbc986e",
    ),
    "B": (
        "64x64_patches_six_cell_types_isolated_samples.zip",
        "setB_isolated",
        "609dbab10028dd768335de581f2b149a",
    ),
}


def load(set_key: str) -> dict:
    import tifffile

    d = RAW / FILES[set_key][1]
    fs = sorted(d.glob("*.tif"))
    X, y, fov = [], [], []
    for f in fs:
        a = tifffile.imread(f)
        if a.shape != (3, 64, 64) or a.dtype != np.uint16:
            raise SystemExit(f"data-type guard: {f.name} is {a.shape} {a.dtype}")
        X.append(a)
        y.append(re.search(r"_class(\d+)\.tif$", f.name).group(1))
        fov.append(re.sub(r"_\d+_class\d+\.tif$", "", f.name))
    if len(fs) != EXPECTED[set_key]:
        raise SystemExit(
            f"data-type guard: set {set_key} has {len(fs)} patches, the "
            f"deposit states {EXPECTED[set_key]}"
        )
    fov = np.array(fov)
    mixed = np.array(["neutrophil" in f.lower() and "cd4" in f.lower() for f in fov])
    return {
        "X": np.stack(X).astype(np.float32),
        "y": np.array(y),
        "fov": fov,
        "mixed_fov": mixed,
    }


def intensity_features(X: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[:64, :64]
    disk = (yy - 31.5) ** 2 + (xx - 31.5) ** 2 <= 10**2
    feats = []
    for c in range(3):
        v = X[:, c][:, disk]
        feats += [
            v.mean(1),
            np.median(v, 1),
            np.percentile(v, 90, axis=1),
            v.std(1),
            X[:, c].reshape(len(X), -1).mean(1),
        ]
    a0, a1 = X[:, 0][:, disk].mean(1), X[:, 1][:, disk].mean(1)
    feats.append(a1 / (a0 + a1 + 1e-6))
    return np.stack(feats, 1)


def handcrafted(X: np.ndarray) -> np.ndarray:
    from tijepa.preprocess import morphology_v1 as M1

    af = ENC.minmax(X[:, 0] + X[:, 1])
    rows = [M1.features_single_frame(img)[0] for img in af]
    return np.hstack([np.stack(rows), intensity_features(X)])


def evaluate(z: np.ndarray, d: dict, mask: np.ndarray) -> dict:
    y, g = d["y"][mask], d["fov"][mask]
    per_seed, preds = [], {}
    for s in SEEDS:
        p = P.grouped_predict(z[mask], y, g, s)
        preds[s] = p
        per_seed.append(float(P.balanced_accuracy_score(y, p)))
    return {
        "balanced_accuracy_mean": float(np.mean(per_seed)),
        "balanced_accuracy_sd": float(np.std(per_seed, ddof=1)),
        "per_seed": per_seed,
        "ci95_seed0": P.cluster_bootstrap(y, preds[0], g)["ci95"],
        "_pred0": preds[0],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="E2,E3,E7,E4,E4s,E0,E6")
    args = ap.parse_args()
    t0 = time.time()
    import hashlib

    for k, (zf, _d, md5) in FILES.items():
        h = hashlib.md5((RAW / zf).read_bytes()).hexdigest()
        if h != md5:
            raise SystemExit(f"set {k}: archive MD5 {h} != Zenodo {md5}")
    data = {k: load(k) for k in ("A", "B")}
    tasks = {
        "A_all_fovs": ("A", np.ones(len(data["A"]["y"]), bool)),
        "A_mixed_fovs_only": ("A", data["A"]["mixed_fov"]),
        "B_six_types": ("B", np.ones(len(data["B"]["y"]), bool)),
    }
    res: dict = {
        "phase": "PLAN_v3 H5.4 (modality family: immunophenotype / cell identity; "
        "label-free metabolic autofluorescence)",
        "dataset": {
            "zenodo": "15657646",
            "doi": "10.5281/zenodo.15657646",
            "licence": "CC BY 4.0",
            "source_publication": "10.1002/jbio.70260",
            "files_md5_verified": {v[0]: v[2] for v in FILES.values()},
        },
        "channels": {
            "0": "AF-0 (autofluorescence band-pass; NAD(P)H or FAD, order not documented)",
            "1": "AF-1",
            "2": "Dodt gradient contrast",
        },
        "class_names": CLASS_NAMES,
        "tasks": {
            t: {
                "set": s,
                "n_cells": int(m.sum()),
                "n_fovs": int(len(np.unique(data[s]["fov"][m]))),
                "class_counts": {
                    CLASS_NAMES[s][c]: int((data[s]["y"][m] == c).sum())
                    for c in sorted(np.unique(data[s]["y"][m]))
                },
                "chance_balanced_accuracy": 1.0 / len(np.unique(data[s]["y"][m])),
            }
            for t, (s, m) in tasks.items()
        },
        "confounds": {
            "B_six_types": "each cell type was isolated and imaged in its own "
            "samples, so cell type and imaging session are "
            "confounded; FOV grouping cannot remove that"
        },
        "probe": "FOV-grouped StratifiedGroupKFold(5), PCA(64) + class-balanced logistic "
        "regression, fold seeds 0-2; 95% cluster bootstrap over FOVs (seed 0)",
        "encoders": {},
        "results": {},
    }

    feats: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for s, d in data.items():
        feats.setdefault("intensity_baseline", {})[s] = {"all": intensity_features(d["X"])}
        feats.setdefault("E1_handcrafted", {})[s] = {"all": handcrafted(d["X"])}
    res["encoders"]["intensity_baseline"] = {
        "note": "per-channel intensity statistics in a 10-px disk + AF-1/(AF-0+AF-1)"
    }
    res["encoders"]["E1_handcrafted"] = {
        "note": "segmentation + shape + Zernike + Haralick "
        "+ intensity on the AF-sum image, plus the "
        "intensity baseline"
    }
    include = tuple(args.encoders.split(","))
    for key, rec, fn in ENC.all_encoders(include):
        PV.label_for_key(key, rec)
        res["encoders"][key] = {"checkpoint": rec}
        if key.startswith("E7"):
            from tijepa.eval import vjepa2 as VJ2

            res["encoders"][key]["model"] = VJ2.record()
        for s, d in data.items():
            zc = [fn(ENC.minmax(d["X"][:, c])) for c in range(3)]
            feats.setdefault(key, {})[s] = {
                "all": np.hstack(zc),
                "dodt": zc[2],
                "af": np.hstack(zc[:2]),
            }
        print(f"  embedded {key}")

    for t, (s, m) in tasks.items():
        res["results"][t] = {}
        for key, per_set in feats.items():
            for fs_name, z in per_set[s].items():
                r = evaluate(z, data[s], m)
                res["results"][t][f"{key}|{fs_name}"] = r
                print(
                    f"  {t:20s} {key:24s} {fs_name:5s} BA {r['balanced_accuracy_mean']:.3f}"
                    f" +/- {r['balanced_accuracy_sd']:.3f} CI {r['ci95_seed0']}"
                )
        # paired comparisons of TiJepa-1 (all channels) against every other row
        ref = "E4_tijepa1|all"
        if ref in res["results"][t]:
            y, g = data[s]["y"][m], data[s]["fov"][m]
            cmp = {}
            for k, r in res["results"][t].items():
                if k == ref:
                    continue
                cmp[k] = P.paired_bootstrap(
                    y, res["results"][t][ref]["_pred0"], r["_pred0"], g
                )
            q = P.bh([c["p_two_sided"] for c in cmp.values()])
            for c, qq in zip(cmp.values(), q, strict=False):
                c["q_bh"] = qq
            res[f"paired_vs_tijepa1_{t}"] = cmp
        for r in res["results"][t].values():
            r.pop("_pred0", None)

    pc = res["results"]["A_all_fovs"]["intensity_baseline|all"]
    res["positive_control"] = {
        "rule": "intensity baseline on Set A: cluster-bootstrap CI lower bound > 0.5",
        "ci95": pc["ci95_seed0"],
        "pass": bool(pc["ci95_seed0"] and pc["ci95_seed0"][0] > 0.5),
    }
    res["seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"positive control: {res['positive_control']['pass']}; wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
