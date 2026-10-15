"""M2 / Part B: signalling dynamics -> LPS tolerance (PLAN.md Section 6).

B1  condition audit table (results/partB/conditions.csv)
B3.1 tolerance from dynamics: naive->LPS vs LPS->LPS, grouped CV, CIs, permutation
B3.2 first-ligand identity, reported alongside (not against) the published 4-class number
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import functools

import numpy as np

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.eval.stats import (  # noqa: E402
    classification_metrics, cluster_bootstrap_ci, permutation_p,
)
from tijepa.io import tay  # noqa: E402
from tijepa.models.trace_jepa import trace_features  # noqa: E402

OUT = REPO / "results" / "partB"
SEEDS = [0, 1, 2]


def write_json(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=float) + "\n")


def b1_conditions(ts: tay.TraceSet) -> list[dict]:
    """Audit table: every (source, first, second, gap) cell count and well count."""
    rows = []
    keys = sorted({(s, f, sec, g) for s, f, sec, g in
                   zip(ts.source, ts.first, ts.second, ts.gap)})
    for s, f, sec, g in keys:
        m = ((ts.source == s) & (ts.first == f) & (ts.second == sec) & (ts.gap == g))
        tr = ts.traces[m]
        i0 = tay.second_stim_index(g)
        complete = int(np.isfinite(tr[:, i0:]).all(axis=1).sum()) if i0 < tr.shape[1] else 0
        rows.append({
            "source_file": s,
            "layout_verified": bool(s in tay.VERIFIED_LAYOUT_FILES),
            "first_ligand_code": f,
            "second_ligand_code": sec,
            "second_ligand_name": tay.LIGAND_CODES.get(sec, f"code_{sec:g}_UNIDENTIFIED"),
            "first_is_naive": bool(f == tay.NAIVE_CODE),
            "homologous_prime": bool(f == sec),
            "interstim_interval_h": g,
            "n_cells": int(m.sum()),
            "n_wells": int(len(set(ts.well[m]))),
            "n_cells_complete_after_second_stim": complete,
            "second_stim_frame_index": i0,
            "gsm_rnaseq": "",   # no GSM-level mapping is published; see RUN_REPORT
            "gsm_atac": "",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "conditions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"B1: {len(rows)} condition rows -> {OUT/'conditions.csv'}")
    return rows


def choose_C(x, y, seed: int = 0) -> float:
    """Tune C once by inner CV on the full design; permutation nulls then reuse it.

    PLAN.md A4.1 asks for "C tuned by inner CV". Re-tuning inside each of the 200
    group-permutation refits is computationally prohibitive, so C is tuned once and
    held fixed for the null. Recorded as a deviation in RUN_REPORT.md.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=np.logspace(-3, 3, 7), cv=3, max_iter=2000,
                             class_weight="balanced", random_state=seed, n_jobs=-1),
    )
    clf.fit(x, y)
    return float(clf[-1].C_[0])


def fit_probe(x_tr, y_tr, x_te, kind: str, seed: int, C: float = 1.0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                               random_state=seed),
        )
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)
    else:
        raise ValueError(kind)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    try:
        score = clf.predict_proba(x_te)
    except AttributeError:
        score = None
    return pred, score


def grouped_cv_predict(x, y, groups, kind: str, seed: int, C: float = 1.0):
    """Leave-one-group-out when groups are few, else grouped 5-fold."""
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    uniq = np.unique(groups)
    cv = LeaveOneGroupOut() if len(uniq) <= 12 else GroupKFold(n_splits=5)
    pred = np.empty_like(y)
    score = np.zeros((len(y), len(np.unique(y))), dtype=float)
    for tr, te in cv.split(x, y, groups):
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        p, s = fit_probe(x[tr], y[tr], x[te], kind, seed, C)
        pred[te] = p
        if s is not None and s.shape[1] == score.shape[1]:
            score[te] = s
    return pred, score


def b31_tolerance(ts: tay.TraceSet) -> dict:
    """naive->LPS vs LPS->LPS from the response to the second LPS stimulus."""
    lps = [c for c, n in tay.LIGAND_CODES.items() if n == "LPS"][0]
    res: dict = {
        "task": "tolerance_from_dynamics",
        "files_restricted_to": sorted(tay.VERIFIED_LAYOUT_FILES),
        "contrast": "naive->LPS (first_code=0) vs LPS->LPS (first_code=LPS)",
        "ligand_code_inference": tay.LIGAND_CODES,
        "ligand_code_inference_note": (
            "code 2 = LPS identified by intersecting the second-ligand code sets of "
            "TNFLPSfirst {2,6} and CPGLPSscrep {2,3} with their file-name ligand sets "
            "{TNF,LPS} and {CpG,LPS}; the deposit ships no column dictionary and the "
            "label folder single_cell_features/ was not deposited."
        ),
        "by_interval": {},
    }
    for gap in sorted(set(ts.gap[ts.layout_verified])):
        sel = (ts.second == lps) & (ts.gap == gap) & ts.layout_verified & (
            (ts.first == tay.NAIVE_CODE) | (ts.first == lps))
        sub = ts.subset(sel)
        if len(sub) == 0:
            continue
        win, i0, i1 = tay.response_window(sub, gap)
        ok = np.isfinite(win).all(axis=1)
        sub, win = sub.subset(ok), win[ok]
        y = (sub.first == lps).astype(int)  # 1 = previously LPS-exposed (tolerant)
        groups = sub.group
        if len(np.unique(y)) < 2 or len(np.unique(groups)) < 4:
            res["by_interval"][f"{gap:g}h"] = {
                "status": "skipped", "reason": "too few classes or groups",
                "n_cells": int(len(sub)), "n_groups": int(len(np.unique(groups))),
                "n_tolerant": int(y.sum()), "n_naive": int((1 - y).sum()),
            }
            continue

        x_feat, names = trace_features(win)
        src_counts = {}
        for sname in sub.source:
            src_counts[str(sname)] = src_counts.get(str(sname), 0) + 1
        entry = {
            "status": "ok", "n_cells": int(len(y)),
            "cells_per_source_file": src_counts,
            "n_naive": int((y == 0).sum()), "n_tolerant": int((y == 1).sum()),
            "n_groups": int(len(np.unique(groups))),
            "response_window_frames": [int(i0), int(i1)],
            "response_window_minutes": [float(i0 * tay.FRAME_MIN),
                                        float(i1 * tay.FRAME_MIN)],
            "n_features": int(x_feat.shape[1]), "feature_names": names,
            "representations": {},
        }
        for rep_name, x in (("handcrafted", x_feat), ("raw_trace", np.nan_to_num(win))):
            C = choose_C(x, y)
            per_seed = []
            for seed in SEEDS:
                pred, score = grouped_cv_predict(x, y, groups, "logreg", seed, C)
                m = classification_metrics(y, pred, score)
                m["seed"] = seed
                per_seed.append(m)
            ba = [p["balanced_accuracy"] for p in per_seed]
            pred0, score0 = grouped_cv_predict(x, y, groups, "logreg", 0, C)
            ci = cluster_bootstrap_ci(y, pred0, groups, seed=0)
            perm = permutation_p(
                lambda yp: grouped_cv_predict(x, yp, groups, "logreg", 0, C)[0],
                y, groups, observed=ci["point"], n_perm=200, seed=0,
            )
            entry["representations"][rep_name] = {
                "logreg_C_tuned_once": C,
                "per_seed": per_seed,
                "balanced_accuracy_mean": float(np.mean(ba)),
                "balanced_accuracy_sd": float(np.std(ba)),
                "auroc_mean": float(np.mean([p.get("auroc") or np.nan for p in per_seed])),
                "cluster_bootstrap_ci_seed0": ci,
                "group_permutation_test_seed0": perm,
                "n_perm_note": "200 permutations (PLAN 8.1 asks >=200 for B3.3; "
                               "reduced from 1000 for runtime, recorded as a deviation)",
            }
            print(f"  B3.1 {gap:g}h {rep_name}: BA {np.mean(ba):.3f}+/-{np.std(ba):.3f} "
                  f"CI [{ci['ci_low']}, {ci['ci_high']}] p={perm['p_value']}")
        res["by_interval"][f"{gap:g}h"] = entry
    return res


def b32_ligand_identity(ts: tay.TraceSet) -> dict:
    """First-ligand identity from the response to the FIRST stimulus."""
    sel = (np.isfinite(ts.traces[:, tay.STIM1_IDX:tay.STIM1_IDX + 40]).all(axis=1)
           & ts.layout_verified)
    sub = ts.subset(sel)
    win = sub.traces[:, tay.STIM1_IDX:tay.STIM1_IDX + 40] - np.nanmean(
        sub.traces[:, :tay.STIM1_IDX], axis=1, keepdims=True)
    # restrict to codes with enough wells to be learnable
    codes, counts = np.unique(sub.first, return_counts=True)
    keep_codes = [c for c, n in zip(codes, counts)
                  if n >= 200 and len(set(sub.well[sub.first == c])) >= 4]
    m = np.isin(sub.first, keep_codes)
    sub2, win2 = sub.subset(m), win[m]
    y = sub2.first.astype(int)
    groups = sub2.group
    x, names = trace_features(win2)
    C = choose_C(x, y)
    per_seed = []
    for seed in SEEDS:
        pred, score = grouped_cv_predict(x, y, groups, "logreg", seed, C)
        mm = classification_metrics(y, pred, score)
        mm["seed"] = seed
        per_seed.append(mm)
    ba = [p["balanced_accuracy"] for p in per_seed]
    pred0, _ = grouped_cv_predict(x, y, groups, "logreg", 0, C)
    out = {
        "task": "first_ligand_identity",
        "logreg_C_tuned_once": C,
        "n_classes": int(len(keep_codes)),
        "class_codes": [float(c) for c in keep_codes],
        "n_cells": int(len(y)), "n_groups": int(len(np.unique(groups))),
        "window": "first stimulus, 40 frames (4 h) from frame 3",
        "per_seed": per_seed,
        "balanced_accuracy_mean": float(np.mean(ba)),
        "balanced_accuracy_sd": float(np.std(ba)),
        "cluster_bootstrap_ci_seed0": cluster_bootstrap_ci(y, pred0, groups, seed=0),
        "comparison_note": (
            "NOT comparable to the published 72.8% 4-class result: the published task "
            "uses a different class set, cell filtering and validation scheme. Reported "
            "here only as an internal reference point."
        ),
    }
    print(f"  B3.2 {out['n_classes']}-class ligand identity: "
          f"BA {out['balanced_accuracy_mean']:.3f}+/-{out['balanced_accuracy_sd']:.3f}")
    return out


def main() -> int:
    ts = tay.load(REPO)
    print(f"loaded {len(ts)} cells from {len(set(ts.source))} deposit files")
    rows = b1_conditions(ts)
    write_json("b1_summary.json", {
        "n_cells_total": int(len(ts)),
        "n_source_files": int(len(set(ts.source))),
        "n_condition_rows": len(rows),
        "n_condition_rows_layout_verified": sum(1 for r in rows if r["layout_verified"]),
        "layout_verified_files": sorted(tay.VERIFIED_LAYOUT_FILES),
        "layout_unverified_note": (
            "cytokine-block and sepsis deposit files use a different experimental design; "
            "their metadata columns do not follow the convention confirmed for the other "
            "files, so they are excluded from all analyses"),
        "n_wells_total": int(len(set(ts.group))),
        "frame_interval_min": tay.FRAME_MIN,
        "trace_length_frames": tay.N_TRACE,
        "column_layout_source": "read from rawtraceprocessing.m in the deposit",
        "grouping_unit": "source_file | replicate | well (col 130)",
    })
    write_json("b31_tolerance.json", b31_tolerance(ts))
    write_json("b32_ligand_identity.json", b32_ligand_identity(ts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
