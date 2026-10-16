"""F4 7.2-7.3: acute endotoxin tolerance from NF-kB dynamics, and ligand identity.

**B3.1 (7.2) acute endotoxin tolerance.** Can single-cell NF-kB dynamics tell a
previously LPS-exposed macrophage from a naive one, when both are challenged with LPS?
The wording matters and is used consistently: this is **acute endotoxin tolerance**
(LPS then LPS, 8 h apart, mouse BMDM), not trained immunity and not the 6-day human
tolerance of Part B4 (review finding B4).

The design supplies 2 LPS->LPS wells and 2 naive->LPS wells at the matched 8 h interval
(every layout-verified file was searched; see `ligand_codes.json`). With 4 grouping
units, PLAN_v1.md rule 3 and Section 9 forbid a bootstrap CI, which is what v0 reported.
Instead this script gives per-well predictions, the **exact** permutation distribution
over all distinct well-label assignments, and the minimum achievable p. With 2 wells
per group there are only 6 distinct assignments, so no p below 1/6 = 0.167 exists and
the design cannot reach significance however good the classifier is. That is stated
plainly rather than worked around.

A secondary analysis adds the 6 further naive->LPS wells found in other files and at
the 4 h interval. It has more groups but confounds priming with both deposit file and
interval, so it is reported as a sensitivity check, labelled as such, and is not the
primary endpoint.

**B3.2 (7.3) ligand identity.** Multi-class classification of the (first, second)
ligand pair, with fold assignment varying per seed so that reported seed variance is
real (rule 4). The class set differs from the published one, which is noted.
"""
from __future__ import annotations

import functools
import itertools
import json
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from tijepa.io import tay  # noqa: E402

OUT = REPO / "results" / "v1" / "partB"
LPS, NAIVE = 2.0, 0.0
SEEDS = (0, 1, 2)


def handcrafted(win: np.ndarray) -> np.ndarray:
    """Interpretable features of a baseline-corrected second-stimulus response."""
    w = np.nan_to_num(win, nan=0.0)
    t = np.arange(w.shape[1])
    peak = w.max(axis=1)
    tpeak = w.argmax(axis=1).astype(float)
    auc = w.sum(axis=1)
    early = w[:, : max(1, w.shape[1] // 3)].sum(axis=1)
    late = w[:, -max(1, w.shape[1] // 3):].sum(axis=1)
    mean = w.mean(axis=1)
    sd = w.std(axis=1)
    # slope of a least-squares line, as a crude decay/rise rate
    tc = t - t.mean()
    slope = (w * tc).sum(axis=1) / max(float((tc ** 2).sum()), 1e-9)
    halfmax = (w > (peak[:, None] / 2.0)).sum(axis=1).astype(float)
    return np.column_stack([peak, tpeak, auc, early, late, mean, sd, slope, halfmax,
                            np.divide(late, np.abs(early) + 1e-6)])


def loo_group_predict(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      seed: int = 0) -> np.ndarray:
    """Leave-one-group-out predictions; the only honest CV when groups are this few."""
    pred = np.zeros(len(y), dtype=int)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            pred[te] = int(round(float(y[tr].mean())))
            continue
        sc = StandardScaler().fit(x[tr])
        clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced",
                                 random_state=seed)
        clf.fit(sc.transform(x[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(x[te]))
    return pred


def well_level_score(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> float:
    """Balanced accuracy over WELL-level majority votes, the grouping unit."""
    gy, gp = [], []
    for g in np.unique(groups):
        m = groups == g
        gy.append(int(round(float(y[m].mean()))))
        gp.append(int(round(float(pred[m].mean()))))
    return float(balanced_accuracy_score(gy, gp))


def exact_permutation(y: np.ndarray, pred_fn, groups: np.ndarray) -> dict:
    """Exact permutation over distinct well-label assignments.

    The observed statistic is the cell-level balanced accuracy; the null is built by
    relabelling WELLS, which is the unit of the design. Every distinct assignment is
    enumerated, so the resulting p is exact and its floor is known.
    """
    uniq = np.unique(groups)
    gy = np.array([int(round(float(y[groups == g].mean()))) for g in uniq])
    n_pos = int(gy.sum())
    obs = pred_fn(y)
    stats = []
    for combo in itertools.combinations(range(len(uniq)), n_pos):
        lab = np.zeros(len(uniq), dtype=int)
        lab[list(combo)] = 1
        y_perm = np.zeros_like(y)
        for gi, g in enumerate(uniq):
            y_perm[groups == g] = lab[gi]
        stats.append(pred_fn(y_perm))
    stats_arr = np.asarray(stats)
    return {
        "observed": float(obs),
        "n_distinct_assignments": int(len(stats)),
        "p_exact": float(np.mean(stats_arr >= obs)),
        "minimum_achievable_p": float(1.0 / len(stats)),
        "null_mean": float(stats_arr.mean()),
        "null_max": float(stats_arr.max()),
        "note": "Exact enumeration over all distinct well-label assignments "
                "(PLAN_v1.md Section 9). No bootstrap CI is reported, because "
                f"{len(uniq)} groups is below the 10-group minimum (rule 3).",
    }


def b31(ts: tay.TraceSet) -> dict:
    out: dict = {
        "name": "B3.1 acute endotoxin tolerance from single-cell NF-kB dynamics",
        "framing": "ACUTE ENDOTOXIN TOLERANCE: LPS then LPS, 8 h apart, mouse BMDM. "
                   "Not trained immunity; not the 6-day human tolerance of B4.",
        "primary_endpoint": True,
    }
    v = ts.layout_verified
    sel = v & (ts.second == LPS) & np.isin(ts.first, [LPS, NAIVE]) & (ts.gap == 8.0)
    sub = ts.subset(sel)
    if len(sub) == 0:
        return {**out, "status": "no_cells"}

    win, i0, i1 = tay.response_window(sub, 8.0)
    y = (sub.first == LPS).astype(int)
    groups = sub.group
    uniq = np.unique(groups)
    out.update({
        "n_cells": int(len(sub)), "n_wells": int(len(uniq)),
        "wells": sorted(uniq.tolist()),
        "n_cells_tolerant": int(y.sum()), "n_cells_naive": int((1 - y).sum()),
        "response_window_frames": [int(i0), int(i1)],
        "interval_h": 8.0,
    })

    results = {}
    for kind, x in (("raw_trace", np.nan_to_num(win, nan=0.0)),
                    ("handcrafted", handcrafted(win))):
        per_seed, per_well_all = [], {}
        for seed in SEEDS:
            # Leave-one-well-out with a fixed C is deterministic, so a seed that only
            # sets `random_state` would give sd exactly 0 and the seed claim would have
            # to be removed (PLAN_v1.md rule 4). The real source of randomness here is
            # a different stratified per-well cell subsample per seed (F3.5).
            rng = np.random.default_rng(seed)
            sel = subsample_per_group(groups, rng, per_group=250)
            xs, ys, gs = x[sel], y[sel], groups[sel]
            pred = loo_group_predict(xs, ys, gs, seed=seed)
            per_seed.append({
                "seed": seed, "n_cells_subsampled": int(len(ys)),
                "cell_balanced_accuracy": float(balanced_accuracy_score(ys, pred)),
                "well_balanced_accuracy": well_level_score(ys, pred, gs),
            })
            for g in uniq:
                m = gs == g
                if not m.any():
                    continue
                per_well_all.setdefault(g, []).append({
                    "seed": seed, "n_cells": int(m.sum()),
                    "true_label": "LPS->LPS" if ys[m][0] == 1 else "naive->LPS",
                    "fraction_predicted_tolerant": float(pred[m].mean()),
                })
        ba = np.array([r["cell_balanced_accuracy"] for r in per_seed])

        def stat(y_lab: np.ndarray, _x=x, _g=groups) -> float:
            return float(balanced_accuracy_score(
                y_lab, loo_group_predict(_x, y_lab, _g, seed=0)))

        perm = exact_permutation(y, stat, groups)
        results[kind] = {
            "per_seed": per_seed,
            "cell_balanced_accuracy_mean": float(ba.mean()),
            "cell_balanced_accuracy_sd": float(ba.std(ddof=1)) if len(ba) > 1 else 0.0,
            "per_well": {k: v2 for k, v2 in per_well_all.items()},
            "exact_permutation": perm,
            "confidence_interval": None,
            "confidence_interval_reason":
                f"{len(uniq)} grouping units is below the 10-group minimum "
                "(PLAN_v1.md rule 3), so no bootstrap CI is reported. v0 reported one.",
            "seed_varies": "a different stratified per-well cell subsample per seed "
                           "(F3.5); the protocol is otherwise deterministic, so "
                           "without this the sd would be exactly 0 and the seed claim "
                           "would have to be withdrawn under rule 4.",
        }
        sd = results[kind]["cell_balanced_accuracy_sd"]
        print(f"  B3.1 {kind:12s} BA {ba.mean():.3f} (sd {sd:.3f}) "
              f"exact p={perm['p_exact']:.3f} "
              f"(floor {perm['minimum_achievable_p']:.3f})")

    out["results"] = results
    out["interpretation"] = (
        f"The design supplies {len(uniq)} wells, so the exact permutation null has "
        f"{results['raw_trace']['exact_permutation']['n_distinct_assignments']} "
        "distinct assignments and no p below "
        f"{results['raw_trace']['exact_permutation']['minimum_achievable_p']:.3f} is "
        "achievable. The contrast therefore CANNOT reach conventional significance, "
        "whatever the classifier does. This is a limitation of the public deposit, not "
        "of the analysis.")
    out["status"] = "ok"
    return out


def b31_sensitivity(ts: tay.TraceSet) -> dict:
    """All naive->LPS wells vs all LPS->LPS wells: more groups, more confounding."""
    v = ts.layout_verified
    sel = v & (ts.second == LPS) & np.isin(ts.first, [LPS, NAIVE])
    sub = ts.subset(sel)
    if len(sub) == 0:
        return {"status": "no_cells"}
    # the response window depends on the interval, so build it per interval
    xs, ys, gs = [], [], []
    for gap in np.unique(sub.gap):
        m = sub.gap == gap
        s2 = sub.subset(m)
        win, _, _ = tay.response_window(s2, float(gap), length=20)
        xs.append(handcrafted(win))
        ys.append((s2.first == LPS).astype(int))
        gs.append(s2.group)
    x = np.vstack(xs)
    y = np.concatenate(ys)
    groups = np.concatenate(gs)
    uniq = np.unique(groups)
    per_seed = []
    for seed in SEEDS:
        pred = loo_group_predict(x, y, groups, seed=seed)
        per_seed.append({"seed": seed,
                         "cell_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                         "well_balanced_accuracy": well_level_score(y, pred, groups)})
    ba = np.array([r["cell_balanced_accuracy"] for r in per_seed])

    def stat(y_lab: np.ndarray) -> float:
        return float(balanced_accuracy_score(
            y_lab, loo_group_predict(x, y_lab, groups, seed=0)))

    perm = exact_permutation(y, stat, groups)
    print(f"  B3.1 sensitivity (all wells) BA {ba.mean():.3f} over {len(uniq)} wells, "
          f"exact p={perm['p_exact']:.3f}")
    return {
        "status": "ok",
        "label": "SENSITIVITY ANALYSIS, NOT the primary endpoint: it pools naive->LPS "
                 "wells across deposit files and across the 4 h and 8 h intervals, so "
                 "priming is confounded with both file and interval. Reported because "
                 "the matched design has only 4 wells, not because it is cleaner.",
        "n_wells": int(len(uniq)), "n_cells": int(len(y)),
        "wells": sorted(uniq.tolist()),
        "per_seed": per_seed,
        "cell_balanced_accuracy_mean": float(ba.mean()),
        "cell_balanced_accuracy_sd": float(ba.std(ddof=1)),
        "exact_permutation": perm,
        "confidence_interval": None,
        "confidence_interval_reason":
            f"{len(uniq)} grouping units is still below the 10-group minimum.",
    }


def subsample_per_group(groups: np.ndarray, rng: np.random.Generator,
                        per_group: int) -> np.ndarray:
    """Indices of up to `per_group` cells per well, drawn with `rng`.

    Drawing a different subsample per seed is one of the real sources of randomness
    PLAN_v1.md F3.5 requires, so the reported sd is not cosmetic. It also keeps the
    multi-class fit tractable: the full set is tens of thousands of cells.
    """
    keep = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) > per_group:
            idx = rng.choice(idx, per_group, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def b32(ts: tay.TraceSet, per_well: int = 120) -> dict:
    """Multi-class (first, second) ligand-pair identity from the full trace."""
    v = ts.layout_verified
    sub = ts.subset(v)
    named = {k: v2 for k, v2 in tay.LIGAND_CODES.items()}

    def name(c: float) -> str:
        if c == 0.0:
            return "naive"
        return named.get(c, f"code{int(c)}")

    labels = np.array([f"{name(f)}->{name(s)}"
                       for f, s in zip(sub.first, sub.second, strict=False)])
    keep = np.array([("code" not in lb) for lb in labels])
    sub2, labels = sub.subset(keep), labels[keep]
    x = np.nan_to_num(sub2.traces, nan=0.0)
    groups = sub2.group
    classes, y = np.unique(labels, return_inverse=True)

    per_seed = []
    for seed in SEEDS:
        # Two real sources of randomness per seed (rule 4 / F3.5): a different
        # stratified cell subsample, and a different fold assignment.
        rng = np.random.default_rng(seed)
        sel = subsample_per_group(groups, rng, per_well)
        xs, ys, gs = x[sel], y[sel], groups[sel]
        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        pred = np.zeros(len(ys), dtype=int)
        for tr, te in cv.split(xs, ys, gs):
            sc = StandardScaler().fit(xs[tr])
            clf = LogisticRegression(max_iter=1500, C=1.0, class_weight="balanced",
                                     random_state=seed)
            clf.fit(sc.transform(xs[tr]), ys[tr])
            pred[te] = clf.predict(sc.transform(xs[te]))
        per_seed.append({"seed": seed, "n_cells_subsampled": int(len(ys)),
                         "balanced_accuracy": float(balanced_accuracy_score(ys, pred))})
    ba = np.array([r["balanced_accuracy"] for r in per_seed])
    print(f"  B3.2 ligand identity: {len(classes)} classes, BA "
          f"{ba.mean():.3f} +/- {ba.std(ddof=1):.3f} (chance {1/len(classes):.3f})")
    return {
        "status": "ok",
        "name": "B3.2 ligand-pair identity from the NF-kB trace",
        "primary_endpoint": False,
        "n_classes": int(len(classes)),
        "classes": classes.tolist(),
        "n_cells": int(len(y)), "n_wells": int(len(np.unique(groups))),
        "chance_balanced_accuracy": float(1.0 / len(classes)),
        "per_seed": per_seed,
        "balanced_accuracy_mean": float(ba.mean()),
        "balanced_accuracy_sd": float(ba.std(ddof=1)),
        "cells_per_well_subsampled": per_well,
        "seed_varies": "a different stratified per-well cell subsample AND a different "
                       "StratifiedGroupKFold fold assignment per seed, so the sd "
                       "reflects real variation rather than being cosmetic",
        "class_set_note": "Only ligand codes resolvable from the deposit (CpG, LPS, "
                          "polyI:C, TNF and the naive control) are used, so this class "
                          "set differs from the published one, which also covers the "
                          "unresolved codes 4, 5, 7 and 17.",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = tay.load(REPO)
    print(f"loaded {len(ts)} cells from {len(set(ts.source.tolist()))} deposit files")

    res = {
        "phase": "F4 7.2-7.3",
        "positive_controls": {
            "rna": "results/v1/partB/rna_positive_control.json",
            "atac": "results/v1/partB/atac_positive_control.json",
            "traces": "results/v1/partB/ligand_codes.json (tolerance direction)",
        },
        "b31_tolerance": b31(ts),
        "b31_sensitivity_all_wells": b31_sensitivity(ts),
        "b32_ligand_identity": b32(ts),
    }
    (OUT / "b3_dynamics.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(f"wrote {OUT/'b3_dynamics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
