"""Grouped linear probes and cluster-bootstrap statistics shared by the v3 benchmarks.

The probe is the BBBC054 one (StandardScaler -> PCA(64) if wider -> StandardScaler ->
class-balanced logistic regression), fitted with StratifiedGroupKFold so no group (FOV,
sample, well) is in both the training and the test folds. Confidence intervals are
cluster bootstraps over groups and are reported only with >= 10 groups (PLAN_v1 rule 3).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PCA_DIM = 64
MIN_GROUPS_FOR_CI = 10


def pipeline(n_features: int, seed: int, clf):
    steps = [StandardScaler()]
    if n_features > PCA_DIM:
        steps += [PCA(n_components=PCA_DIM, random_state=seed), StandardScaler()]
    return make_pipeline(*steps, clf)


def grouped_predict(
    z: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int = 5
) -> np.ndarray:
    z = np.asarray(z, np.float64)
    z = z[:, np.isfinite(z).all(0)]
    pred = np.empty(len(y), dtype=y.dtype)
    n_splits = min(n_splits, len(np.unique(groups)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(z, y, groups):
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        clf = LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0, random_state=seed
        )
        m = pipeline(z.shape[1], seed, clf).fit(z[tr], y[tr])
        pred[te] = m.predict(z[te])
    return pred


def grouped_regress(
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_splits: int = 5,
    alpha: float = 10.0,
) -> np.ndarray:
    z = np.asarray(z, np.float64)
    z = z[:, np.isfinite(z).all(0)]
    pred = np.empty(len(y), dtype=float)
    n_splits = min(n_splits, len(np.unique(groups)))
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    remap = dict(zip(ug, rng.permutation(len(ug)), strict=False))
    g2 = np.array([remap[g] for g in groups])
    for tr, te in GroupKFold(n_splits=n_splits).split(z, y, g2):
        m = pipeline(z.shape[1], seed, Ridge(alpha=alpha)).fit(z[tr], y[tr])
        pred[te] = m.predict(z[te])
    return pred


def cluster_bootstrap(
    y, pred, groups, stat=balanced_accuracy_score, n_boot: int = 2000, seed: int = 0
) -> dict:
    ug = np.unique(groups)
    if len(ug) < MIN_GROUPS_FOR_CI:
        return {"ci95": None, "reason": f"{len(ug)} groups < {MIN_GROUPS_FOR_CI}"}
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        take = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        if len(np.unique(y[take])) < 2:
            continue
        vals.append(stat(y[take], pred[take]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"ci95": [float(lo), float(hi)], "n_groups": int(len(ug)), "n_boot": n_boot}


def paired_bootstrap(
    y, pred_a, pred_b, groups, stat=balanced_accuracy_score, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Cluster bootstrap of stat(a) - stat(b), both scored on the same resample."""
    ug = np.unique(groups)
    obs = float(stat(y, pred_a) - stat(y, pred_b))
    if len(ug) < MIN_GROUPS_FOR_CI:
        return {
            "difference": obs,
            "ci95": None,
            "p_two_sided": None,
            "reason": f"{len(ug)} groups < {MIN_GROUPS_FOR_CI}",
        }
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        take = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        if len(np.unique(y[take])) < 2:
            continue
        d.append(stat(y[take], pred_a[take]) - stat(y[take], pred_b[take]))
    d = np.asarray(d)
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {
        "difference": obs,
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_two_sided": p,
        "n_groups": int(len(ug)),
    }


def bh(pvals: list[float | None]) -> list[float | None]:
    idx = [i for i, p in enumerate(pvals) if p is not None]
    if not idx:
        return list(pvals)
    p = np.array([pvals[i] for i in idx])
    order = np.argsort(p)
    m = len(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    out = list(pvals)
    for j, i in enumerate(idx):
        out[i] = float(q[j])
    return out


def effective_rank(z: np.ndarray) -> float:
    z = np.asarray(z, np.float64)
    z = z - z.mean(0)
    s = np.linalg.svd(z, compute_uv=False) ** 2
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))
