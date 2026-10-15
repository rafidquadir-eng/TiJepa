"""Grouped CV, cluster bootstrap, and permutation tests (PLAN.md 8.1)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    metric=balanced_accuracy_score,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Resample whole groups (movies/wells), not cells (PLAN.md 1.5, 8.1)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    point = float(metric(y_true, y_pred))
    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            vals.append(float(metric(yt, yp)))
        except ValueError:
            continue
    vals = np.asarray(vals)
    if vals.size == 0:
        return {"point": point, "ci_low": None, "ci_high": None, "n_boot_valid": 0}
    return {
        "point": point,
        "ci_low": float(np.quantile(vals, alpha / 2)),
        "ci_high": float(np.quantile(vals, 1 - alpha / 2)),
        "n_boot_valid": int(vals.size),
    }


def permutation_p(
    fit_predict,
    y: np.ndarray,
    groups: np.ndarray,
    observed: float,
    n_perm: int = 1000,
    seed: int = 0,
    metric=balanced_accuracy_score,
) -> dict:
    """Permute labels at the GROUP level, preserving group structure.

    fit_predict(y_permuted) -> predictions aligned to y.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    # one label per group (labels are constant within a well/movie by design)
    glab = {g: y[groups == g][0] for g in uniq}
    null = []
    for _ in range(n_perm):
        perm = rng.permutation([glab[g] for g in uniq])
        mapping = dict(zip(uniq, perm))
        y_perm = np.array([mapping[g] for g in groups])
        try:
            null.append(float(metric(y_perm, fit_predict(y_perm))))
        except Exception:
            continue
    null = np.asarray(null)
    if null.size == 0:
        return {"p_value": None, "n_perm_valid": 0}
    # +1 correction: p is never reported as exactly 0
    p = (1.0 + float((null >= observed).sum())) / (1.0 + null.size)
    return {
        "p_value": float(p),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "n_perm_valid": int(null.size),
    }


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    metric=balanced_accuracy_score,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired cluster bootstrap of metric(A) - metric(B) on identical resamples."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    diffs = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            diffs.append(float(metric(yt, pred_a[idx])) - float(metric(yt, pred_b[idx])))
        except ValueError:
            continue
    diffs = np.asarray(diffs)
    if diffs.size == 0:
        return {"delta": None, "n_boot_valid": 0}
    return {
        "delta": float(metric(y_true, pred_a) - metric(y_true, pred_b)),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
        "n_boot_valid": int(diffs.size),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_score: np.ndarray | None = None) -> dict:
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "n": int(len(y_true)),
        "n_classes": int(len(np.unique(y_true))),
    }
    if y_score is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                s = y_score if y_score.ndim == 1 else y_score[:, 1]
                out["auroc"] = float(roc_auc_score(y_true, s))
            else:
                out["auroc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
                )
        except (ValueError, IndexError):
            out["auroc"] = None
    return out


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out
