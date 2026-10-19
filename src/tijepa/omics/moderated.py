"""Empirical-Bayes moderated t-tests (the limma method) in pure Python.

PLAN_v1.md 7.5 and Section 9 require moderated tests for the human and anchor
contrasts, with paired or blocked designs where donors are shared. Bioconductor
`limma` is the reference implementation and `inmoose.limma` is its Python port, but
`inmoose` needs a C++ extension that will not build on this instance (no `Python.h`).
The plan's third option is "a Python empirical-Bayes implementation validated in tests
against a known example", which is what this module is. `tests/test_moderated.py`
checks it against hand-computable cases and against the limiting behaviour that defines
the method.

### The method (Smyth 2004, *Stat Appl Genet Mol Biol* 3:3)

For each gene g, an ordinary linear model gives a coefficient estimate `b_g`, an
unscaled standard error factor `v_g` and a residual variance `s_g^2` on `d_g` degrees
of freedom. Rather than trusting each gene's own variance, limma assumes

    s_g^2 ~ (s_0^2) * chi^2_{d_0} / d_0

and estimates the hyperparameters `d_0`, `s_0^2` from the observed spread of
`log(s_g^2)` by matching the first two moments of the Fisher z-distribution:

    e_g   = log(s_g^2) - digamma(d_g/2) + log(d_g/2)
    var_e = mean((e_g - mean(e_g))^2) * n/(n-1) - mean(trigamma(d_g/2))

`d_0` solves `trigamma(d_0/2) = var_e`, and `s_0^2` follows from the mean. The
posterior variance shrinks each gene towards the prior,

    s~_g^2 = (d_0 s_0^2 + d_g s_g^2) / (d_0 + d_g),

and the moderated t statistic is `b_g / (s~_g sqrt(v_g))` on `d_0 + d_g` degrees of
freedom. When `d_0 -> 0` this reduces to the ordinary t-test, and when `d_0 -> inf` to
a z-test against a common variance; both limits are asserted in the tests.

This matters here because the contrasts have 4-7 donors. An ordinary per-gene t-test on
n = 4 is dominated by genes that happen to have a small sample variance, which is
precisely the failure mode that produced v0's non-canonical LPS signature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats


def _trigamma(x):
    """Trigamma function. scipy exposes it only as polygamma(1, .)."""
    return special.polygamma(1, x)


@dataclass
class ModeratedResult:
    coefficient: np.ndarray      # effect size per gene (e.g. mean paired difference)
    t: np.ndarray                # moderated t statistic
    p_value: np.ndarray
    q_value: np.ndarray          # Benjamini-Hochberg FDR
    df_total: np.ndarray         # d_0 + d_g
    s2_post: np.ndarray          # posterior variance
    d0: float
    s2_prior: float
    n_genes: int
    n_samples: int

    def to_frame(self, index):
        import pandas as pd
        return pd.DataFrame({
            "coefficient": self.coefficient, "t": self.t,
            "p_value": self.p_value, "q_value": self.q_value,
            "df_total": self.df_total,
        }, index=index)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    pv = p[ok]
    n = pv.size
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def fit_prior(s2: np.ndarray, df: np.ndarray) -> tuple[float, float]:
    """Estimate (d_0, s_0^2) from the spread of log residual variances."""
    good = np.isfinite(s2) & (s2 > 0) & (df > 0)
    s2, df = s2[good], df[good]
    if s2.size < 2:
        return 0.0, float(np.mean(s2)) if s2.size else 1.0
    z = np.log(s2)
    e = z - special.digamma(df / 2.0) + np.log(df / 2.0)
    n = e.size
    var_e = np.sum((e - e.mean()) ** 2) / (n - 1) - np.mean(_trigamma(df / 2.0))
    if var_e <= 0:
        # no excess spread: the variances look like a single common value
        return np.inf, float(np.exp(e.mean()))

    def f(x: float) -> float:
        return _trigamma(np.exp(x) / 2.0) - var_e

    # trigamma is decreasing, so bracket d_0 on a log scale
    try:
        lo, hi = -12.0, 12.0
        if f(lo) * f(hi) > 0:
            d0 = 0.0 if f(hi) > 0 else np.inf
        else:
            d0 = float(np.exp(optimize.brentq(f, lo, hi, xtol=1e-8)))
    except Exception:
        d0 = 0.0
    if not np.isfinite(d0) or d0 <= 0:
        return (0.0, float(np.exp(e.mean()))) if d0 == 0 else (np.inf,
                                                               float(np.exp(e.mean())))
    s2_prior = float(np.exp(e.mean() + special.digamma(d0 / 2.0) - np.log(d0 / 2.0)))
    return d0, s2_prior


def moderated_t(coef: np.ndarray, s2: np.ndarray, df: np.ndarray,
                unscaled_var: np.ndarray | float) -> ModeratedResult:
    """Moderated t from per-gene coefficients, residual variances and df."""
    coef = np.asarray(coef, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    df = np.asarray(df, dtype=float)
    v = np.asarray(unscaled_var, dtype=float)
    if v.ndim == 0:
        v = np.full(coef.shape, float(v))

    d0, s2_prior = fit_prior(s2, df)
    if np.isinf(d0):
        s2_post = np.full(coef.shape, s2_prior)
        df_total = np.full(coef.shape, np.inf)
    else:
        s2_post = (d0 * s2_prior + df * s2) / (d0 + df)
        df_total = df + d0
    se = np.sqrt(s2_post * v)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = coef / se
    p = np.where(np.isfinite(df_total),
                 2.0 * stats.t.sf(np.abs(t), np.where(np.isfinite(df_total),
                                                      df_total, 1.0)),
                 2.0 * stats.norm.sf(np.abs(t)))
    p = np.clip(np.where(np.isfinite(t), p, np.nan), 0.0, 1.0)
    return ModeratedResult(
        coefficient=coef, t=t, p_value=p, q_value=bh_fdr(p), df_total=df_total,
        s2_post=s2_post, d0=float(d0), s2_prior=float(s2_prior),
        n_genes=int(coef.size), n_samples=0)


def paired_moderated_test(treated: np.ndarray, control: np.ndarray) -> ModeratedResult:
    """Moderated paired test on matched columns: genes x donors, treated vs control.

    `treated[:, i]` and `control[:, i]` must be the same donor. The model is a
    one-sample test on the per-donor differences, which is the paired design; using an
    unpaired test here discards the pairing and loses most of the power, as v0 found.
    """
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)
    if treated.shape != control.shape:
        raise ValueError(f"shape mismatch {treated.shape} vs {control.shape}")
    n = treated.shape[1]
    if n < 2:
        raise ValueError("need at least 2 donor pairs")
    d = treated - control
    coef = np.nanmean(d, axis=1)
    s2 = np.nanvar(d, axis=1, ddof=1)
    df = np.full(coef.shape, float(n - 1))
    res = moderated_t(coef, s2, df, unscaled_var=1.0 / n)
    res.n_samples = n
    return res


def blocked_moderated_test(x: np.ndarray, group: np.ndarray,
                           block: np.ndarray) -> ModeratedResult:
    """Two-group contrast with a block (donor) factor, via least squares per gene.

    Equivalent to limma's `~0 + group + block` design. Donors present in only one
    group contribute to the intercept but not to the pairing, so the effective design
    is reported through `df_total`.
    """
    x = np.asarray(x, dtype=float)
    group = np.asarray(group)
    block = np.asarray(block)
    g_levels = np.unique(group)
    if g_levels.size != 2:
        raise ValueError(f"expected 2 groups, got {g_levels.tolist()}")
    b_levels = np.unique(block)
    # design: intercept + group indicator + block indicators (first block absorbed)
    cols = [np.ones(x.shape[1]), (group == g_levels[1]).astype(float)]
    for b in b_levels[1:]:
        cols.append((block == b).astype(float))
    design = np.column_stack(cols)
    n, p = design.shape
    if n - p < 1:
        raise ValueError(f"design is saturated: {n} samples, {p} parameters")
    pinv = np.linalg.pinv(design)
    beta = x @ pinv.T                      # genes x parameters
    resid = x - beta @ design.T
    s2 = (resid ** 2).sum(axis=1) / (n - p)
    xtx_inv = np.linalg.pinv(design.T @ design)
    res = moderated_t(beta[:, 1], s2, np.full(x.shape[0], float(n - p)),
                      unscaled_var=float(xtx_inv[1, 1]))
    res.n_samples = n
    return res
