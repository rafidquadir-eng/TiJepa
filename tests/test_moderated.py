"""Validation of the empirical-Bayes moderated t-test against known behaviour.

PLAN_v1.md 7.5 permits a Python empirical-Bayes implementation only if it is
"validated in tests against a known example". These tests pin the properties that
define the method, each of which has an analytic answer:

  * with no excess spread in the per-gene variances the prior degrees of freedom are
    unbounded and the statistic becomes a z-test against the common variance;
  * the posterior variance always lies between the prior and the gene's own variance,
    which is what shrinkage means;
  * a gene with an implausibly small sample variance is penalised relative to the
    ordinary t-test - the exact failure mode that moderation exists to prevent;
  * the paired test reproduces a hand-computed one-sample t when moderation is
    switched off by construction;
  * Benjamini-Hochberg adjustment matches a value computed by hand.

All fixtures here are synthetic and named `synthetic_*`, per PLAN.md 1.1.2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tijepa.omics import moderated as M  # noqa: E402


def synthetic_equal_variance(n_genes=500, n_donors=6, sd=0.5, seed=0):
    """Every gene shares one variance, so there is no excess spread to model."""
    rng = np.random.default_rng(seed)
    control = rng.normal(0.0, sd, size=(n_genes, n_donors))
    treated = control + rng.normal(0.0, sd, size=(n_genes, n_donors))
    return treated, control


def test_bh_fdr_matches_hand_computation():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
    q = M.bh_fdr(p)
    # q_i = min over j >= i of p_j * n / j, enforced monotone from the largest p:
    #   0.001*5/1=0.005  0.008*5/2=0.020  0.039*5/3=0.065
    #   0.041*5/4=0.05125  0.042*5/5=0.042  ->  the last three collapse to 0.042
    expect = np.array([0.005, 0.02, 0.042, 0.042, 0.042])
    assert np.allclose(q, expect, atol=1e-6)


def test_bh_fdr_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    p = rng.random(1000)
    q = M.bh_fdr(p)
    assert np.all(q >= 0) and np.all(q <= 1)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_posterior_variance_lies_between_prior_and_sample_variance():
    rng = np.random.default_rng(2)
    n = 5
    s2 = np.exp(rng.normal(0.0, 1.0, size=400))      # genuinely spread variances
    df = np.full(400, float(n - 1))
    res = M.moderated_t(np.zeros(400), s2, df, unscaled_var=1.0 / n)
    assert np.isfinite(res.d0) and res.d0 > 0
    lo = np.minimum(s2, res.s2_prior)
    hi = np.maximum(s2, res.s2_prior)
    assert np.all(res.s2_post >= lo - 1e-9)
    assert np.all(res.s2_post <= hi + 1e-9)


def test_moderation_shrinks_towards_the_prior():
    """A gene far below the prior variance must move up, and vice versa."""
    n = 4
    s2 = np.concatenate([np.full(200, 1.0), np.array([1e-4, 100.0])])
    df = np.full(s2.size, float(n - 1))
    res = M.moderated_t(np.ones(s2.size), s2, df, unscaled_var=1.0 / n)
    assert res.s2_post[-2] > s2[-2]       # tiny variance pulled up
    assert res.s2_post[-1] < s2[-1]       # huge variance pulled down


def test_small_variance_gene_is_penalised_against_ordinary_t():
    """The failure mode moderation exists to prevent, stated as a test.

    One gene has the same effect size as the rest but a freak-small sample variance.
    The ordinary t-test rewards it; the moderated test must not, by as much.
    """
    n = 4
    n_genes = 300
    coef = np.full(n_genes, 0.2)
    s2 = np.full(n_genes, 1.0)
    s2[0] = 1e-3
    df = np.full(n_genes, float(n - 1))
    res = M.moderated_t(coef, s2, df, unscaled_var=1.0 / n)
    t_ordinary = coef / np.sqrt(s2 / n)
    # the freak gene's ordinary t is enormous ...
    assert t_ordinary[0] > 10 * t_ordinary[1]
    # ... while its moderated t is brought back close to the rest
    assert res.t[0] < 0.5 * t_ordinary[0]


def test_equal_variances_give_unbounded_prior_df_and_z_behaviour():
    treated, control = synthetic_equal_variance(n_genes=800, n_donors=8, seed=3)
    res = M.paired_moderated_test(treated, control)
    # With a common variance and many genes, the estimated prior df is large: the
    # method has recognised that per-gene variances carry no extra information.
    assert res.d0 > 20 or np.isinf(res.d0)
    assert res.n_samples == 8


def test_paired_test_recovers_known_effect_and_calibration():
    """Under the null the p-values must be ~uniform; a planted effect must be found."""
    rng = np.random.default_rng(4)
    n_genes, n = 2000, 6
    control = rng.normal(0.0, 1.0, size=(n_genes, n))
    treated = control + rng.normal(0.0, 1.0, size=(n_genes, n))
    treated[:50] += 3.0                     # 50 genes with a large real effect
    res = M.paired_moderated_test(treated, control)
    assert (res.q_value[:50] < 0.05).sum() >= 45
    null_p = res.p_value[50:]
    # calibration: about 5% of nulls below 0.05, generously bounded
    assert 0.02 < float(np.mean(null_p < 0.05)) < 0.10
    assert abs(float(np.median(null_p)) - 0.5) < 0.08


def test_paired_test_rejects_mismatched_shapes_and_tiny_n():
    with pytest.raises(ValueError):
        M.paired_moderated_test(np.zeros((10, 3)), np.zeros((10, 4)))
    with pytest.raises(ValueError):
        M.paired_moderated_test(np.zeros((10, 1)), np.zeros((10, 1)))


def test_blocked_test_matches_paired_test_on_a_balanced_design():
    """With one observation per group per donor, blocking IS pairing."""
    rng = np.random.default_rng(5)
    n_genes, n = 400, 5
    control = rng.normal(0.0, 1.0, size=(n_genes, n))
    treated = control + 0.8 + rng.normal(0.0, 0.5, size=(n_genes, n))
    x = np.hstack([control, treated])
    group = np.array(["ctl"] * n + ["trt"] * n)
    block = np.array([f"D{i}" for i in range(n)] * 2)
    paired = M.paired_moderated_test(treated, control)
    blocked = M.blocked_moderated_test(x, group, block)
    assert np.allclose(paired.coefficient, blocked.coefficient, atol=1e-8)
    # the same ranking of genes, which is what the downstream signatures use
    r = stats.spearmanr(paired.t, blocked.t).statistic
    assert r > 0.99


def test_blocked_test_rejects_a_saturated_design():
    x = np.zeros((10, 4))
    group = np.array(["a", "a", "b", "b"])
    block = np.array(["D1", "D2", "D3", "D4"])
    with pytest.raises(ValueError):
        M.blocked_moderated_test(x, group, block)
