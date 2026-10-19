"""v3 evaluation logic: forecasting controls, re-identification, association IDF1,
sample-identity QC and power (PLAN_v3 H8.4)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.eval import forecast_controls as FC  # noqa: E402
from tijepa.eval import probe as P  # noqa: E402
from tijepa.eval import reid as R  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), REPO / "scripts" / name
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- forecasting controls
def _synthetic(kind: str, n_movies: int = 6, n: int = 40, seed: int = 0):
    """Target latents = slot (position) component + clip-specific content that moves.

    A 'position' model predicts only the slot component; a 'dynamics' model predicts
    the content's future from its context (here: the true future, plus noise).
    """
    rng = np.random.default_rng(seed)
    S, Pn, D = 2, 4, 8
    slot = rng.normal(0, 2, (S, 1, D))
    per_movie, static = {}, {}
    stores = {}
    for m in range(n_movies):
        content = rng.normal(0, 1, (n, 1, Pn, D))
        drift = rng.normal(0, 1, (n, S, Pn, D)) * 0.8
        tgt = slot[None] + content + drift
        last = content[:, 0] + rng.normal(0, 0.1, (n, Pn, D))
        copy = np.repeat(last[:, None], S, axis=1)
        if kind == "dynamics":
            pred = tgt + rng.normal(0, 0.2, tgt.shape)
        else:
            pred = np.broadcast_to(slot[None] + content, tgt.shape) + rng.normal(
                0, 0.2, tgt.shape
            )
        perm = np.roll(np.arange(n), 1)
        swap = (
            pred[perm]
            if kind == "dynamics"
            else (slot[None] + content[perm] + rng.normal(0, 0.2, tgt.shape))
        )
        stores[f"m{m}"] = (pred, tgt, copy, swap)
        # static clip: nothing moves, the target is slot + content
        st_t = slot[None] + content
        st_copy = np.repeat(content[:, 0][:, None], S, axis=1)
        st_pred = (
            (st_t + rng.normal(0, 0.2, st_t.shape))
            if kind == "position"
            else (st_copy + rng.normal(0, 0.2, st_t.shape))
        )
        static[f"m{m}"] = FC.l1(st_pred, st_t) / FC.l1(st_copy, st_t)
    folds = FC.two_fold(list(stores))
    for name, (pred, tgt, copy, swap) in stores.items():
        means = FC.slot_means([stores[f][1] for f in folds[name]])
        per_movie[name] = FC.movie_scores(pred, tgt, copy, swap, means)
    return per_movie, static


def test_position_model_fails_the_forecasting_criterion():
    per, static = _synthetic("position")
    g = FC.gate(per, static)
    assert g["beats_copy_last"]
    assert not g["static_removes_most"]
    assert not g["pass"]


def test_dynamics_model_passes_the_forecasting_criterion():
    per, static = _synthetic("dynamics")
    g = FC.gate(per, static)
    assert g["beats_copy_last"] and g["beats_slot_mean"]
    assert g["context_swap_removes_most"] and g["static_removes_most"]
    assert g["pass"]


def test_two_fold_never_fits_a_movie_on_itself():
    f = FC.two_fold([f"m{i}" for i in range(6)])
    assert all(m not in v for m, v in f.items())
    assert all(len(v) == 3 for v in f.values())


def test_sign_flip_minimum_p():
    r = FC.sign_flip([-1.0] * 6)
    assert r["p_two_sided"] == pytest.approx(2 / 64)
    assert FC.sign_flip([1.0, -1.0, 1.0, -1.0])["p_two_sided"] == 1.0


# ------------------------------------------------------------------ re-identification
def test_reid_perfect_and_chance():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(20, 16))
    anchors = np.repeat([0, 1], 10)
    perfect = R.score(z, {k: z + 1e-3 * rng.normal(size=z.shape) for k in R.GAPS}, anchors)
    assert perfect["validation_score"] == 1.0
    assert perfect["chance"] == pytest.approx(0.1)
    shuffled = R.score(z, {k: rng.normal(size=z.shape) for k in R.GAPS}, anchors)
    assert shuffled["validation_score"] < 0.5


def test_effective_rank():
    rng = np.random.default_rng(0)
    assert P.effective_rank(rng.normal(size=(2000, 8))) == pytest.approx(8, rel=0.05)
    low = rng.normal(size=(2000, 1)) @ rng.normal(size=(1, 8))
    assert P.effective_rank(low) < 1.1


# ------------------------------------------------------------------------ association
def test_idf1_perfect_and_swapped():
    A = _load("76_association_v3.py")
    gt = [np.array([1, 2, 3])] * 4
    assert A.idf1(gt, [np.array([7, 8, 9])] * 4) == 1.0
    swapped = [np.array([7, 8, 9])] * 2 + [np.array([8, 7, 9])] * 2
    assert A.idf1(gt, swapped) == pytest.approx(8 / 12)


def test_hungarian_recovers_identity():
    A = _load("76_association_v3.py")
    pa = np.array([[0, 0], [10, 0], [0, 10.0]])
    pb = pa[[2, 0, 1]] + 0.5
    c = A.costs(None, None, pa, pb)["position"]
    assert list(A.hungarian(c)) == [1, 2, 0]


# ------------------------------------------------------------------- sample identity
def test_genotype_check_flags_a_swapped_donor():
    Q = _load("46_partC_sample_qc_v3.py")
    meta = pd.DataFrame(
        {"donor": ["A"] * 3 + ["B"] * 3, "time": "d6", "arm": "x"},
        index=[f"s{i}" for i in range(6)],
    )
    good = pd.DataFrame(
        {f"s{i}": {"HLA-DRB5": 8.0 if i < 3 else -20.0, "GSTM1": 4.0} for i in range(6)}
    )
    assert Q.genotype_check(good, meta)["consistent"]
    bad = good.copy()
    bad.loc["HLA-DRB5", "s4"] = 8.0
    assert not Q.genotype_check(bad, meta)["consistent"]


# ------------------------------------------------------------------------------ power
def test_power_matches_known_value():
    from scipy import stats

    def n_for(d):
        for n in range(2, 200):
            df = 2 * n - 2
            tc = stats.t.ppf(0.975, df)
            nc = d * np.sqrt(n / 2)
            if 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc) >= 0.8:
                return n

    assert n_for(1.0) == 17  # standard two-sample result for d = 1
    assert n_for(0.5) == 64


def test_bh_monotone():
    q = P.bh([0.01, 0.04, 0.03, None, 0.5])
    assert q[3] is None and q[0] <= q[2] <= q[1] <= q[4]
