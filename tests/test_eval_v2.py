"""E1 finite-MAE assertion, buffered LOFO, and the B4 ortholog control (PLAN_v2 G7.1)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(file: str):
    spec = importlib.util.spec_from_file_location(file.replace(".", "_"),
                                                  REPO / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bbbc():
    return _load("62_bbbc054_v1.py")


def test_mae_assertion_rejects_the_v1_blow_up(bbbc):
    with pytest.raises(bbbc.RegressionBlowUp):
        bbbc.check_mae(1.0e8, 7.5, "synthetic")
    with pytest.raises(bbbc.RegressionBlowUp):
        bbbc.check_mae(float("nan"), 7.5, "synthetic")
    bbbc.check_mae(8.0, 7.5, "synthetic")


def test_lofo_is_finite_with_extreme_and_infinite_features(bbbc):
    rng = np.random.default_rng(0)
    fovs = np.repeat(np.arange(20), 100)
    hours = fovs * 0.5
    synthetic_x = rng.normal(size=(len(fovs), 5))
    synthetic_x[:, 0] += hours            # a real time signal
    synthetic_x[3, 1] = 1e300             # an outlier that must be clipped
    synthetic_x[5, 2] = np.inf            # a non-finite feature, dropped
    r = bbbc.lofo_buffered_regression(synthetic_x, hours, fovs, seed=0)
    assert np.isfinite(r["mae_hours"]) and r["max_nonfinite_features_dropped"] == 1
    assert r["skill_vs_cv_mean_predictor"] > 0.5
    assert r["spearman_rho_cv_mean_predictor"] < -0.5  # the leave-out artefact itself


def test_ortholog_positive_control():
    b4 = _load("27_partB_b33_b4_v2.py")
    good = {m: {h} for m, h in b4.ORTHOLOG_CONTROL.items()}
    assert b4.ortholog_control(good)["status"] == "PASS"
    bad = dict(good)
    bad["Il6"] = {"IL6R"}
    with pytest.raises(SystemExit):
        b4.ortholog_control(bad)
