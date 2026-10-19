"""Controls for the latent-forecasting result (PLAN_v3 H2, finding R6).

A predictor that receives the positional embedding of each future slot can beat
"copy the last context latent" without modelling any cell dynamics, because the target
latents themselves carry temporal-position structure. These functions score a model's
future-slot predictions against baselines and controls that separate the two
explanations; the G-Forecast gate is fixed here, before any result is seen.

All L1 values are within one model's latent space; only ratios are compared across
models.
"""
from __future__ import annotations

import itertools

import numpy as np

#: G-Forecast thresholds, fixed in advance (PLAN_v3 §2)
GATE_ALPHA = 0.05
#: a control "removes most of the advantage" if what is left is at most this fraction
GATE_RESIDUAL_FRACTION = 0.5


def l1(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference over every element."""
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def slot_means(targets: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Slot-conditional mean target latents fitted on a set of movies.

    targets: per movie, (n_clips, n_slots, n_pos, dim). Returns the mean per
    (slot, position) and the mean per slot (averaged over positions as well), each
    broadcastable against one clip's (n_slots, n_pos, dim) targets.
    """
    t = np.concatenate([x.astype(np.float32) for x in targets], axis=0)
    per_pos = t.mean(axis=0)                                   # (S, P, D)
    per_slot = per_pos.mean(axis=1, keepdims=True)             # (S, 1, D)
    return {"slot_position": per_pos, "slot": per_slot}


def two_fold(movies: list[str]) -> dict[str, list[str]]:
    """Each movie -> the movies its baselines are fitted on (the other half)."""
    ms = sorted(movies)
    half = len(ms) // 2
    a, b = ms[:half], ms[half:]
    return {m: (b if m in a else a) for m in ms}


def sign_flip(values: list[float]) -> dict:
    """Exact two-sided sign-flip test of mean(values) == 0 over groups."""
    d = np.asarray(values, dtype=float)
    if len(d) == 0:
        return {"n": 0, "mean": None, "p_two_sided": None, "minimum_achievable_p": None}
    obs = float(d.mean())
    flips = np.array([np.mean(d * np.array(s))
                      for s in itertools.product([1, -1], repeat=len(d))])
    return {"n": int(len(d)), "mean": obs,
            "p_two_sided": float(np.mean(np.abs(flips) >= abs(obs) - 1e-12)),
            "minimum_achievable_p": float(2.0 / len(flips))}


def movie_scores(pred: np.ndarray, tgt: np.ndarray, copy: np.ndarray,
                 pred_swap: np.ndarray, means: dict[str, np.ndarray]) -> dict:
    """Per-movie L1s for one model; arrays are (n_clips, n_slots, n_pos, dim)."""
    s = {"l1_predictor": l1(pred, tgt), "l1_copy_last": l1(copy, tgt),
         "l1_context_swap": l1(pred_swap, tgt)}
    for k, m in means.items():
        s[f"l1_mean_{k}"] = l1(np.broadcast_to(m, tgt.shape), tgt)
    s["l1_slot_mean_best"] = min(s["l1_mean_slot"], s["l1_mean_slot_position"])
    s["ratio_vs_copy_last"] = s["l1_predictor"] / s["l1_copy_last"]
    s["ratio_vs_slot_mean"] = s["l1_predictor"] / s["l1_slot_mean_best"]
    s["swap_ratio_vs_copy_last"] = s["l1_context_swap"] / s["l1_copy_last"]
    s["swap_degradation"] = s["l1_context_swap"] / s["l1_predictor"]
    return s


def advantage(ratio: float) -> float:
    """Fractional improvement over copy-last (0 = none, 1 = perfect)."""
    return 1.0 - ratio


def gate(per_movie: dict[str, dict], static_ratio: dict[str, float]) -> dict:
    """G-Forecast on one model at the primary stride.

    Passes iff the predictor beats copy-last AND the slot-conditional mean (each a mean
    log-ratio below 0 with exact sign-flip p < GATE_ALPHA over movies) AND both the
    context-swap and the static-clip controls leave at most GATE_RESIDUAL_FRACTION of
    its advantage over copy-last.
    """
    movies = sorted(per_movie)
    lr_copy = [float(np.log(per_movie[m]["ratio_vs_copy_last"])) for m in movies]
    lr_slot = [float(np.log(per_movie[m]["ratio_vs_slot_mean"])) for m in movies]
    t_copy, t_slot = sign_flip(lr_copy), sign_flip(lr_slot)
    adv = float(np.mean([advantage(per_movie[m]["ratio_vs_copy_last"]) for m in movies]))
    adv_swap = float(np.mean([advantage(per_movie[m]["swap_ratio_vs_copy_last"])
                              for m in movies]))
    sm = [m for m in movies if m in static_ratio]
    adv_static = float(np.mean([advantage(static_ratio[m]) for m in sm])) if sm else None
    beats_copy = t_copy["mean"] < 0 and t_copy["p_two_sided"] < GATE_ALPHA
    beats_slot = t_slot["mean"] < 0 and t_slot["p_two_sided"] < GATE_ALPHA
    lim = GATE_RESIDUAL_FRACTION * adv
    swap_kills = adv > 0 and adv_swap <= lim
    static_kills = adv > 0 and adv_static is not None and adv_static <= lim
    return {
        "rule": "pass iff beats copy-last and the slot-conditional mean (mean log ratio < 0, "
                f"exact sign-flip p < {GATE_ALPHA} over movies) and the context-swap and "
                f"static-clip controls each leave <= {GATE_RESIDUAL_FRACTION} of the "
                "advantage over copy-last",
        "n_movies": len(movies),
        "log_ratio_vs_copy_last": t_copy, "log_ratio_vs_slot_mean": t_slot,
        "advantage_real": adv, "advantage_context_swap": adv_swap,
        "advantage_static": adv_static,
        "beats_copy_last": bool(beats_copy), "beats_slot_mean": bool(beats_slot),
        "context_swap_removes_most": bool(swap_kills),
        "static_removes_most": bool(static_kills),
        "pass": bool(beats_copy and beats_slot and swap_kills and static_kills),
    }


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Closed-form ridge with an intercept column; returns (dim+1, out)."""
    xa = np.hstack([x.astype(np.float64), np.ones((len(x), 1))])
    reg = lam * np.eye(xa.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(xa.T @ xa + reg, xa.T @ y.astype(np.float64))


def ridge_apply(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.hstack([x.astype(np.float64), np.ones((len(x), 1))]) @ w
