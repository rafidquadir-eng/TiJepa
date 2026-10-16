"""v2 paper figures (PLAN_v2.md G6; finding N11).

Rules enforced here: a panel is drawn only from data that exist (a missing analysis is
omitted, never shown as a text box); titles name what is plotted, never a plan section;
every encoder label comes from `tijepa.eval.provenance`; where n is small (wells, movies,
donors) the individual points are shown.

Colour carries one job only, emphasis: TiJepa-1 in the first categorical slot, the
spatial-only ablation in the second, every other encoder in neutral grey. Identity is
always given by a text label on the axis, never by colour alone.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..eval import provenance as PV  # noqa: E402

TIJEPA1 = "#2a78d6"
SPATIAL = "#eb6834"
GREY = "#8a8985"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
})


def _r(repo: Path, rel: str):
    p = repo / "results" / "v2" / rel
    return json.loads(p.read_text()) if p.exists() else None


def _colour(key: str) -> str:
    if key.startswith("E4s_"):
        return SPATIAL
    if key.startswith("E4_"):
        return TIJEPA1
    return GREY


def _label(key: str, rec: dict | None) -> str:
    cp = (rec or {}).get("checkpoint")
    if cp is None and (rec or {}).get("checkpoint_per_seed"):
        cp = next(iter(rec["checkpoint_per_seed"].values()))
    base = PV.label_for_key(key, cp)
    return base + {"E1_handcrafted_logreg": " (logistic)",
                   "E1_handcrafted_rf": " (random forest)"}.get(key, "")


def _save(fig, repo: Path, name: str) -> list[str]:
    out = repo / "figures"
    out.mkdir(exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(str(p.relative_to(repo)))
    plt.close(fig)
    return paths


def _log(repo: Path, run: str) -> dict[str, np.ndarray]:
    p = repo / "checkpoints" / f"pretrain_log_{run}_seed0.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(p.open()))

    def col(k):
        return np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    return {k: col(k) for k in ("step", "loss_spatial", "loss_future", "effective_rank")}


def fig2_pretraining(repo: Path) -> list[str]:
    ps = _r(repo, "partA/pretrain_tijepa1_seed0.json")
    if not ps:
        return []
    logs = {"tijepa1": _log(repo, "tijepa1"), "tijepa1_spatial": _log(repo, "tijepa1_spatial")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]

    def smooth(y, k=20):
        if len(y) < k:
            return y
        return np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), np.ones(k) / k, "same")

    L = logs["tijepa1"]
    ax.plot(L["step"], smooth(L["loss_spatial"]), color=TIJEPA1, lw=1.6,
            label="TiJepa-1, spatial-masked batches")
    ax.plot(L["step"], smooth(L["loss_future"]), color=TIJEPA1, lw=1.6, ls="--",
            label="TiJepa-1, future-masked batches")
    if logs["tijepa1_spatial"]:
        S = logs["tijepa1_spatial"]
        ax.plot(S["step"], smooth(S["loss_spatial"]), color=SPATIAL, lw=1.6,
                label="TiJepa-1-spatial")
    ax.set_xlabel("step")
    ax.set_ylabel("latent L1 loss (moving mean)")
    ax.set_title("Training loss")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    ax.plot(L["step"], L["effective_rank"], color=TIJEPA1, lw=1.2, label="TiJepa-1")
    if logs["tijepa1_spatial"]:
        ax.plot(S["step"], S["effective_rank"], color=SPATIAL, lw=1.2,
                label="TiJepa-1-spatial")
    ax.set_xlabel("step")
    ax.set_ylabel("effective rank (of 384)")
    ax.set_ylim(bottom=0)
    ax.set_title("Embedding effective rank")
    ax.legend(frameon=False, fontsize=7)
    if len(ps["segments"]) > 1:
        for a in axes:
            a.axvline(ps["segments"][1]["start_step"], color=MUTED, lw=0.8, ls=":")
        axes[1].annotate("resume", (ps["segments"][1]["start_step"], 2), fontsize=7,
                         color=MUTED, xytext=(4, 0), textcoords="offset points")
    fig.tight_layout()
    return _save(fig, repo, "fig2_pretraining")


def fig4_bbbc054(repo: Path) -> list[str]:
    bb = _r(repo, "partA/bbbc054.json")
    if not bb:
        return []
    enc = {k: v for k, v in bb["encoders"].items()
           if isinstance(v, dict) and "three_class_mean" in v}
    order = sorted(enc, key=lambda k: enc[k]["three_class_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.42 * len(order) + 1.1),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    for i, k in enumerate(order):
        vals = [r["balanced_accuracy"] for r in enc[k]["tasks"]["three_class"]]
        ax.plot([min(vals), max(vals)], [i, i], color=_colour(k), lw=2, alpha=0.5)
        ax.scatter(vals, [i] * len(vals), s=22, color=_colour(k), zorder=3,
                   edgecolor="white", linewidth=0.8)
    tob = bb["time_only_baseline"]["three_class"]["mean"]
    ax.axvline(1 / 3, color=MUTED, lw=0.8, ls=":")
    ax.axvline(tob, color=MUTED, lw=0.8, ls="--")
    ax.text(1 / 3, len(order) - 0.4, "chance", fontsize=7, color=MUTED, ha="right")
    ax.text(tob, len(order) - 0.4, " time only", fontsize=7, color=MUTED, ha="left")
    ax.set_yticks(range(len(order)), [_label(k, enc[k]) for k in order])
    ax.set_xlabel("three-class balanced accuracy (each dot one seed)")
    ax.set_title("Morphology-class probe")
    ax.set_ylim(-0.6, len(order) - 0.1)
    ax = axes[1]
    for i, k in enumerate(order):
        vals = [r["exposure_time_lofo"]["skill_vs_cv_mean_predictor"]
                for r in enc[k]["per_seed"]]
        ax.scatter(vals, [i] * len(vals), s=22, color=_colour(k), zorder=3,
                   edgecolor="white", linewidth=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(order)), [""] * len(order))
    ax.set_xlabel("exposure-time skill vs fold-mean predictor")
    ax.set_title("Exposure-time regression (exploratory)")
    ax.set_ylim(-0.6, len(order) - 0.1)
    fig.tight_layout()
    return _save(fig, repo, "fig4_bbbc054")


def fig5_tracking_forecast(repo: Path) -> list[str]:
    tf = _r(repo, "partA/tracking_forecast.json")
    if not tf:
        return []
    agg = tf["tracking"]["aggregate"]
    encs = tf["encoders"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axes[0]
    th = [1, 2, 4, 8, 16]
    show = ["baseline_nearest_detection", "E0_tijepa0", "E3_dinov2_vit_s",
            "E4s_tijepa1_spatial", "E4_tijepa1"]
    for k in show:
        v = agg.get(k, {})
        if "per_threshold_success" not in v:
            continue
        y = [v["per_threshold_success"][f"within_{t}um"] for t in th]
        name = ("nearest-detection baseline" if k.startswith("baseline")
                else _label(k, encs.get(k)))
        style = {"color": _colour(k), "lw": 1.8 if k.startswith("E4") else 1.2,
                 "ls": "-" if k.startswith("E4") else "--"}
        if k.startswith("baseline"):
            style.update(color=INK, ls=":")
        ax.plot(th, y, marker="o", ms=4, label=name, **style)
    ax.set_xscale("log", base=2)
    ax.set_xticks(th, [str(t) for t in th])
    ax.set_ylim(0, 1)
    ax.set_xlabel("error threshold (µm)")
    ax.set_ylabel("fraction of frames within threshold")
    ax.set_title("Point tracking, all tracks")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    summ = tf["forecasting"]["summary"]
    rows = [k for k in ("E4_tijepa1", "E4s_tijepa1_spatial", "E0_tijepa0") if k in summ]
    movies = sorted({m for k in rows for m in summ[k]["per_movie_ratio_vs_copy_last"]})
    for j, k in enumerate(rows):
        pm = summ[k]["per_movie_ratio_vs_copy_last"]
        xs = [movies.index(m) + (j - 1) * 0.18 for m in pm]
        ax.scatter(xs, list(pm.values()), s=26, color=_colour(k), zorder=3,
                   edgecolor="white", linewidth=0.8, label=_label(k, encs[k]))
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.text(len(movies) - 0.5, 1, "copy-last", fontsize=7, color=MUTED, va="bottom",
            ha="right")
    ax.set_xticks(range(len(movies)), [m.split(".")[0] for m in movies], rotation=30)
    ax.set_ylabel("L1(predictor) / L1(copy-last)")
    ax.set_title("Latent forecasting, per movie")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _save(fig, repo, "fig5_tracking_forecast")


def fig6_partB(repo: Path) -> list[str]:
    b3 = _r(repo, "partB/b3_dynamics.json")
    b34 = _r(repo, "partB/b33_b4.json")
    if not b3 or not b34:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    pw = b3["b31_tolerance"]["results"]["raw_trace"]["per_well"]
    for i, (_w, recs) in enumerate(sorted(pw.items())):
        lab = recs[0]["true_label"]
        vals = [r["fraction_predicted_tolerant"] for r in recs]
        c = INK if lab == "LPS->LPS" else GREY
        ax.scatter([i] * len(vals), vals, s=24, color=c, zorder=3, edgecolor="white",
                   linewidth=0.8)
        ax.text(i, -0.12, lab.replace("->", "→"), fontsize=6.5, ha="center", color=MUTED)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(range(len(pw)), [f"well {i + 1}" for i in range(len(pw))])
    ax.set_ylim(-0.18, 1)
    ax.set_ylabel("fraction of cells called tolerant")
    ax.set_title("Acute endotoxin tolerance, held-out wells")
    ax = axes[1]
    fi = b34["b4"]["fisher"]
    keys = [k for k in fi if k.startswith("GSE185433")] + \
        [k for k in fi if k.startswith("GSE58310")]
    for i, k in enumerate(keys):
        t = fi[k]
        ax.scatter(t["n_overlap"], i, s=28, color=INK, zorder=3)
        ax.scatter(t["expected_overlap"], i, s=28, facecolor="white", edgecolor=INK,
                   zorder=3)
        ax.plot([t["n_overlap"], t["expected_overlap"]], [i, i], color=GREY, lw=1)
    names = {"GSE185433_primary": "restimulated (GSE185433)",
             "GSE58310_secondary": "resting (GSE58310)"}
    ax.set_yticks(range(len(keys)),
                  [f"{names[k.split(':')[0]]}, {k.split(':')[1]}" for k in keys])
    ax.set_xlabel("overlap genes (filled: observed, open: expected)")
    ax.set_title("Mouse priming-blunted genes vs human tolerance")
    fig.tight_layout()
    return _save(fig, repo, "fig6_partB")


def fig7_partC(repo: Path) -> list[str]:
    rs = _r(repo, "partC/c3_restimulation.json")
    if not rs:
        return []
    rows = []
    for gse in ("GSE212282", "GSE185433"):
        for arm, r in (rs[gse].get("arms") or {}).items():
            if "primary_TNF_IL6" in r:
                rows.append((gse, arm, r["primary_TNF_IL6"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.36 * len(rows) + 1.2), sharey=True)
    for ax, g in zip(axes, ("TNF", "IL6"), strict=True):
        for i, (_gse, _arm, p) in enumerate(rows):
            x = p[g]
            exp = x["sourced_expectation"]
            c = INK if exp in ("higher", "lower") else GREY
            ax.scatter(x["per_donor_log2_difference"], [i] * 3, s=22, color=c, zorder=3,
                       edgecolor="white", linewidth=0.8)
            if exp in ("higher", "lower"):
                ax.annotate("expected ↑" if exp == "higher" else "expected ↓",
                            (1.0, i), xycoords=("axes fraction", "data"), fontsize=6.5,
                            color=MUTED, ha="right", va="center")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(f"{g}: log2 difference vs untrained → LPS (per donor)")
        ax.set_title(g)
    axes[0].set_yticks(range(len(rows)), [f"{a.replace('_', ' ')} ({g})"
                                          for g, a, _ in rows])
    fig.tight_layout()
    return _save(fig, repo, "fig7_partC")


ALL = [fig2_pretraining, fig4_bbbc054, fig5_tracking_forecast, fig6_partB, fig7_partC]
