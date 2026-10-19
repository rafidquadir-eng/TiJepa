"""v3 manuscript figures (figures/v3/).

Same rules as v2: a panel is drawn only from results that exist; titles name what is
plotted; every encoder label comes from `tijepa.eval.provenance`; with few groups (movies,
wells, donors) the individual points are shown. Colour carries emphasis only: TiJepa-1
(blue), its spatial-only ablation (orange), the pretrained video model V-JEPA 2 (teal),
everything else grey; identity is always written on the axis or in a direct label.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from ..eval import provenance as PV  # noqa: E402
from .figures_v2 import GREY, GRID, INK, MUTED, SPATIAL, TIJEPA1  # noqa: E402,F401

VJEPA = "#1b8a7a"
OUT_DIR = Path("figures") / "v3"


def _r(repo: Path, rel: str):
    p = repo / "results" / rel
    return json.loads(p.read_text()) if p.exists() else None


def colour(key: str) -> str:
    if key.startswith("E4s_"):
        return SPATIAL
    if key.startswith(("E4_", "trace_jepa")) and "random" not in key:
        return TIJEPA1
    if key.startswith("E7_"):
        return VJEPA
    return GREY


def label(key: str, rec: dict | None = None) -> str:
    extra = {
        "E1_handcrafted_logreg": "hand-crafted (logistic)",
        "E1_handcrafted_rf": "hand-crafted (random forest)",
        "E1_handcrafted": "hand-crafted",
        "intensity_baseline": "intensity only",
        "raw_pixels": "raw pixels",
        "position_only": "position only",
        "raw_trace": "raw trace",
        "handcrafted": "hand-crafted features",
        "trace_jepa": "trace JEPA",
        "trace_jepa_random_init": "trace JEPA, random init",
        "E5_frame_shuffled": "TiJepa-1, frames shuffled",
        "E6_random_init": "random init",
    }
    if key in extra:
        return extra[key]
    if key.startswith("E6_random_init_draw"):
        return f"random init (draw {int(key[-1]) + 1})"
    cp = (rec or {}).get("checkpoint")
    if cp is None and (rec or {}).get("checkpoint_per_seed"):
        cp = next(iter(rec["checkpoint_per_seed"].values()))
    return PV.label_for_key(key, cp)


def save(fig, repo: Path, name: str) -> list[str]:
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(str(p.relative_to(repo)))
    plt.close(fig)
    return paths


def _log(repo: Path, path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))

    def col(k):
        return np.array([float(r[k]) if r.get(k) not in ("", None) else np.nan for r in rows])

    out = {
        k: col(k)
        for k in ("step", "loss_spatial", "loss_future", "effective_rank", "std_mean", "loss")
    }
    if not np.isfinite(out["loss_spatial"]).any():  # older log: one loss column
        out["loss_spatial"] = out["loss"]
    return out


def _smooth(x, y, k=40):
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    x, y = np.asarray(x)[ok], y[ok]
    if len(y) < k:
        return x, y
    return x[k - 1 :], np.convolve(y, np.ones(k) / k, "valid")


# ---------------------------------------------------------------- Figure 1: coverage
def fig1_coverage(repo: Path) -> list[str]:
    cov = _r(repo, "v3/coverage/coverage_map.json")
    if not cov:
        return []
    rows = cov["rows"]
    levels = ["same cell", "same condition", "cross-study", "no public pairing"]
    glyph = {
        "positive": ("●", INK),
        "null": ("○", MUTED),
        "negative": ("○", MUTED),
        "not tested": ("–", MUTED),
        "absent": ("×", MUTED),
    }
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(rows) + 1.0))
    ax.set_xlim(-0.75, len(levels) - 0.35)
    ax.set_ylim(len(rows) - 0.5, -0.9)
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(range(len(levels)), levels)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0)
    ax.set_yticks(range(len(rows)), [r["family"] for r in rows])
    for i, r in enumerate(rows):
        j = levels.index(r["pairing"])
        g, c = glyph.get(r["outcome"], ("?", INK))
        ax.text(
            j,
            i,
            f"{g} {r['short']}",
            ha="center",
            va="center",
            fontsize=6.4,
            color=c,
            clip_on=False,
        )
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=GRID, alpha=0.35, lw=0)
    ax.text(
        len(levels) - 0.35,
        len(rows) - 0.1,
        "● the imaging-derived readout predicts the modality beyond the row's reference   "
        "○ tested, no gain   – not testable   × no public pairing or data",
        ha="right",
        va="top",
        fontsize=6.4,
        color=MUTED,
    )
    fig.tight_layout()
    return save(fig, repo, "fig1_coverage_map")


# ------------------------------------------------- Figure 2: training and collapse
def fig2_training(repo: Path) -> list[str]:
    runs = {
        "E0_tijepa0": repo / "checkpoints" / "pretrain_log_seed0.csv",
        "E4_tijepa1": repo / "checkpoints" / "pretrain_log_tijepa1_seed0.csv",
        "E4s_tijepa1_spatial": repo / "checkpoints" / "pretrain_log_tijepa1_spatial_seed0.csv",
    }
    logs = {k: _log(repo, p) for k, p in runs.items()}
    if not logs["E4_tijepa1"]:
        return []
    names = {
        "E0_tijepa0": "TiJepa-0",
        "E4_tijepa1": "TiJepa-1",
        "E4s_tijepa1_spatial": "TiJepa-1 (spatial-only)",
    }
    reid = _r(repo, "v3/partA/validation_reid.json")
    sweep = _r(repo, "v3/partA/recipe_sweep.json")
    ncol = 2 + bool(reid) + bool(sweep)
    fig, axes = plt.subplots(1, ncol, figsize=(9.6 if ncol == 4 else 7.4, 2.5))
    ax = axes[0]
    for k, L in logs.items():
        if not L:
            continue
        c = colour(k) if k != "E0_tijepa0" else INK
        ax.plot(*_smooth(L["step"], L["loss_spatial"]), color=c, lw=1.3, label=names[k])
        if np.isfinite(L["loss_future"]).any():
            ax.plot(*_smooth(L["step"], L["loss_future"]), color=c, lw=1.1, ls="--")
    ax.set_xlabel("step")
    ax.set_ylabel("latent L1 loss (moving mean)")
    ax.set_title("Training loss (dashed: future masks)", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)
    ax = axes[1]
    for k, L in logs.items():
        if not L:
            continue
        c = colour(k) if k != "E0_tijepa0" else INK
        ax.plot(*_smooth(L["step"], L["effective_rank"]), color=c, lw=1.3, label=names[k])
    ax.set_xlabel("step")
    ax.set_ylabel("effective rank")
    ax.set_ylim(bottom=0)
    ax.set_title("Rank of training-batch embedding", fontsize=8)
    if reid:
        ax = axes[2]
        diag = reid["diagnosis"]
        for k in ("E4_tijepa1", "E4s_tijepa1_spatial", "E0_tijepa0", "E6_random_init_draw0"):
            if k not in diag:
                continue
            f = diag[k]["validation_clips"]["target"]["top_variance_fractions"][:16]
            ax.plot(
                range(1, len(f) + 1),
                np.cumsum(f),
                marker="o",
                ms=2.5,
                lw=1.2,
                color=colour(k) if k != "E0_tijepa0" else INK,
                label=label(k, reid["encoders"].get(k)),
            )
        ax.set_xlabel("principal component")
        ax.set_ylabel("cumulative variance fraction")
        ax.set_ylim(0, 1.02)
        ax.set_title("Spectrum on held-out clips", fontsize=8)
        ax.legend(frameon=False, fontsize=6.3, loc="lower right")
    if sweep:
        ax = axes[-1]
        shades = {
            "S0": INK,
            "S1": "#5b5a56",
            "S2": "#8a8985",
            "S3": "#b0afab",
            "S4": "#cfcdc8",
        }
        for vid, e in sweep["variants"].items():
            st = sorted((int(s), v["validation_score"]) for s, v in e["snapshots"].items())
            if st:
                xs, ys = zip(*st, strict=False)
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    ms=3,
                    lw=1.2 if vid == "S0" else 0.9,
                    color=shades.get(vid, GREY),
                    label=vid,
                )
        ax.set_xlabel("step")
        ax.set_ylabel("validation re-ID top-1")
        ax.set_title("Recipe variants (20k steps)", fontsize=8)
        ax.legend(frameon=False, fontsize=6.3, ncol=2)
    fig.tight_layout()
    return save(fig, repo, "fig2_training_collapse")


# ------------------------------------------------------- Figure 3: Part A benchmarks
def fig3_benchmarks(repo: Path) -> list[str]:
    bb = _r(repo, "v3/partA/bbbc054.json")
    af = _r(repo, "v3/coverage/autofluorescence.json")
    reid = _r(repo, "v3/partA/validation_reid.json")
    if not bb:
        return []
    panels = [p for p in (bb, af, reid) if p]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.4, 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    enc = {
        k: v
        for k, v in bb["encoders"].items()
        if isinstance(v, dict) and "three_class_mean" in v
    }
    order = sorted(enc, key=lambda k: enc[k]["three_class_mean"])
    for i, k in enumerate(order):
        vals = [r["balanced_accuracy"] for r in enc[k]["tasks"]["three_class"]]
        ax.scatter(
            vals,
            [i] * len(vals),
            s=16,
            color=colour(k),
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.axvline(1 / 3, color=MUTED, lw=0.8, ls=":")
    ax.axvline(bb["time_only_baseline"]["three_class"]["mean"], color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(range(len(order)), [label(k, enc[k]) for k in order], fontsize=6.6)
    ax.set_xlabel("balanced accuracy (dots: seeds)")
    ax.set_title(
        "Microglia morphology\n(3 classes; dotted chance, dashed time only)", fontsize=7.5
    )
    j = 1
    if af:
        ax = axes[j]
        j += 1
        res = af["results"]["A_mixed_fovs_only"]
        keys = [k for k in res if k.endswith("|all")]
        keys = sorted(keys, key=lambda k: res[k]["balanced_accuracy_mean"])
        for i, k in enumerate(keys):
            r = res[k]
            base = k.split("|")[0]
            lo, hi = r["ci95_seed0"]
            ax.plot([lo, hi], [i, i], color=colour(base), lw=1.6, alpha=0.6)
            ax.scatter([r["balanced_accuracy_mean"]], [i], s=16, color=colour(base), zorder=3)
        ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_yticks(
            range(len(keys)),
            [label(k.split("|")[0], af["encoders"].get(k.split("|")[0])) for k in keys],
            fontsize=6.6,
        )
        ax.set_xlabel("balanced accuracy (95% CI over fields)")
        ax.set_title("T cell vs neutrophil, label-free\n(mixed fields only)", fontsize=7.5)
    if reid:
        ax = axes[j]
        r = {k: v for k, v in reid["reid"].items()}
        keys = sorted(r, key=lambda k: r[k]["validation_score_mean"])
        for i, k in enumerate(keys):
            lo, hi = r[k]["ci95_over_movies"]
            ax.plot([lo, hi], [i, i], color=colour(k), lw=1.6, alpha=0.6)
            ax.scatter([r[k]["validation_score_mean"]], [i], s=16, color=colour(k), zorder=3)
        ax.axvline(
            np.mean([v["chance_mean"] for v in r.values()]), color=MUTED, lw=0.8, ls=":"
        )
        ax.set_yticks(
            range(len(keys)), [label(k, reid["encoders"].get(k)) for k in keys], fontsize=6.6
        )
        ax.set_xlabel("re-identification top-1 (95% CI over movies)")
        ax.set_title("Cell re-identification\n(held-out neutrophil movies)", fontsize=7.5)
    fig.tight_layout()
    return save(fig, repo, "fig3_benchmarks")


# --------------------------------------------- Figure 4: association and forecasting
def fig4_association_forecast(repo: Path) -> list[str]:
    asc = _r(repo, "v3/partA/association.json")
    fc = _r(repo, "v3/partA/forecast_controls.json")
    if not fc:
        return []
    n = 2 if asc else 1
    fig, axes = plt.subplots(
        1, n, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1, 1.25][:n]}
    )
    axes = np.atleast_1d(axes)
    j = 0
    if asc:
        ax = axes[0]
        j = 1
        s = asc["summary"]
        pos = s["position_only"]["position"]
        keys = [k for k in s if k != "position_only"]
        keys = sorted(keys, key=lambda k: s[k]["embedding"]["9"])
        for i, k in enumerate(keys):
            ax.scatter(
                [s[k]["embedding"]["9"]], [i], s=16, color=colour(k), zorder=3, marker="o"
            )
            ax.scatter(
                [s[k]["embedding+position"]["9"]],
                [i],
                s=16,
                color=colour(k),
                zorder=3,
                marker="s",
                facecolor="white",
            )
        ax.axvline(pos["9"], color=INK, lw=0.9, ls="--")
        ax.text(
            pos["9"] - 0.01,
            0.2,
            "position only",
            fontsize=6.3,
            color=INK,
            rotation=90,
            ha="right",
            va="bottom",
        )
        ax.set_yticks(
            range(len(keys)),
            [label(k, asc["encoders"].get(k, {})) for k in keys],
            fontsize=6.6,
        )
        ax.set_xlabel("identity accuracy, 9-frame gap")
        ax.set_title("Association (● embedding, □ + position)", fontsize=7.5)
    ax = axes[j]
    models = [
        k for k in ("E4_tijepa1", "E4s_tijepa1_spatial", "E0_tijepa0") if k in fc["results"]
    ]
    metrics = [
        ("ratio_vs_copy_last", "vs\ncopy-last"),
        ("ratio_vs_slot_mean", "vs\nslot mean"),
        ("swap_ratio_vs_copy_last", "context\nswapped"),
        ("static_ratio_vs_copy_last", "static\nclip"),
    ]
    w = 0.8 / len(models)
    for mi, k in enumerate(models):
        s1 = fc["results"][k]["1"]["per_movie"]
        for xi, (m, _) in enumerate(metrics):
            vals = [v[m] for v in s1.values()]
            x = xi + (mi - (len(models) - 1) / 2) * w
            ax.bar(
                x,
                np.mean(vals),
                width=w * 0.9,
                color=colour(k) if k != "E0_tijepa0" else INK,
                alpha=0.75,
                label=label(k, fc["models"][k]) if xi == 0 else None,
            )
            ax.scatter(
                [x] * len(vals),
                vals,
                s=5,
                color="white",
                edgecolor=INK,
                linewidth=0.4,
                zorder=3,
            )
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(metrics)), [m[1] for m in metrics], fontsize=6.8)
    ax.set_ylabel("predictor error / copy-last error\n(slot mean: / slot-mean error)")
    ax.set_title("Latent forecasting controls (stride 1; dots: movies)", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    return save(fig, repo, "fig4_association_forecasting")


# ---------------------------------------------------------------- Figure 5: Part B
def fig5_partB(repo: Path) -> list[str]:
    pb = _r(repo, "v3/partB/partB_v3.json")
    if not pb:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))
    ax = axes[0]
    b = pb["b32_ligand_identity"]
    keys = sorted(b["results"], key=lambda k: b["results"][k]["balanced_accuracy_mean"])
    for i, k in enumerate(keys):
        r = b["results"][k]
        lo, hi = r["ci95_seed0_well_bootstrap"]
        ax.plot([lo, hi], [i, i], color=colour(k), lw=1.8, alpha=0.6)
        ax.scatter([r["balanced_accuracy_mean"]], [i], s=18, color=colour(k), zorder=3)
    ax.axvline(b["chance_balanced_accuracy"], color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(range(len(keys)), [label(k) for k in keys])
    ax.set_xlabel("balanced accuracy, 11 ligand pairs (95% CI over wells)")
    ax.set_title(
        f"Ligand-pair identity from NF-κB dynamics ({b['n_wells']} wells)", fontsize=7.5
    )
    ax = axes[1]
    from scipy import stats

    pw = pb["b31_power_analysis"]["metrics"]
    ns = np.arange(2, 41)
    for name, ls in (("peak_amplitude", "-"), ("auc", "--")):
        m = pw[name]
        for d, alpha, lab in (
            (m["effect_size_d"], 1.0, "observed SD"),
            (
                m["tolerant_minus_naive"] / m["sd_upper_80pct"],
                0.45,
                "SD at its upper 80% limit",
            ),
        ):
            if d is None:
                continue
            p = []
            for n in ns:
                df = 2 * n - 2
                tc = stats.t.ppf(0.975, df)
                nc = abs(d) * np.sqrt(n / 2)
                p.append(1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))
            ax.plot(
                ns,
                p,
                ls=ls,
                color=TIJEPA1,
                alpha=alpha,
                label=f"{name.replace('_', ' ')}, {lab}",
            )
    ax.axhline(0.8, color=MUTED, lw=0.8, ls=":")
    ax.axvline(2, color=INK, lw=0.8)
    ax.text(
        1.3,
        0.5,
        "deposited: 2 per arm",
        fontsize=6.3,
        color=INK,
        rotation=90,
        ha="right",
        va="center",
    )
    ax.set_xlabel("wells per arm")
    ax.set_ylabel("power (two-sided, α = 0.05)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Detecting acute tolerance from well means", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return save(fig, repo, "fig5_signalling")


# ---------------------------------------------------------------- Figure 6: Part C
def fig6_partC(repo: Path) -> list[str]:
    qc = _r(repo, "v3/partC/sample_qc.json")
    rep = _r(repo, "v3/partC/beta_glucan_replication.json")
    if not qc or not rep:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    tr = qc["datasets"]["GSE212282"]["cytokine_trajectories"]["TNF"]
    times = ["4H", "d1", "d6", "d7"]
    tl = {
        "4H": "4 h\n(first exposure)",
        "d1": "day 1",
        "d6": "day 6\n(rested)",
        "d7": "day 6 + 4 h LPS",
    }
    for d, mk in zip(("A", "B", "C"), ("o", "s", "^"), strict=False):
        ax.plot(
            range(4),
            [tr[t][d]["RPMI"] for t in times],
            color=GREY,
            marker=mk,
            ms=4,
            lw=1,
            label="untrained" if d == "A" else None,
        )
        ax.plot(
            range(4),
            [tr[t][d]["BG"] for t in times],
            color=TIJEPA1,
            marker=mk,
            ms=4,
            lw=1,
            label="β-glucan" if d == "A" else None,
        )
    ax.set_xticks(range(4), [tl[t] for t in times], fontsize=6.6)
    ax.set_ylabel("TNF mRNA (log2 expression)")
    ax.set_title(
        "TNF transcript over the training protocol (GSE212282; 3 donors)", fontsize=7.5
    )
    ax.legend(frameon=False, fontsize=6.5)
    ax = axes[1]
    rows = []
    g212 = qc["datasets"]["GSE212282"]["cytokine_trajectories"]["TNF"]["d7"]
    rows.append(("GSE212282\nLPS 4 h", [g212[d]["BG_minus_RPMI"] for d in "ABC"]))
    t141 = rep["GSE141656"]["trained_minus_untrained_after_restim"]["TNF"]
    rows.append(("GSE141656\nM. tuberculosis 4 h", t141["per_donor_log2_difference"]))
    for i, (_name, vals) in enumerate(rows):
        ax.scatter(vals, [i] * len(vals), s=22, color=TIJEPA1, zorder=3)
        ax.plot([np.mean(vals)] * 2, [i - 0.25, i + 0.25], color=INK, lw=1.5)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6.8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("β-glucan − untrained, TNF log2 (dots: donors)")
    ax.set_title("After restimulation", fontsize=7.5)
    fig.tight_layout()
    return save(fig, repo, "fig6_restimulation")


# --------------------------------------------------- Figure 7: the proposed study
def fig7_design(repo: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.4, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    steps = [
        (
            0.1,
            "Day 0",
            "CD14+ monocytes\n(n donors)\nRPMI / β-glucan / LPS\nIFN-γ, IL-4 anchors",
        ),
        (2.1, "Day 1", "wash"),
        (
            3.6,
            "Days 1–6",
            "rest; label-free\ntime-lapse and\nNAD(P)H/FAD imaging\n(days 0, 1, 3, 6)",
        ),
        (5.9, "Day 6", "LPS restimulation;\ntime-lapse\n0–24 h"),
        (
            8.0,
            "Readouts",
            "TNF/IL-6 protein\nat 4 h and 24 h;\nRNA-seq 4 h;\nATAC day 6; lactate",
        ),
    ]
    for x, head, body in steps:
        w = 1.8 if head != "Day 1" else 1.2
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.4), w, 2.1, boxstyle="round,pad=0.03", fc="white", ec=MUTED, lw=0.8
            )
        )
        ax.text(
            x + w / 2,
            2.25,
            head,
            ha="center",
            va="center",
            fontsize=7.5,
            color=INK,
            weight="bold",
        )
        ax.text(x + w / 2, 1.3, body, ha="center", va="center", fontsize=6.6, color=INK)
    for x0, x1 in ((1.9, 2.1), (3.3, 3.6), (5.4, 5.9), (7.7, 8.0)):
        ax.annotate(
            "",
            xy=(x1, 1.45),
            xytext=(x0, 1.45),
            arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.9},
        )
    ax.text(
        5.0,
        0.12,
        "pre-specified analysis: frozen video features from the rest period "
        "→ donor-held-out prediction of naive / trained / tolerant and of 24 h TNF protein",
        ha="center",
        fontsize=6.6,
        color=MUTED,
    )
    return save(fig, repo, "fig7_proposed_study")


ALL = (
    fig1_coverage,
    fig2_training,
    fig3_benchmarks,
    fig4_association_forecast,
    fig5_partB,
    fig6_partC,
    fig7_design,
)
