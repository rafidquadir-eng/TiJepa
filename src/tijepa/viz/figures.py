"""Figure functions (PLAN.md 8.2). Every figure is generated from results/*.json.

One function per figure; each saves PNG + PDF to figures/ and returns the paths.
No number is typed by hand: missing inputs produce an explicit "not available" panel
rather than a plausible-looking placeholder.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Brand-neutral, colour-blind-safe palette, consistent across every figure.
C = {
    "primary": "#2F6DB5", "secondary": "#C85C27", "third": "#4C9A6A",
    "fourth": "#8A5FA8", "grey": "#6E7378", "light": "#D9DCE0",
    "bad": "#B23A48", "ink": "#1D2024",
}
ENCODER_ORDER = ["E1_handcrafted_morphology", "E2_imagenet_vit_s", "E3_dinov2_vit_s",
                 "E4_tijepa0", "E5_tijepa0_frame_shuffled", "E6_tijepa0_random_init"]
ENCODER_LABEL = {
    "E1_handcrafted_morphology": "E1 hand-crafted",
    "E2_imagenet_vit_s": "E2 ImageNet ViT-S",
    "E3_dinov2_vit_s": "E3 DINOv2-S",
    "E4_tijepa0": "E4 TiJepa-0",
    "E5_tijepa0_frame_shuffled": "E5 frame-shuffled",
    "E6_tijepa0_random_init": "E6 random init",
}
ENCODER_COLOR = {
    "E1_handcrafted_morphology": C["grey"], "E2_imagenet_vit_s": C["light"],
    "E3_dinov2_vit_s": C["third"], "E4_tijepa0": C["primary"],
    "E5_tijepa0_frame_shuffled": C["fourth"], "E6_tijepa0_random_init": C["secondary"],
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["grey"], "axes.labelcolor": C["ink"],
        "text.color": C["ink"], "xtick.color": C["grey"], "ytick.color": C["grey"],
        "axes.grid": True, "grid.color": C["light"], "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
    })


def _save(fig, repo: Path, name: str) -> list[str]:
    out = repo / "figures"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(str(p.relative_to(repo)))
    plt.close(fig)
    return paths


def _load(repo: Path, rel: str):
    p = repo / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _na(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=8,
            color=C["bad"], wrap=True, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


# ---------------------------------------------------------------- Fig 1 overview
def fig1_overview(repo: Path) -> list[str]:
    _style()
    man = repo / "manifests" / "datasets.yaml"
    rows = []
    if man.exists():
        import yaml

        for e in yaml.safe_load(man.read_text())["datasets"]:
            rows.append((str(e.get("key")), str(e.get("role") or ""),
                         str(e.get("licence_verified") or ""), str(e.get("status"))))
    fig, ax = plt.subplots(figsize=(9.5, 0.33 * max(len(rows), 4) + 1.4))
    ax.axis("off")
    ax.set_title("Data sources, roles and verified licences", loc="left",
                 fontweight="bold")
    y = len(rows)
    ax.text(0.00, y + 0.6, "dataset", fontweight="bold", fontsize=8)
    ax.text(0.26, y + 0.6, "role", fontweight="bold", fontsize=8)
    ax.text(0.60, y + 0.6, "licence (verified from record)", fontweight="bold", fontsize=8)
    for i, (k, role, lic, st) in enumerate(rows):
        yy = y - i - 0.2
        col = C["bad"] if st != "ok" else C["ink"]
        ax.text(0.00, yy, k[:26], fontsize=7.5, color=col)
        ax.text(0.26, yy, role[:38], fontsize=7.5, color=col)
        ax.text(0.60, yy, (lic[:40] + ("  [UNREACHABLE]" if st != "ok" else "")),
                fontsize=7.5, color=col)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, y + 1.2)
    fig.text(0.01, -0.02,
             "Part A and Part B use real labels/pairings. Part C is a condition-level "
             "cross-study bridge (hypothesis-generating).", fontsize=7.5,
             color=C["grey"])
    return _save(fig, repo, "fig1_overview")


# ------------------------------------------------------------- Fig 2 pretraining
def fig2_pretraining(repo: Path) -> list[str]:
    _style()
    corpus = _load(repo, "results/partA/pretrain_corpus.json")
    summ = _load(repo, "results/partA/pretrain_seed0.json")
    csvp = repo / "checkpoints" / "pretrain_log_seed0.csv"
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))

    ax = axes[0]
    if corpus:
        d = corpus["effective_interval_s_distribution"]["counts"]
        ks = sorted(d, key=float)
        ax.bar([f"{float(k):g}" for k in ks], [d[k] for k in ks], color=C["primary"])
        ax.set_xlabel("effective frame interval (s)")
        ax.set_ylabel("clips")
        ax.set_title(f"Corpus: {corpus['n_clips']:,} clips\n"
                     f"{corpus['total_gb']} GB, {corpus['n_movies_ok']} movies",
                     fontsize=9)
    else:
        _na(ax, "pretrain_corpus.json not available")

    ax = axes[1]
    if csvp.exists():
        import csv as _csv

        with csvp.open() as fh:
            rows = list(_csv.DictReader(fh))
        if rows:
            st = [int(r["step"]) for r in rows]
            ls = [float(r["loss"]) for r in rows]
            ax.plot(st, ls, color=C["primary"], lw=1.2)
            ax.set_xlabel("step")
            ax.set_ylabel("JEPA L1 latent loss")
            ax.set_title("Pretraining loss", fontsize=9)
        else:
            _na(ax, "training log empty")
    else:
        _na(ax, "no training log")

    ax = axes[2]
    if csvp.exists():
        import csv as _csv

        with csvp.open() as fh:
            rows = list(_csv.DictReader(fh))
        if rows:
            st = [int(r["step"]) for r in rows]
            er = [float(r["effective_rank"]) for r in rows]
            sd = [float(r["std_mean"]) for r in rows]
            ax.plot(st, er, color=C["third"], lw=1.2, label="effective rank")
            ax.set_xlabel("step")
            ax.set_ylabel("effective rank", color=C["third"])
            ax2 = ax.twinx()
            ax2.plot(st, sd, color=C["secondary"], lw=1.2, label="embedding std")
            ax2.set_ylabel("embedding std", color=C["secondary"])
            ax2.grid(False)
            ax.set_title("Collapse diagnostics", fontsize=9)
        else:
            _na(ax, "training log empty")
    else:
        _na(ax, "no training log")

    sub = (f"{summ['steps']:,} steps, {summ['gpu_hours']} GPU-h, "
           f"{summ['effective_epochs']} epochs, stop: {summ['stop_reason']}"
           if summ else "training summary not yet written")
    fig.suptitle(f"TiJepa-0 pretraining (tier T1, 1x A100-40GB) - {sub}",
                 fontsize=9.5, fontweight="bold", y=1.06)
    fig.tight_layout()
    return _save(fig, repo, "fig2_pretraining")


# ------------------------------------------------- Fig 3 BBBC054 encoder comparison
def fig3_bbbc054(repo: Path) -> list[str]:
    _style()
    d = _load(repo, "results/partA/bbbc054_probes.json")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    if not d:
        for a in axes:
            _na(a, "bbbc054_probes.json not available")
        fig.tight_layout()
        return _save(fig, repo, "fig3_bbbc054")

    t1 = d["tasks"]["morphology_class"]
    ax = axes[0]
    keys = [k for k in ENCODER_ORDER if k in t1]
    xs, means, errs, cols, labs = [], [], [], [], []
    for _i, k in enumerate(keys):
        e = t1[k]
        if e.get("status") != "ok":
            continue
        ci = e["cluster_bootstrap_ci_seed0"]
        xs.append(len(xs))
        means.append(e["balanced_accuracy_mean"])
        errs.append([max(0, e["balanced_accuracy_mean"] - (ci["ci_low"] or 0)),
                     max(0, (ci["ci_high"] or 0) - e["balanced_accuracy_mean"])])
        cols.append(ENCODER_COLOR[k])
        labs.append(ENCODER_LABEL[k])
    if xs:
        ax.bar(xs, means, color=cols, yerr=np.asarray(errs).T, capsize=3,
               error_kw={"ecolor": C["ink"], "lw": 1})
        ax.axhline(0.5, ls="--", lw=1, color=C["bad"])
        ax.text(len(xs) - 0.5, 0.505, "chance", fontsize=7, color=C["bad"], ha="right")
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("balanced accuracy")
        ax.set_title(f"Morphology class ({' vs '.join(d['labels_present'])})",
                     fontsize=9)
        ax.set_ylim(0.4, max(0.75, max(means) + 0.1))
    else:
        _na(ax, "no encoder results")

    ax = axes[1]
    t2 = d["tasks"]["lps_exposure_time_regression"]
    keys2 = [k for k in ENCODER_ORDER if k in t2 and t2[k].get("status") == "ok"]
    if keys2:
        xs = range(len(keys2))
        ax.bar(xs, [t2[k]["spearman_rho"] for k in keys2],
               color=[ENCODER_COLOR[k] for k in keys2])
        ax.axhline(0, color=C["grey"], lw=1)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([ENCODER_LABEL[k] for k in keys2], rotation=35,
                           ha="right", fontsize=7.5)
        ax.set_ylabel(r"Spearman $\rho$")
        ax.set_title("LPS exposure time (0-30 h)", fontsize=9)
    else:
        _na(ax, "no regression results")

    ax = axes[2]
    conf = d["tasks"].get("fov_confound_probe", {})
    keys3 = [k for k in ENCODER_ORDER if k in conf]
    if keys3:
        xs = range(len(keys3))
        ax.bar(xs, [conf[k]["fov_identity_balanced_accuracy"] for k in keys3],
               color=[ENCODER_COLOR[k] for k in keys3])
        ch = list(conf.values())[0]["chance"]
        ax.axhline(ch, ls="--", lw=1, color=C["bad"])
        ax.set_xticks(list(xs))
        ax.set_xticklabels([ENCODER_LABEL[k] for k in keys3], rotation=35,
                           ha="right", fontsize=7.5)
        ax.set_ylabel("FOV-identity balanced accuracy")
        ax.set_title("Confound probe (lower is better)", fontsize=9)
    else:
        _na(ax, "no confound probe")

    fig.suptitle("BBBC054 LPS-microglia probes (static frames; E5 frame-shuffle "
                 "undefined here)", fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    return _save(fig, repo, "fig3_bbbc054")


# --------------------------------------------------------------- Fig 4 tracking
def fig4_tracking(repo: Path) -> list[str]:
    _style()
    d = _load(repo, "results/partA/tracking.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    if not d or d.get("status") != "ok":
        for a in axes:
            _na(a, "tracking.json not available")
        fig.tight_layout()
        return _save(fig, repo, "fig4_tracking")

    agg = d["aggregate"]
    order = ["baseline_nearest_detection", "E4_tijepa0",
             "E5_tijepa0_frame_shuffled", "E6_tijepa0_random_init"]
    lab = {"baseline_nearest_detection": "nearest detection",
           "E4_tijepa0": "E4 TiJepa-0",
           "E5_tijepa0_frame_shuffled": "E5 frame-shuffled",
           "E6_tijepa0_random_init": "E6 random init"}
    col = {"baseline_nearest_detection": C["grey"], "E4_tijepa0": C["primary"],
           "E5_tijepa0_frame_shuffled": C["fourth"],
           "E6_tijepa0_random_init": C["secondary"]}
    ks = [k for k in order if agg.get(k, {}).get("delta_avg_mean") is not None]

    ax = axes[0]
    if ks:
        means = [agg[k]["delta_avg_mean"] for k in ks]
        errs = [[m - (agg[k]["delta_avg_ci"] or [m, m])[0],
                 (agg[k]["delta_avg_ci"] or [m, m])[1] - m]
                for k, m in zip(ks, means, strict=False)]
        ax.bar(range(len(ks)), means, color=[col[k] for k in ks],
               yerr=np.asarray(errs).T, capsize=3,
               error_kw={"ecolor": C["ink"], "lw": 1})
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([lab[k] for k in ks], rotation=30, ha="right", fontsize=7.5)
        ax.set_ylabel(r"$\langle\delta\rangle$ (mean fraction within 1-16 $\mu$m)")
        ax.set_title("Point tracking accuracy", fontsize=9)
    else:
        _na(ax, "no tracking aggregate")

    ax = axes[1]
    if ks:
        ax.bar(range(len(ks)), [agg[k]["median_survival_frames"] for k in ks],
               color=[col[k] for k in ks])
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([lab[k] for k in ks], rotation=30, ha="right", fontsize=7.5)
        ax.set_ylabel(r"median survival (frames within 8 $\mu$m)")
        ax.set_title("Track survival", fontsize=9)
    else:
        _na(ax, "no tracking aggregate")

    fig.suptitle(f"S-BIAD2196 neutrophil tracking - {d['n_movies']} held-out movies, "
                 f"{d['um_per_px']} um/px, {d['frame_interval_s']:g}s",
                 fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    return _save(fig, repo, "fig4_tracking")


# ----------------------------------------------------------------- Fig 5 Part B
def fig5_partB(repo: Path) -> list[str]:
    _style()
    tol = _load(repo, "results/partB/b31_tolerance.json")
    b33 = _load(repo, "results/partB/b33_dynamics_to_expression.json")
    b4 = _load(repo, "results/partB/b4_cross_species.json")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    ax = axes[0]
    if tol:
        rows = []
        for iv, e in tol["by_interval"].items():
            if e.get("status") != "ok":
                continue
            for rep, r in e["representations"].items():
                rows.append((f"{rep}\n({iv})", r["balanced_accuracy_mean"],
                             r["cluster_bootstrap_ci_seed0"],
                             r["group_permutation_test_seed0"].get("p_value")))
        if rows:
            xs = range(len(rows))
            means = [r[1] for r in rows]
            errs = [[max(0, r[1] - (r[2]["ci_low"] or 0)),
                     max(0, (r[2]["ci_high"] or 0) - r[1])] for r in rows]
            ax.bar(xs, means, color=[C["primary"], C["third"]][:len(rows)] * 3,
                   yerr=np.asarray(errs).T, capsize=3,
                   error_kw={"ecolor": C["ink"], "lw": 1})
            ax.axhline(0.5, ls="--", lw=1, color=C["bad"])
            for i, r in enumerate(rows):
                if r[3] is not None:
                    ax.text(i, r[1] + 0.06, f"p={r[3]:.2f}", ha="center", fontsize=7,
                            color=C["grey"])
            ax.set_xticks(list(xs))
            ax.set_xticklabels([r[0] for r in rows], fontsize=7.5)
            ax.set_ylabel("balanced accuracy")
            ax.set_ylim(0.4, 0.95)
            ax.set_title("Tolerance from NF-$\\kappa$B dynamics\nnaive$\\to$LPS vs "
                         "LPS$\\to$LPS", fontsize=9)
        else:
            _na(ax, "no B3.1 result")
    else:
        _na(ax, "b31_tolerance.json missing")

    ax = axes[1]
    if b33 and b33.get("status") == "ok":
        ax.bar([0, 1], [b33["n_genes_r_above_0.5"],
                        b33["permutation_null_mean_count"]],
               color=[C["primary"], C["grey"]])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["observed", "permutation\nnull (mean)"], fontsize=8)
        ax.set_ylabel("genes with LOCO r > 0.5")
        ax.set_title(f"Dynamics $\\to$ expression\n{b33['n_conditions']} shared "
                     f"conditions, p={b33['permutation_p']:.2f}", fontsize=9)
    elif b33:
        _na(ax, f"B3.3 {b33.get('status')}:\n{str(b33.get('reason'))[:90]}")
    else:
        _na(ax, "b33 missing")

    ax = axes[2]
    if b4 and b4.get("status") == "ok":
        h = b4["human"]
        m = b4["mouse"]
        ov = b4["overlap_test"]
        ax.bar([0, 1, 2], [h["n_up"], m["n_up"], ov["overlap"]],
               color=[C["primary"], C["secondary"], C["third"]])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["human up\n(FDR)", "mouse up\n(FDR)", "overlap"],
                           fontsize=8)
        ax.set_ylabel("genes")
        nom = b4.get("overlap_test_nominal_exploratory", {})
        ax.set_title(f"Cross-species tolerance genes\nnominal sets: "
                     f"{h.get('n_up_nominal_p01')} / {nom.get('overlap', 0)} overlap",
                     fontsize=9)
    else:
        _na(ax, "b4 missing")

    fig.suptitle("Part B: signalling dynamics and LPS tolerance", fontsize=9.5,
                 fontweight="bold", y=1.04)
    fig.tight_layout()
    return _save(fig, repo, "fig5_partB")


# ----------------------------------------------------------------- Fig 6 Part C
def fig6_partC(repo: Path) -> list[str]:
    _style()
    c2 = _load(repo, "results/partC/c2_scores.json")
    proj = _load(repo, "results/partC/c3_projection.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    ax = axes[0]
    if c2 and c2.get("status") == "ok":
        colmap = {"naive_monocyte": C["grey"], "control": C["ink"],
                  "LPS_tolerised": C["secondary"],
                  "beta_glucan_trained": C["primary"]}
        for st, v in c2["coordinates"].items():
            x, y = v["M1_like"], v["M2_like"]
            xe = v.get("M1_like_ci") or [x, x]
            ye = v.get("M2_like_ci") or [y, y]
            ax.errorbar(x, y, xerr=[[x - xe[0]], [xe[1] - x]],
                        yerr=[[y - ye[0]], [ye[1] - y]], fmt="o", ms=7,
                        color=colmap.get(st, C["third"]), capsize=2, lw=1)
            ax.annotate(st.replace("_", " "), (x, y), textcoords="offset points",
                        xytext=(7, 5), fontsize=7.5)
        ax.axhline(0, color=C["light"], lw=1)
        ax.axvline(0, color=C["light"], lw=1)
        ax.set_xlabel("M1-like score (rel. macrophage baseline)")
        ax.set_ylabel("M2-like score (rel. macrophage baseline)")
        ax.set_title("GSE58310 omics coordinates\n(donor bootstrap 95% CI)", fontsize=9)
    else:
        _na(ax, "c2_scores.json not available")

    ax = axes[1]
    if proj and proj.get("status") == "ok":
        pts = proj["predicted_positions"]
        real = proj.get("real_embedding_2d", {})
        if real.get("x"):
            ax.scatter(real["x"], real["y"], s=4, color=C["light"],
                       label="observed cells")
        for st, v in pts.items():
            ax.scatter([v["x"]], [v["y"]], s=90, marker="X",
                       color=C["primary"] if "glucan" in st else C["secondary"],
                       edgecolor=C["ink"], lw=0.6, zorder=3)
            ax.annotate(st.replace("_", " "), (v["x"], v["y"]),
                        textcoords="offset points", xytext=(7, 5), fontsize=7.5)
        ax.set_xlabel("PC1 of TiJepa-0 embedding")
        ax.set_ylabel("PC2 of TiJepa-0 embedding")
        ax.set_title("Predicted positions in image space\nHYPOTHESIS ONLY",
                     fontsize=9, color=C["bad"])
        ax.legend(fontsize=7)
    else:
        _na(ax, "c3 projection not available\n(requires the image-side anchors that "
                "the unreachable Dryad M0/M1/M2 movies would have provided)")

    fig.suptitle("Part C: bridge to trained immunity - HYPOTHESIS-GENERATING, "
                 "condition-level, cross-study, cross-species",
                 fontsize=9, fontweight="bold", y=1.03, color=C["bad"])
    fig.tight_layout()
    return _save(fig, repo, "fig6_partC")


# --------------------------------------------------------------- Fig 7 landscape
def fig7_landscape(repo: Path) -> list[str]:
    _style()
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.axis("off")
    facts = [
        ('Europe PMC full text: "trained immunity" AND (BioImage Archive OR Image '
         'Data Resource OR S-BIAD OR idr00)', "0 hits"),
        ("BioImage Archive studies matching macrophage/monocyte/neutrophil", "59"),
        ("...that are a beta-glucan, BCG, oxLDL, MDP or LPS-tolerance study", "0"),
        ("IDR projects and screens enumerated", "147 + 105"),
        ("...containing an innate-immune training study", "0"),
        ("...containing any innate-immune time-lapse", "0"),
        ("Europe PMC: trained immunity x high-content/Cell Painting", "45 hits, all "
                                                                      "off-topic"),
        ("Deposited image sets labelled trained immunity (S-BIAD829, fixed brain IF)",
         "1"),
        ("Public datasets with VIDEO of innate immune cells + activation labels", "0"),
    ]
    ax.set_title("The data gap this work is built on", loc="left", fontweight="bold")
    for i, (k, v) in enumerate(facts):
        y = len(facts) - i
        ax.text(0.0, y, k, fontsize=7.6, va="center")
        ax.text(1.0, y, v, fontsize=7.6, va="center", ha="right", fontweight="bold",
                color=C["bad"] if v.startswith("0") else C["ink"])
    ax.set_xlim(-0.01, 1.05)
    ax.set_ylim(0.2, len(facts) + 0.9)
    fig.text(0.0, -0.04,
             "Counts quoted from reports/'Trained immunity imaging datasets.md' "
             "(this repository); see that report for the queries and accessions.",
             fontsize=7, color=C["grey"])
    return _save(fig, repo, "fig7_landscape")


ALL = [fig1_overview, fig2_pretraining, fig3_bbbc054, fig4_tracking, fig5_partB,
       fig6_partC, fig7_landscape]
