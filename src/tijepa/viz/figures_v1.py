"""v1 figures (PLAN_v1.md 10.3, fixes E4).

Conventions inherited from `figures.py`: colour-blind-safe palette, minimum 8 pt type,
dot-and-CI plots or bars starting at zero (never a truncated axis), and an explicit
"not available" panel instead of a plausible-looking placeholder when an input is
missing. Every number is read from `results/v1/`.

Eight figures:

  1. overview schematic - data sources, pretraining, evaluations, with paired and
     bridged links drawn differently so the one hypothesis-only link cannot be mistaken
     for a measured one;
  2. pretraining - corpus composition, QC and leakage exclusions, stride mix, loss by
     mask type and effective rank;
  3. the Dryad headline - or, in this run, an explicit panel saying it was not provided
     and what that costs;
  4. BBBC054 - encoder dots with CIs, the time-only baseline and within-bin scores
     drawn on the same axis so the confound controls are visible next to the claim;
  5. tracking and forecasting;
  6. Part B - positive controls, per-well tolerance, dynamics to RNA and to ATAC, B4;
  7. Part C - anchor sanity results, resting scores with naive monocytes separated, the
     restimulation test, and the C3 refusal;
  8. the landscape survey.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .figures import C, _load, _na, _save, _style  # noqa: E402

ENC_LABEL = {
    "E1_handcrafted_improved": "E1 hand-crafted+",
    "E2_imagenet_vit_s": "E2 ImageNet",
    "E3_dinov2_vit_s": "E3 DINOv2",
    "E4_tijepa1": "E4 TiJepa-1",
    "E4_tijepa0_only_available": "E4 TiJepa-1",
    "E4s_tijepa1_spatial_only": "E4s spatial-only",
    "E6_tijepa_random_init": "E6 random init",
    "E0_tijepa0_reference": "E0 TiJepa-0 (ref)",
}
ENC_COLOR = {
    "E1_handcrafted_improved": C["grey"], "E2_imagenet_vit_s": C["light"],
    "E3_dinov2_vit_s": C["third"], "E4_tijepa1": C["primary"],
    "E4_tijepa0_only_available": C["primary"],
    "E4s_tijepa1_spatial_only": C["fourth"],
    "E6_tijepa_random_init": C["secondary"], "E0_tijepa0_reference": C["ink"],
}


def _lab(k: str) -> str:
    return ENC_LABEL.get(k, k)


def fig1_overview(repo: Path) -> list[str]:
    _style()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.16,
                                   edgecolor=color, linewidth=1.3, zorder=2))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9,
                fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=7.5,
                zorder=3, linespacing=1.35)

    def arrow(x1, y1, x2, y2, style="solid", label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": C["ink"],
                                "linewidth": 1.2,
                                "linestyle": "solid" if style == "solid" else (0, (4, 3)),
                                "shrinkA": 2, "shrinkB": 2})
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, label, ha="center",
                    fontsize=7, color=C["grey"])

    box(0.2, 3.5, 2.5, 2.2, "Pretraining video",
        "ComplexEye neutrophils\n(CC BY 4.0, Zenodo)\n+ Leica reference movies\n"
        "QC, leakage exclusion\n4 temporal strides", C["primary"])
    box(3.3, 3.9, 2.4, 1.8, "TiJepa-1",
        "V-JEPA, ViT-S, bf16\nmixed masking:\nspatial tubes + FUTURE\n(latent forecaster)",
        C["third"])
    box(6.4, 4.3, 3.4, 1.4, "Part A evaluations",
        "BBBC054 morphology class\nS-BIAD2196 tracking\nlatent forecasting",
        C["secondary"])
    box(6.4, 2.5, 3.4, 1.4, "Part B",
        "single-cell NF-kB dynamics\n-> acute endotoxin tolerance\n-> RNA and ATAC",
        C["fourth"])
    box(6.4, 0.5, 3.4, 1.6, "Part C (hypothesis only)",
        "donor-matched activation anchors\nresting + RESTIMULATION scores\n"
        "no image paired to any sample", C["grey"])
    box(0.2, 0.7, 5.5, 2.4, "Molecular data (no images)",
        "GSE260996 RNA-seq (33)   GSE261647 ATAC (18)\n"
        "Tay NF-kB single-cell traces\n"
        "GSE46903 macrophage atlas   GSE58310 trained/tolerant\n"
        "GSE212282 + GSE185433 restimulation", C["secondary"])

    arrow(2.7, 4.6, 3.3, 4.8)
    arrow(5.7, 4.9, 6.4, 5.0, label="frozen features")
    arrow(5.7, 1.9, 6.4, 3.2, label="measured")
    arrow(5.7, 1.6, 6.4, 1.3, style="dashed", label="bridged, not paired")
    ax.plot([], [], color=C["ink"], lw=1.2, label="measured on the same cells")
    ax.plot([], [], color=C["ink"], lw=1.2, ls=(0, (4, 3)),
            label="bridged across studies: hypothesis only")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.08), ncol=2, fontsize=7.5)
    ax.set_title("TiJepa-1: what is measured and what is only bridged", fontsize=10)
    return _save(fig, repo, "fig1_overview_v1")


def fig2_pretraining(repo: Path) -> list[str]:
    _style()
    corpus = _load(repo, "results/v1/pretrain_corpus.json")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))

    ax = axes[0]
    if corpus and corpus.get("clips_by_source"):
        src = corpus["clips_by_source"]
        ks = list(src)
        ax.bar(range(len(ks)), [src[k] for k in ks],
               color=[C["primary"], C["third"], C["fourth"]][: len(ks)])
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([k.replace("complexeye_", "") for k in ks], fontsize=8)
        ax.set_ylabel("clips")
        ax.set_ylim(0)
        ax.set_title(f"Corpus: {corpus['n_clips']:,} clips\n"
                     f"{corpus['n_movies_ok']} movies, "
                     f"{corpus['total_gb']} GB", fontsize=9)
    else:
        _na(ax, "corpus summary not available")

    ax = axes[1]
    if corpus:
        labels = ["used", "QC", "leakage"]
        vals = [corpus.get("n_movies_ok", 0),
                corpus.get("n_movies_excluded_qc", 0),
                corpus.get("n_movies_excluded_leakage", 0)]
        ax.bar(range(3), vals, color=[C["third"], C["secondary"], C["bad"]])
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("movies")
        ax.set_ylim(0)
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        ax.set_title("Movies kept and excluded", fontsize=9)
    else:
        _na(ax, "not available")

    ax = axes[2]
    if corpus and corpus.get("clips_by_stride"):
        st = corpus["clips_by_stride"]
        ks = sorted(st, key=lambda s: int(s))
        ax.bar(range(len(ks)), [st[k] for k in ks], color=C["primary"])
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"x{k}" for k in ks], fontsize=8)
        ax.set_ylabel("clips")
        ax.set_ylim(0)
        ax.set_title(f"Temporal strides\n{corpus.get('static_fraction', 0):.1%} "
                     "static clips", fontsize=9)
    else:
        _na(ax, "stride mix not available")

    ax = axes[3]
    log = repo / "checkpoints" / "pretrain_log_tijepa1_seed0.csv"
    if log.exists():
        import csv as _csv
        rows = list(_csv.DictReader(log.open()))
        step = [int(r["step"]) for r in rows]

        def col(k):
            return [float(r[k]) if r.get(k) not in (None, "") else np.nan
                    for r in rows]
        ax.plot(step, col("loss"), color=C["ink"], lw=1.2, label="all batches")
        ax.plot(step, col("loss_spatial"), color=C["primary"], lw=1.0,
                label="spatial mask")
        ax.plot(step, col("loss_future"), color=C["secondary"], lw=1.0,
                label="future mask")
        ax.set_xlabel("step")
        ax.set_ylabel("L1 latent loss")
        ax.legend(fontsize=7)
        ax2 = ax.twinx()
        ax2.plot(step, col("effective_rank"), color=C["third"], lw=1.0, ls=":")
        ax2.set_ylabel("effective rank", color=C["third"])
        ax2.grid(False)
        ax.set_title("Loss by mask type, and collapse diagnostic", fontsize=9)
    else:
        _na(ax, "pretraining log not available")
    fig.tight_layout()
    return _save(fig, repo, "fig2_pretraining_v1")


def fig3_dryad(repo: Path) -> list[str]:
    _style()
    d = _load(repo, "results/v1/partA/dryad_benchmark.json")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axis("off")
    if d and d.get("status") == "not_provided":
        ax.text(0.5, 0.95, "Planned headline benchmark: NOT PROVIDED",
                ha="center", va="top", fontsize=11, fontweight="bold", color=C["bad"],
                transform=ax.transAxes)
        body = ("Dryad doi:10.5061/dryad.m0cfxpp8v - BMDM movies labelled M0/M1/M2 -\n"
                "is the only public dataset with ACTIVATION-STATE LABELS ON VIDEO.\n\n"
                "It is an author prerequisite: every automated endpoint serves a\n"
                "proof-of-work interstitial, which the plan forbids bypassing, so it\n"
                "had to be fetched by hand. It was not provided for this run.\n\n"
                "Consequences, none of them worked around:")
        ax.text(0.5, 0.82, body, ha="center", va="top", fontsize=8.5,
                transform=ax.transAxes, linespacing=1.5)
        for i, c in enumerate(d.get("consequences", [])[:4]):
            ax.text(0.04, 0.34 - i * 0.085, "- " + c, ha="left", va="top", fontsize=7.4,
                    transform=ax.transAxes, wrap=True)
    else:
        _na(ax, "Dryad benchmark status not available")
    return _save(fig, repo, "fig3_dryad_headline_v1")


def fig4_bbbc054(repo: Path) -> list[str]:
    _style()
    d = _load(repo, "results/v1/partA/bbbc054_v1.json")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    if not d:
        for a in axes:
            _na(a, "BBBC054 results not available")
        fig.tight_layout()
        return _save(fig, repo, "fig4_bbbc054_v1")

    for ax, task in zip(axes[:2], ("three_class", "round_vs_amoeboid"), strict=False):
        encs = [k for k, v in d["encoders"].items()
                if isinstance(v, dict) and f"{task}_mean" in v]
        encs.sort(key=lambda k: d["encoders"][k][f"{task}_mean"])
        y = np.arange(len(encs))
        means = [d["encoders"][k][f"{task}_mean"] for k in encs]
        sds = [d["encoders"][k].get(f"{task}_sd") or 0 for k in encs]
        wb = [d["encoders"][k].get(f"{task}_within_bin_mean") for k in encs]
        ax.errorbar(means, y, xerr=sds, fmt="o", color=C["ink"], ecolor=C["grey"],
                    capsize=3, ms=6, label="all FOVs (mean +/- sd over seeds)")
        for i, k in enumerate(encs):
            ax.plot(means[i], i, "o", color=ENC_COLOR.get(k, C["ink"]), ms=6)
        # Within-bin scores sit almost exactly on the all-FOV scores - that IS the
        # result - so they are drawn on a small y offset, otherwise they are invisible
        # underneath the dots and the panel looks as if the control is missing.
        ok = [(v, i) for i, v in enumerate(wb) if v is not None]
        if ok:
            ax.plot([v for v, _ in ok], [i + 0.22 for _, i in ok], "x",
                    color=C["fourth"], ms=7,
                    label="within time bin (offset for visibility)")
        tob = (d.get("time_only_baseline") or {}).get(task)
        if tob:
            ax.axvline(tob["mean"], color=C["bad"], ls="--", lw=1.2,
                       label=f"time only ({tob['mean']:.3f})")
        chance = 1 / 3 if task == "three_class" else 0.5
        ax.axvline(chance, color=C["grey"], ls=":", lw=1.0,
                   label=f"chance ({chance:.3f})")
        ax.set_yticks(y)
        ax.set_yticklabels([_lab(k) for k in encs], fontsize=8)
        ax.set_ylim(-0.8, len(encs) - 0.1)
        ax.set_xlabel("balanced accuracy")
        ax.set_xlim(0, 1)
        ax.set_title(f"{task.replace('_', ' ')}", fontsize=9)
        ax.legend(fontsize=6.2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=2)

    ax = axes[2]
    sc = d.get("scale_sensitivity")
    if sc and sc.get("three_class_balanced_accuracy"):
        ks = sorted(sc["three_class_balanced_accuracy"], key=float)
        ax.plot([float(k) for k in ks],
                [sc["three_class_balanced_accuracy"][k] for k in ks],
                "o-", color=C["primary"])
        ax.axvline(1.0, color=C["grey"], ls=":", lw=1.0)
        ax.set_xlabel("resample factor (1.0 = derived 0.65 um/px)")
        ax.set_ylabel("balanced accuracy")
        ax.set_ylim(0)
        ax.set_title("Scale sensitivity", fontsize=9)
    else:
        _na(ax, "scale-sensitivity analysis\nnot run in this pass\n"
                "(pass --scale-sensitivity)")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    return _save(fig, repo, "fig4_bbbc054_v1")


def fig5_tracking_forecast(repo: Path) -> list[str]:
    _style()
    d = _load(repo, "results/v1/partA/tracking_forecast_v1.json")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    if not d:
        for a in axes:
            _na(a, "tracking/forecasting results not available")
        fig.tight_layout()
        return _save(fig, repo, "fig5_tracking_forecast_v1")

    agg = d.get("tracking", {}).get("aggregate", {})
    ax = axes[0]
    ks = [k for k, v in agg.items() if isinstance(v, dict) and "delta_avg_mean" in v]
    ks.sort(key=lambda k: agg[k]["delta_avg_mean"])
    if ks:
        vals = [agg[k]["delta_avg_mean"] for k in ks]
        ax.barh(range(len(ks)), vals,
                color=[C["bad"] if k.startswith("baseline") else
                       ENC_COLOR.get(k, C["primary"]) for k in ks])
        for i, k in enumerate(ks):
            pm = agg[k].get("per_movie_delta_avg") or {}
            if pm:
                ax.plot(list(pm.values()), [i] * len(pm), "|", color=C["ink"], ms=8)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([_lab(k) for k in ks], fontsize=7.5)
        ax.set_xlabel(r"$\langle\delta\rangle$ (1-16 um), per-movie ticks")
        ax.set_xlim(0)
        ax.set_title("Point tracking, 6 movies (no CI: < 10 groups)", fontsize=9)
    else:
        _na(ax, "no tracking aggregate")

    ax = axes[1]
    surv = [(k, agg[k]["median_survival_frames"]) for k in ks
            if "median_survival_frames" in agg[k]]
    if surv:
        ax.barh(range(len(surv)), [v for _, v in surv],
                color=[C["bad"] if k.startswith("baseline") else
                       ENC_COLOR.get(k, C["primary"]) for k, _ in surv])
        ax.set_yticks(range(len(surv)))
        ax.set_yticklabels([_lab(k) for k, _ in surv], fontsize=7.5)
        ax.set_xlabel("median track survival (frames)")
        ax.set_xlim(0)
        ax.set_title("Track survival", fontsize=9)
    else:
        _na(ax, "no survival data")

    ax = axes[2]
    fc = (d.get("forecasting") or {}).get("summary") or {}
    names = [k for k, v in fc.items()
             if v.get("mean_ratio_vs_copy_last") is not None]
    names.sort(key=lambda k: fc[k]["mean_ratio_vs_copy_last"])
    if names:
        vals = [fc[k]["mean_ratio_vs_copy_last"] for k in names]
        ax.barh(range(len(names)), vals,
                color=[ENC_COLOR.get(k, C["primary"]) for k in names])
        ax.axvline(1.0, color=C["bad"], ls="--", lw=1.2)
        ax.text(1.0, len(names) - 0.4, " copy-last baseline", fontsize=7,
                color=C["bad"], va="top")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([_lab(k) for k in names], fontsize=7.5)
        ax.set_xlabel("L1(predictor) / L1(copy-last)   lower is better")
        ax.set_xlim(0)
        ax.set_title("Latent forecasting (within-model ratios only)", fontsize=9)
    else:
        _na(ax, "no forecasting results")
    fig.tight_layout()
    return _save(fig, repo, "fig5_tracking_forecast_v1")


def fig6_partB(repo: Path) -> list[str]:
    _style()
    rna = _load(repo, "results/v1/partB/rna_positive_control.json")
    atac = _load(repo, "results/v1/partB/atac_positive_control.json")
    b3 = _load(repo, "results/v1/partB/b3_dynamics.json")
    b34 = _load(repo, "results/v1/partB/b33_b4.json")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))

    ax = axes[0]
    if rna and rna.get("genes"):
        g = [r for r in rna["genes"] if r.get("log2FC") is not None]
        ax.barh(range(len(g)), [r["log2FC"] for r in g], color=C["third"])
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([r["gene"] for r in g], fontsize=8)
        ax.axvline(1.0, color=C["grey"], ls=":", lw=1)
        ax.set_xlabel("log2FC, media->LPS vs media->media")
        ax.set_xlim(0)
        ax.set_title(f"RNA positive control {rna['n_passing']}/7\n"
                     f"{'PASS' if rna['passed'] else 'FAIL'}", fontsize=9)
    else:
        _na(ax, "RNA positive control not available")

    ax = axes[1]
    if atac and atac.get("genes"):
        g = [r for r in atac["genes"] if r.get("mean_delta") is not None]
        ax.barh(range(len(g)), [r["mean_delta"] for r in g], color=C["fourth"])
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([r["gene"] for r in g], fontsize=8)
        ax.axvline(atac["mean_delta_genome_wide"], color=C["bad"], ls="--", lw=1.2,
                   label="genome-wide")
        ax.set_xlabel("d logCPM accessibility, ML vs MM")
        ax.legend(fontsize=7)
        ax.set_title(f"ATAC positive control {atac['n_genes_increased']}/7\n"
                     f"{'PASS' if atac['passed'] else 'FAIL'}", fontsize=9)
    else:
        _na(ax, "ATAC positive control not available")

    ax = axes[2]
    tol = (b3 or {}).get("b31_tolerance", {})
    res = (tol.get("results") or {}).get("raw_trace", {})
    pw = res.get("per_well") or {}
    if pw:
        keys = sorted(pw)
        for i, k in enumerate(keys):
            rows = pw[k]
            fr = [r["fraction_predicted_tolerant"] for r in rows]
            true_lab = rows[0]["true_label"]
            ax.plot(fr, [i] * len(fr), "o",
                    color=C["secondary"] if true_lab == "LPS->LPS" else C["primary"],
                    ms=6)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([k.split("|")[-1] for k in keys], fontsize=7)
        ax.axvline(0.5, color=C["grey"], ls=":", lw=1)
        ax.set_xlabel("fraction of cells predicted tolerant")
        ax.set_xlim(0, 1)
        perm = res.get("exact_permutation") or {}
        ax.set_title("Acute endotoxin tolerance, per well\n"
                     f"exact p={perm.get('p_exact', float('nan')):.3f} "
                     f"(floor {perm.get('minimum_achievable_p', float('nan')):.3f})",
                     fontsize=9)
    else:
        _na(ax, "B3.1 per-well results not available")

    ax = axes[3]
    an = ((b34 or {}).get("b33") or {}).get("analyses") or {}
    rows = [(k, v) for k, v in an.items() if v.get("status") == "ok"]
    if rows:
        labels, obs, nulls = [], [], []
        for k, v in rows:
            labels.append(k.replace("_", "\n"))
            p = v["exact_condition_permutation"]
            obs.append(p["observed"])
            nulls.append(p["null_mean"])
        x = np.arange(len(labels))
        ax.bar(x - 0.2, obs, 0.4, label="observed", color=C["primary"])
        ax.bar(x + 0.2, nulls, 0.4, label="permutation null mean", color=C["grey"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel("targets with LOO r > 0.5")
        ax.set_ylim(0)
        ax.legend(fontsize=7)
        ps = ", ".join(f"p={v['exact_condition_permutation']['p_value']:.2f}"
                       for _, v in rows)
        ax.set_title(f"B3.3 dynamics -> omics\n{ps}", fontsize=9)
    else:
        _na(ax, "B3.3 results not available")
    fig.tight_layout()
    return _save(fig, repo, "fig6_partB_v1")


def fig7_partC(repo: Path) -> list[str]:
    _style()
    anc = _load(repo, "results/v1/partC/c1_anchors.json")
    re_ = _load(repo, "results/v1/partC/c3_restimulation.json")
    c3 = _load(repo, "results/v1/partC/c3_projection.json")
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))

    ax = axes[0]
    if anc and anc.get("anchors"):
        names = list(anc["anchors"])
        pairs = [anc["anchors"][n].get("n_donor_matched_pairs", 0) for n in names]
        colors = [C["third"] if anc["anchors"][n].get("status") == "ok" else C["bad"]
                  for n in names]
        ax.bar(range(len(names)), pairs, color=colors)
        ax.axhline(3, color=C["ink"], ls="--", lw=1.2, label="minimum 3 pairs")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("donor-matched pairs")
        ax.set_ylim(0)
        ax.legend(fontsize=7)
        ax.set_ylim(0, max(pairs) * 1.35 if pairs else 1)
        for i, n in enumerate(names):
            sr = anc["anchors"][n].get("sanity_rule")
            txt = (f"sanity {sr['n_found']}/{sr['required']}" if sr
                   else "cannot form")
            ax.text(i, pairs[i] + max(pairs) * 0.04, txt, ha="center", fontsize=7)
        ax.set_title("8.1 anchors: donor matching and sanity", fontsize=9)
    else:
        _na(ax, "anchor results not available")

    ax = axes[1]
    sc = (anc or {}).get("resting_scores") or {}
    per = sc.get("per_condition") or {}
    usable = (anc or {}).get("anchors_usable") or []
    if per and usable:
        a = usable[0]
        conds = [c for c in per if per[c].get(a, {}).get("mean") is not None]
        naive = [c for c in conds if "naive" in c]
        others = [c for c in conds if "naive" not in c]
        ordered = others + naive
        ypos = {c: i for i, c in enumerate(ordered)}
        for grp, col, mk, lbl in ((others, C["primary"], "o", "day-6 macrophages"),
                                  (naive, C["grey"], "s", "naive monocytes (apart)")):
            if grp:
                ax.plot([per[c][a]["mean"] for c in grp], [ypos[c] for c in grp],
                        mk, color=col, ms=7, linestyle="none", label=lbl)
        if naive:
            ax.axhline(len(others) - 0.5, color=C["light"], lw=1.2)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered, fontsize=7.5)
        ax.set_xlabel(f"{a} anchor score (resting day 6)")
        ax.set_title("8.2 resting scores\n(naive monocytes shown apart)", fontsize=9)
    else:
        _na(ax, "no sanity-passed anchor,\nso nothing to score")

    ax = axes[2]
    if re_:
        rows = []
        for ds in ("GSE212282", "GSE185433"):
            for k, v in ((re_.get(ds) or {}).get("contrasts") or {}).items():
                # sign_flip_test returns no "status" key, so test for the statistic
                # itself; filtering on a key that is never set left this panel empty.
                if "mean_difference" in v and k.endswith("mean_log_expression"):
                    rows.append((k.split("_vs_")[0], v["mean_difference"],
                                 v.get("matches_expectation", False)))
        if rows:
            rows.sort(key=lambda r: r[1])
            ax.barh(range(len(rows)), [r[1] for r in rows],
                    color=[C["third"] if r[2] else C["bad"] for r in rows])
            ax.axvline(0, color=C["ink"], lw=1)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([r[0].replace("_restim", "") for r in rows], fontsize=7)
            ax.set_xlabel("d mean log expression, LPS/NF-kB panel\nvs untrained->LPS")
            ax.set_title("8.3 restimulation test\n(green = predicted direction)",
                         fontsize=9)
        else:
            _na(ax, "restimulation contrasts not available")
    else:
        _na(ax, "restimulation test not run")

    ax = axes[3]
    ax.axis("off")
    if c3:
        ax.text(0.5, 0.95, "8.4 C3 projection: NOT PERFORMED", ha="center", va="top",
                fontsize=10, fontweight="bold", color=C["bad"],
                transform=ax.transAxes)
        y = 0.80
        for r in c3.get("blocking_reasons", []):
            ax.text(0.03, y, f"x {r['requirement']}", ha="left", va="top", fontsize=8,
                    fontweight="bold", transform=ax.transAxes)
            # wrap rather than slice: a sentence cut mid-clause reads as a rendering
            # bug and hides the reason the panel exists to give
            body = "\n".join(textwrap.wrap(r["detail"], 86)[:6])
            ax.text(0.04, y - 0.07, body, ha="left", va="top",
                    fontsize=6.4, transform=ax.transAxes, linespacing=1.45)
            y -= 0.36
        ax.text(0.03, 0.10, "v0's one-axis substitute is DELETED as circular "
                            "(PLAN_v1 8.5)", ha="left", va="top", fontsize=7,
                color=C["grey"], transform=ax.transAxes, wrap=True)
    else:
        _na(ax, "C3 status not available")
    fig.tight_layout()
    return _save(fig, repo, "fig7_partC_v1")


def fig8_landscape(repo: Path) -> list[str]:
    """The landscape survey: what the public record does and does not contain."""
    _style()
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    rows = [
        ("Trained-immunity omics\n(BCG, beta-glucan, oxLDL)", 1.0, C["third"]),
        ("Innate-immune time-lapse video\n(migration, phagocytosis)", 1.0, C["third"]),
        ("Label-free / FLIM imaging of\nmacrophage activation", 0.45, C["secondary"]),
        ("Video WITH activation-state labels\n(M0/M1/M2)", 0.12, C["secondary"]),
        ("Imaging of TRAINED or TOLERANT\ninnate immune cells", 0.0, C["bad"]),
    ]
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows])
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("availability in public repositories (schematic)")
    ax.text(0.02, 4, "none found", va="center", fontsize=8, color=C["bad"],
            fontweight="bold")
    ax.text(0.14, 3, "behind an access control in this run", va="center", fontsize=7.5,
            color=C["ink"])
    ax.set_title("The data gap this work is built around\n"
                 "(Europe PMC full-text search for \"trained immunity\" with any image "
                 "repository term: 0 hits)", fontsize=9)
    fig.tight_layout()
    return _save(fig, repo, "fig8_landscape_v1")


ALL = [fig1_overview, fig2_pretraining, fig3_dryad, fig4_bbbc054,
       fig5_tracking_forecast, fig6_partB, fig7_partC, fig8_landscape]
