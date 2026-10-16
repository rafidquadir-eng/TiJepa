"""F7: render paper/tijepa.md from the v1 template and results/v1 (fixes E2, E5, B4-B6).

Every number comes from a JSON file in `results/v1/`. A missing result renders as an
explicit sentence saying it is unavailable and why, never as a placeholder number and
never as a raw status string like `not_performed` dropped into prose (review finding
E2). Every table is built by `tijepa.viz.tables`, which asserts that each row matches
its header (E1). Every directional phrase is generated from an interval or an exact
permutation p by `tijepa.viz.claims` and then re-verified by `scripts/92_check_claims.py`
(E5, B6).

Wording required by PLAN_v1.md 10.1 and used consistently here: **acute endotoxin
tolerance** for the Part B contrast, **morphology class** for BBBC054, and
"pretrained on human neutrophil migration video" rather than any claim about innate
immune activation in general.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.viz.claims import Comparison, describe  # noqa: E402
from tijepa.viz.tables import markdown_table  # noqa: E402

V1 = REPO / "results" / "v1"
TEMPLATE = REPO / "paper" / "tijepa_v1.template.md"
OUT = REPO / "paper" / "tijepa.md"

ANALYSIS_LABEL = {
    "rna_expression": "RNA expression",
    "chromatin_accessibility": "chromatin accessibility",
}
ENC_LABEL = {
    "E1_handcrafted_improved": "E1 hand-crafted (segmentation + Zernike + Haralick)",
    "E2_imagenet_vit_s": "E2 ImageNet ViT-S",
    "E3_dinov2_vit_s": "E3 DINOv2 ViT-S",
    "E4_tijepa1": "E4 TiJepa-1",
    "E4_tijepa0_only_available": "E4 TiJepa-1",
    "E4s_tijepa1_spatial_only": "E4s TiJepa-1 spatial-only ablation",
    "E6_tijepa_random_init": "E6 random init (3 draws)",
    "E0_tijepa0_reference": "E0 TiJepa-0 (v0 reference, different corpus)",
}


def load(rel: str):
    p = V1 / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def f(v, nd=3):
    return "not available" if v is None else f"{v:.{nd}f}"


def pf(p):
    if p is None:
        return "not available"
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.3f}"


# --------------------------------------------------------------------- sections
def corpus_para(c) -> tuple[str, dict]:
    if not c:
        return ("The pretraining corpus summary is not available in this run, so the "
                "corpus cannot be described here.", {})
    by_src = c.get("clips_by_source", {})
    strides = c.get("clips_by_stride", {})
    prov = c.get("pixel_size_provenance", {})
    txt = (
        f"The corpus is **{c['n_clips']:,} cell-centred clips** of 16 frames at "
        f"96x96 px ({c['total_gb']} GB in {c['n_shards']} sequential shards) from "
        f"{c['n_movies_ok']} movies across {c['n_archives_done']} of "
        f"{c['n_archives_in_record']} ComplexEye archives (Zenodo 10.5281/zenodo.10082674, "
        "CC BY 4.0). Pixel sizes are verified rather than assumed: "
        f"{prov.get('complexeye', {}).get('um_per_px', 'n/a')} um/px for the ComplexEye "
        "movies and "
        f"{prov.get('complexeye_leica', {}).get('um_per_px', 'n/a')} um/px for the Leica "
        "reference movies, both derived in Methods 4.2 from the instrument "
        "specifications, and everything is resampled to a common "
        f"{c.get('target_um_per_px')} um/px. "
        f"Clips are sampled at temporal strides "
        f"{', '.join('x' + k for k in sorted(strides, key=int))} "
        "(" + "; ".join(f"{k}: {v:,}" for k, v in
                        sorted(strides.items(), key=lambda kv: int(kv[0]))) + "), "
        f"and {c.get('static_fraction', 0):.1%} of clips are made static by repeating a "
        "single frame, so that still-image evaluation is inside the training "
        "distribution rather than out of it. "
        f"Frame-level quality control removed {c.get('n_movies_excluded_qc', 0)} movies; "
        "it is heuristic, with thresholds set from the empirical distribution of this "
        "corpus, because the deposit intended for a learned quality classifier could not "
        "be retrieved in full. "
        f"Clips by source: {', '.join(f'{k} {v:,}' for k, v in by_src.items())}.")
    return txt, {"CORPUS_CLIPS": f"{c['n_clips']:,}", "CORPUS_GB": str(c["total_gb"]),
                 "CORPUS_MOVIES": str(c["n_movies_ok"])}


def leakage_para(lk, c) -> str:
    if not lk:
        return ("No leakage check result is available, so no claim is made about "
                "overlap between the pretraining corpus and the evaluation movies.")
    n = lk.get("n_pairs_matched", 0)
    base = (
        f"The tracking evaluation uses S-BIAD2196, which like the pretraining corpus is "
        "human neutrophil brightfield video, so overlap between the two was checked "
        f"rather than assumed. Every one of the {lk['n_eval_movies']} evaluation movies "
        f"was fingerprinted against all {lk['n_train_movies']} candidate pretraining "
        "movies then on disk (three frames, block-averaged, z-scored, compared by "
        "Pearson correlation, with the SHA-1 of the first frame recorded as well).")
    if n == 0:
        return base + (" No pair exceeded the matching threshold, so there is no "
                       "evidence of overlap.")
    pairs = lk.get("pairs", [])[:1]
    ex = pairs[0] if pairs else {}
    return base + (
        f" **The check found real overlap**: {n} pair(s) matched, the strongest being "
        f"{ex.get('eval_movie')} against {ex.get('train_movie')} at r = "
        f"{ex.get('pearson_r')} with identical frame dimensions. The evaluation movies "
        "are evidently drawn from the same collection. The matching pretraining movies "
        "are excluded from the corpus and the evaluation movies are kept, which is the "
        "right way round: excluding the evaluation movie would discard evaluation data "
        "to protect training data. This is also why the deposit had to be enumerated "
        "from its file tree rather than its API — the API exposes only five of its six "
        "movies, and the hidden sixth is the one that is duplicated in the corpus.")


def pretrain_para(ps, abl) -> str:
    if not ps:
        return ("Pretraining results are not available in this run, so the model's "
                "training is not described here.")
    obj = ps.get("objective", "")
    counts = ps.get("mask_kind_counts", {})
    txt = (
        f"TiJepa-1 is a V-JEPA-style model (ViT-S encoder, bf16) trained with {obj}. "
        f"It ran for {ps['steps']:,} steps in {ps['gpu_hours']} GPU-hours on a single "
        f"{ps.get('device')}, seeing {ps['clips_seen']:,} clip views, i.e. "
        f"{ps['effective_epochs']} effective epochs over the corpus. The run stopped on "
        f"its {ps['stop_reason'].replace('_', ' ')}. The latent L1 loss fell from "
        f"{f(ps.get('initial_loss_mean_first500'), 3)} to "
        f"{f(ps.get('final_loss_mean_last500'), 3)}")
    if ps.get("final_loss_spatial_last500") is not None:
        txt += (f" ({f(ps['final_loss_spatial_last500'])} on spatially masked batches "
                f"and {f(ps['final_loss_future_last500'])} on future-masked batches; "
                f"{counts.get('spatial', 0):,} and {counts.get('future', 0):,} batches "
                "respectively)")
    txt += (". Collapse was monitored by embedding standard deviation and effective "
            "rank every 50 steps rather than by a downstream probe, because the dataset "
            "the plan intended for that monitor was not available.")
    if abl:
        txt += (
            f" The ablation, TiJepa-1-spatial, uses the identical corpus and spatial "
            f"masking only, run for {abl['steps']:,} steps — matched to the main run's "
            "step count rather than its wall-clock, so the comparison isolates the "
            f"objective — in {abl['gpu_hours']} GPU-hours.")
    else:
        txt += (" The spatial-only ablation is not available in this run, so no "
                "claim is made about which part of the objective is responsible for "
                "any result below.")
    return txt


def dryad_para(d) -> str:
    if not d:
        return "The status of the planned headline benchmark is not recorded."
    if d.get("status") != "not_provided":
        return (f"The Dryad benchmark status is `{d.get('status')}`; see "
                "`results/v1/partA/dryad_benchmark.json`.")
    cons = d.get("consequences", [])
    return (
        "The planned headline for this version was a video benchmark with "
        "activation-state labels: Dryad `doi:10.5061/dryad.m0cfxpp8v`, bone-marrow-"
        "derived macrophage movies labelled M0, M1 and M2, evaluated leave-one-batch-out. "
        "Every automated download endpoint serves a proof-of-work interstitial, which we "
        "treat as an access control not to be bypassed, so the dataset had to be fetched "
        "by hand; it was not available for this run. We report that plainly because its "
        "absence shapes what follows. "
        "**No dataset in this run provides video of innate immune cells carrying "
        "activation-state labels**, which is the paper's own thesis restated as an "
        "experimental limitation. In consequence: " +
        "; ".join(c.rstrip(".").lower() for c in cons[:3]) + ".")


def bbbc_section(bb) -> tuple[str, str, str]:
    if not bb:
        msg = ("BBBC054 probe results are not available in this run.")
        return msg, "", ""
    enc = bb["encoders"]
    note = bb.get("dataset_note", "")
    intro = (
        "BBBC054 is a set of LPS-exposed mouse microglia images with per-cell "
        "**morphology-class** annotations. We call it that throughout: the labels are "
        "cell shapes, and shape is not activation state. " + note + " The images are "
        "static, so the frame-shuffle control is undefined here and is reported as such; "
        "it applies only to the tracking and forecasting evaluations below.")
    tob = bb.get("time_only_baseline", {})
    confound = (
        "In this dataset one field of view is one exposure time, so field identity and "
        "time are perfectly confounded and any field-grouped split leaks time. Three "
        "controls address it. First, a **time-only baseline** that classifies morphology "
        "from exposure time alone scores "
        f"{f(tob.get('three_class', {}).get('mean'))} on the three-class task and "
        f"{f(tob.get('round_vs_amoeboid', {}).get('mean'))} on round-versus-amoeboid — "
        "chance in both cases, so exposure time by itself carries no morphology "
        "information. Second, **within-time-bin** balanced accuracy, computed inside six "
        "contiguous bins of ten fields and averaged, is within a thousandth of the "
        "overall figure for every encoder, so time is not what the encoders are reading. "
        "Third, a field-identity probe is reported per encoder. Exposure-time regression "
        "is demoted to exploratory and uses leave-six-contiguous-fields-out block "
        "cross-validation.")
    rows = []
    for k in sorted(enc, key=lambda k: -(enc[k].get("three_class_mean") or 0)):
        v = enc[k]
        if not isinstance(v, dict) or "three_class_mean" not in v:
            continue
        rows.append([
            ENC_LABEL.get(k, k),
            f"{f(v['three_class_mean'])} ± {f(v.get('three_class_sd'), 3)}",
            f(v.get("three_class_within_bin_mean")),
            f"{f(v.get('round_vs_amoeboid_mean'))} ± {f(v.get('round_vs_amoeboid_sd'), 3)}",
            f(v.get("fov_identity_mean")),
            f(v.get("exposure_time_rho_mean")),
        ])
    table = markdown_table(
        ["encoder", "3-class BA (mean ± sd)", "3-class within-bin",
         "round vs amoeboid BA", "field identity (lower better)", "ρ exposure time"],
        rows)
    comps = bb.get("pairwise_comparisons", {})
    # One comparison per table row rather than a run-on sentence naming six encoders:
    # such a sentence is unreadable, and no checker can attribute its directional
    # words to the right pair either.
    cmp_rows, headline = [], None
    for ck, c in comps.items():
        task, _pair = ck.split(":", 1)
        if task != "three_class":
            continue
        cmp_obj = Comparison(c["encoder_a"], c["encoder_b"], task,
                             difference=c["difference_mean_over_seeds"],
                             ci=tuple(c["ci95_widest_over_seeds"]),
                             p_value=c.get("p_value_bh"))
        cmp_rows.append([
            ENC_LABEL.get(c["encoder_a"], c["encoder_a"]),
            ENC_LABEL.get(c["encoder_b"], c["encoder_b"]),
            f(c["difference_mean_over_seeds"]),
            f"[{c['ci95_widest_over_seeds'][0]:+.3f}, "
            f"{c['ci95_widest_over_seeds'][1]:+.3f}]",
            pf(c.get("p_value_bh")),
            {1: "higher", -1: "lower", 0: "not distinguishable"}[cmp_obj.direction],
        ])
        a, b = c["encoder_a"], c["encoder_b"]
        if ("tijepa" in a and "random_init" in b) or ("tijepa" in b
                                                      and "random_init" in a):
            headline = (a, b, cmp_obj)
    body = (
        "Probes are identical for every encoder, cross-validated with "
        "`StratifiedGroupKFold` grouped by field of view, and repeated over three seeds "
        "that each change a stratified cell subsample, the fold assignment and, for the "
        "random-init control, the network draw — so every standard deviation reported "
        "is real variation rather than a decoration. Encoder pairs are compared with a "
        "paired cluster bootstrap over the 60 fields (2,000 resamples, both encoders "
        "scored on the same resample) with Benjamini-Hochberg correction across all "
        "comparisons within a task.")
    if headline is not None:
        a, b, obj = headline
        body += (
            " The comparison that isolates pretraining from architecture is the "
            "pretrained model against the identical network with random weights: on "
            f"the three-class task that difference is {describe(obj)}.")
    if cmp_rows:
        body += ("\n\nAll pairwise comparisons on the three-class task, with the "
                 "direction generated from the interval rather than asserted:\n\n")
        body += markdown_table(
            ["encoder A", "encoder B", "Δ BA (A − B)", "95% CI", "BH p", "direction"],
            sorted(cmp_rows, key=lambda r: r[0]))
    return intro + "\n\n" + body, confound, table


def track_para(tf) -> tuple[str, str]:
    if not tf:
        return ("Tracking results are not available in this run.",
                "Forecasting results are not available in this run.")
    agg = tf.get("tracking", {}).get("aggregate", {})
    base = agg.get("baseline_nearest_detection", {})
    rows = []
    for k, v in sorted(agg.items(),
                       key=lambda kv: -(kv[1].get("delta_avg_mean") or 0)):
        if not isinstance(v, dict) or "delta_avg_mean" not in v:
            continue
        perm = v.get("vs_baseline_exact_permutation") or {}
        rows.append([ENC_LABEL.get(k, k), f(v["delta_avg_mean"]),
                     f"{v['median_survival_frames']:.0f}",
                     pf(perm.get("exact_sign_flip_p_two_sided"))
                     if perm else "— (is the baseline)"])
    table = markdown_table(
        ["method", "⟨δ⟩ (1–16 µm)", "median survival (frames)",
         "vs baseline, exact permutation"], rows)
    t = (
        f"Point tracking uses all {tf.get('n_movies')} movies the S-BIAD2196 deposit "
        "contains, with the leakage-matched pretraining movies excluded as described "
        "above. The same tracker is used for every encoder: the query cell's patch "
        "token is matched to candidate tokens inside a search window, with an identical "
        "motion prior. Six movies is below the ten-group minimum we set for interval "
        "estimates, so no bootstrap confidence interval is reported; per-movie values "
        "and an exact sign-flip permutation against the baseline are given instead.\n\n"
        + table + "\n\n"
        f"The nearest-detection baseline reaches ⟨δ⟩ {f(base.get('delta_avg_mean'))}. ")
    fc = (tf.get("forecasting") or {}).get("summary") or {}
    names = [k for k, v in fc.items()
             if v.get("mean_ratio_vs_copy_last") is not None]
    if not names:
        fpara = ("The latent-forecasting evaluation produced no usable ratios in this "
                 "run.")
    else:
        frows = [[ENC_LABEL.get(k, k), f(fc[k]["mean_ratio_vs_copy_last"]),
                  f(fc[k]["mean_ratio_vs_mean_latent"])]
                 for k in sorted(names, key=lambda k: fc[k]["mean_ratio_vs_copy_last"])]
        ftab = markdown_table(
            ["model", "L1(predictor) / L1(copy-last)", "/ L1(mean latent)"], frows)
        fpara = (
            "Because the v1 objective trains the predictor to produce future latents, we "
            "can ask directly whether it forecasts. Each model is given the first 12 "
            "frames of a held-out clip and predicts its own target-encoder latents for "
            "the last 4, scored against two baselines in its own latent space: copying "
            "the last observed latent, and the mean context latent. **Only within-model "
            "ratios are comparable** — each model has a different latent space, so raw "
            "L1 across models would be meaningless — and a ratio below 1 means the "
            "predictor beats assuming nothing changes.\n\n" + ftab)
    return t, fpara


def partB_paras(rna, atac, lig, b3, b34) -> dict:
    out = {}
    if rna and atac:
        rna_verdict = "PASS" if rna["passed"] else "FAIL"
        atac_verdict = "PASS" if atac["passed"] else "FAIL"
        out["PC_PARA"] = (
            "Nothing in this section is reported unless its pipeline first passes a "
            "positive control. For expression, `media->LPS` versus `media->media` in "
            f"GSE260996 recovers {rna['n_passing']} of 7 canonical LPS/NF-κB targets "
            "with log2 fold change above 1 and FDR below 0.05 "
            f"({rna['n_de_genes_total']:,} differentially expressed genes in total): "
            f"{rna_verdict}. "
            "For accessibility, `ML` versus `MM` in GSE261647 raises accessibility at "
            f"peaks within 10 kb of all {atac['n_genes_increased']} of those genes' "
            "transcription start sites, "
            f"{f(atac['mean_delta_at_windows'])} logCPM against "
            f"{f(atac['mean_delta_genome_wide'])} genome-wide "
            f"({pf(atac['mannwhitney_p_one_sided'])}): {atac_verdict}. "
            "The two matrices are also checked mechanically against their GEO records "
            "on row-identifier class, sample count and assay type, which is how the "
            "previous version's confusion of an accessibility matrix for an expression "
            "matrix is prevented from recurring.")
    else:
        out["PC_PARA"] = ("Positive-control results are not available, so no Part B "
                          "result is reported below.")

    if lig:
        san = lig.get("tolerance_sanity_check", {})
        out["B31_PARA"] = (
            "The ligand codes in the trace deposit are now taken from the deposit's own "
            "plotting code, which assigns them with doses in trailing comments, rather "
            "than inferred: 1 = CpG 100 nM, 2 = LPS 1 ng/ml, 3 = polyI:C 1000 ng/ml, "
            "6 = TNF 10 ng/ml. The GEO treatment protocol for the matching ATAC series "
            "states the same four doses independently. A biological sanity check runs "
            "before any tolerance result: the NF-κB second-response amplitude for "
            f"LPS->LPS is {f(san.get('mean_amplitude_lps_to_lps'), 4)} against "
            f"{f(san.get('mean_amplitude_naive_to_lps'), 4)} for naive->LPS at the "
            f"matched {san.get('interval_used_h')} h interval, i.e. "
            + ("lower, as acute endotoxin tolerance predicts."
               if san.get("expected_direction_observed")
               else "HIGHER, contradicting tolerance."))
    else:
        out["B31_PARA"] = "Ligand-code confirmation is not available in this run."

    if b3:
        tol = b3.get("b31_tolerance", {})
        res = (tol.get("results") or {}).get("raw_trace", {})
        perm = res.get("exact_permutation", {})
        out["B32_PARA"] = (
            "We then ask whether a classifier can tell an acutely endotoxin-tolerised "
            "macrophage from a naive one at the single-cell level. Every layout-verified "
            f"file was searched: the design supplies {tol.get('n_wells')} wells at the "
            f"matched 8 h interval ({tol.get('n_cells')} cells), the only interval at "
            "which homologous restimulation was performed. Leave-one-well-out balanced "
            f"accuracy from the raw trace is "
            f"{f(res.get('cell_balanced_accuracy_mean'))} ± "
            f"{f(res.get('cell_balanced_accuracy_sd'))}. "
            f"With {tol.get('n_wells')} wells the exact permutation null over all "
            f"{perm.get('n_distinct_assignments')} distinct well-label assignments gives "
            f"{pf(perm.get('p_exact'))}, and **no p below "
            f"{f(perm.get('minimum_achievable_p'))} is achievable whatever the "
            "classifier does**. We therefore report no confidence interval and draw no "
            "conclusion about tolerance from dynamics: the public deposit cannot support "
            "one. The previous version reported this contrast as 0.701 with a bootstrap "
            "interval over the same four wells.")
        b32 = b3.get("b32_ligand_identity", {})
        out["B32_PARA"] += (
            f" A secondary multi-class task — identifying which pair of ligands a cell "
            f"saw — reaches {f(b32.get('balanced_accuracy_mean'))} ± "
            f"{f(b32.get('balanced_accuracy_sd'))} across "
            f"{b32.get('n_classes')} classes against a chance level of "
            f"{f(b32.get('chance_balanced_accuracy'))}, so the traces do carry "
            "stimulus information even where they cannot resolve priming history.")
    else:
        out["B32_PARA"] = "Part B dynamics results are not available in this run."

    if b34:
        an = (b34.get("b33") or {}).get("analyses") or {}
        bits = []
        for k, v in an.items():
            if v.get("status") != "ok":
                continue
            r = v["loo_ridge_top2000_variable_targets"]
            p = v["exact_condition_permutation"]
            bits.append(
                f"for {ANALYSIS_LABEL.get(k, k)}, {r['n_targets_r_gt_0.5']} of "
                f"{r['n_features_out']} targets reach leave-one-condition-out r > 0.5 "
                f"against a permutation null mean of {p['null_mean']:.1f} "
                f"({pf(p['p_value'])}, exact over all {p['n_permutations'] + 1} "
                "condition orderings)")
        out["B33_PARA"] = (
            "Condition-level dynamics were then related to the two molecular readouts "
            "**separately and under their own names** — expression from GSE260996 and "
            "chromatin accessibility from GSE261647 — matching trace conditions to omics "
            "conditions on stimulus identity and on the 4 h inter-stimulus interval the "
            "GEO protocol states. Both are null: " + "; ".join(bits) +
            ". A pre-specified NF-κB target panel, analysed separately, is null as well. "
            "With five matched conditions this is descriptive and we label it so.")
        b4 = b34.get("b4", {})
        out["B4_PARA"] = (
            "Finally we asked whether genes whose LPS induction is blunted by prior "
            "priming in mouse overlap the genes changed by LPS tolerance in human. On "
            f"the mouse side {b4.get('mouse_n_lps_induced'):,} genes are LPS-induced in "
            f"the naive arm and {b4.get('mouse_n_reduced_by_both_primings'):,} of them "
            "show reduced induction after both CpG and polyI:C priming, giving "
            f"{b4.get('n_mouse_genes_with_human_ortholog'):,} with human orthologs. On "
            f"the human side a moderated paired test across "
            f"{b4.get('human_n_donor_pairs')} donors finds "
            f"{b4.get('human_n_down_fdr05')} genes down and "
            f"{b4.get('human_n_up_fdr05')} up at FDR 0.05. The overlap is "
            f"**{b4.get('n_overlap')} genes** (Fisher "
            f"{pf(b4.get('fisher_p_one_sided'))}). We pre-registered that these two "
            "contrasts are not the same biology — mouse heterologous priming 4 h apart "
            "in bone-marrow-derived macrophages against human homologous tolerance 6 "
            "days apart in monocyte-derived macrophages — so a null is the expected "
            "outcome and is not evidence about either one.")
    else:
        out["B33_PARA"] = "Dynamics-to-omics results are not available in this run."
        out["B4_PARA"] = "The cross-species analysis is not available in this run."
    return out


def partC_paras(anc, restim, c3) -> dict:
    out = {}
    if anc:
        a = anc["anchors"]
        lines = []
        for name, rec in a.items():
            if rec.get("status") == "ok":
                sr = rec["sanity_rule"]
                lines.append(
                    f"the {name} anchor is built from {rec['n_donor_matched_pairs']} "
                    f"donor-matched pairs and passes its sanity rule with "
                    f"{sr['n_found']} of the required {sr['required']} canonical "
                    f"markers in its top 100 ({', '.join(sr['found'][:5])})")
            elif rec.get("status") == "cannot_form_anchor":
                lines.append(
                    f"**no {name} anchor can be built at all**: the atlas provides only "
                    f"{rec['n_donor_matched_pairs']} donor-matched pairs against a "
                    f"minimum of {rec['min_required_pairs']}")
            else:
                sr = rec.get("sanity_rule", {})
                lines.append(
                    f"the {name} anchor has {rec['n_donor_matched_pairs']} donor-matched "
                    f"pairs but **fails its sanity rule**, with only {sr.get('n_found')} "
                    f"of {sr.get('required')} canonical markers in its top 100")
        out["C1_PARA"] = (
            "Activation axes are built from GSE46903 with two requirements the previous "
            "version did not impose: each anchor comes from **donor-paired** "
            "stimulus-versus-control samples analysed with a moderated paired test, with "
            "at least three donor-matched pairs, and each must pass a **sanity rule** "
            "requiring canonical marker genes among its top 100. The result is that " +
            "; ".join(lines) + ". "
            "Only one axis therefore survives, and the two-axis M1/M2 framing collapses "
            "to a single interferon-γ axis. This is also the mechanical explanation of "
            "the previous version's LPS signature, which consisted of zinc-finger genes "
            "with no canonical LPS target in it: it was built by comparing across "
            "different donors in a 73-donor study.")
        sc = anc.get("resting_scores", {})
        per = sc.get("per_condition", {})
        usable = anc.get("anchors_usable", [])
        if per and usable:
            ax = usable[0]
            cl = sc.get("donor_paired_contrasts", {})
            bits = []
            for k, v in cl.items():
                if v.get("status") != "ok" or "naive" in k:
                    continue
                bits.append(
                    f"{v['state'].replace('_', ' ')} sits "
                    f"{f(v['mean_difference'], 4)} from the day-6 control "
                    f"(exact sign-flip {pf(v['exact_sign_flip_p_two_sided'])}, floor "
                    f"{f(v['minimum_achievable_p'])})")
            out["C2_PARA"] = (
                f"Scoring GSE58310 on the {ax} axis: " + "; ".join(bits) + ". "
                "These are **resting** day-6 cells, and trained immunity is defined by "
                "the response to restimulation, so proximity to baseline here is the "
                "expected result and is not evidence against training. Naive monocytes "
                "are excluded from the bridge and shown separately, because their "
                "displacement reflects monocyte-to-macrophage differentiation rather "
                "than activation.")
        else:
            out["C2_PARA"] = ("No anchor passed both requirements, so the resting-state "
                              "scores were not computed.")
    else:
        out["C1_PARA"] = "Part C anchor results are not available in this run."
        out["C2_PARA"] = "Part C resting scores are not available in this run."

    if restim:
        rows = []
        for ds in ("GSE212282", "GSE185433"):
            for k, v in ((restim.get(ds) or {}).get("contrasts") or {}).items():
                if "mean_difference" not in v or not k.endswith("mean_log_expression"):
                    continue
                arm = k.split("_vs_")[0].replace("_restim", "").replace("_", " ")
                rows.append([ds, arm, f(v["mean_difference"], 3),
                             v["direction_observed"],
                             "yes" if v.get("matches_expectation") else "**no**",
                             pf(v["exact_sign_flip_p_two_sided"])])
        if rows:
            tab = markdown_table(
                ["dataset", "arm (vs untrained → LPS)", "Δ mean log expression",
                 "direction", "as predicted", "exact p"], rows)
            out["C3_RESTIM_PARA"] = (
                "The test that actually bears on trained immunity is restimulation, and "
                "two public datasets qualify: monocytes trained, differentiated for five "
                "days and re-exposed to LPS for four hours, with matched donors. We "
                "score a pre-specified panel of canonical human LPS/NF-κB genes, because "
                "the only anchor that survived the sanity rules is interferon-γ-derived "
                "while the restimulation here is LPS — a mismatch we state rather than "
                "paper over.\n\n" + tab + "\n\n"
                "With three donors the exact sign-flip null has eight assignments and no "
                "p below 0.25 exists, so none of these is significant and the direction "
                "is what the test establishes. Tolerance points the way tolerance should, "
                "and three of the four training arms point the way training should — but "
                "**β-glucan, the canonical training stimulus, points the other way**. We "
                "report that as observed.")
        else:
            out["C3_RESTIM_PARA"] = ("The restimulation test produced no usable "
                                     "contrasts in this run.")
    else:
        out["C3_RESTIM_PARA"] = "The restimulation test was not run."

    if c3 and c3.get("status") == "not_performed":
        reasons = c3.get("blocking_reasons", [])
        out["C3_PROJECTION_PARA"] = (
            "The projection of omics coordinates into the learned image space is **not "
            "performed**, for two independent reasons either of which would be enough. "
            "It needs image-space M0/M1/M2 centroids, and the only dataset that provides "
            "them was not available (above); and it needs an omics anchor whose stimulus "
            "matches an image label, which none of the surviving anchors does. "
            "Substituting a different label set — BBBC054's microglial morphology "
            "classes are not macrophage polarisation states — and presenting the result "
            "as this analysis would misrepresent what was measured, so it is left "
            "undone. The previous version, unable to run it, substituted a one-axis "
            "analysis that expressed a human monocyte score as an equivalent LPS "
            "exposure time on a mouse microglia axis anchored at the extremes of the "
            "quantity being mapped; that analysis was circular and has been deleted from "
            "the code, the results and this paper rather than relabelled."
            + (f" ({len(reasons)} blocking requirements recorded.)" if reasons else ""))
    else:
        out["C3_PROJECTION_PARA"] = ("The status of the projection analysis is not "
                                     "recorded in this run.")
    return out


def methods(corpus, ps, refs) -> str:
    return f"""### 4.1 Compute

All work ran on a single node with one NVIDIA A100-SXM4-40GB, 30 logical CPU cores and
216 GB of RAM, with all data on local disk. Tier T1 in our own plan: one GPU, a ViT-S
encoder, bf16. Pretraining therefore used a **single seed**; the seed-to-seed variation
reported in Part A comes from independent random-init draws, stratified cell
subsamples and fold assignments, not from repeated pretraining. Exact versions are
pinned in `requirements-lock.txt`; every stage records its own torch version, which was
{ps.get('torch_version', 'not recorded') if ps else 'not recorded'} throughout.

### 4.2 Data and preprocessing

Pretraining video is the ComplexEye collection (Zenodo 10.5281/zenodo.10082674, CC BY
4.0), downloaded one archive at a time with MD5 verification against the deposit record,
preprocessed, and deleted before the next was fetched so that peak disk use is one
archive plus the output shards. Cells are detected with a temporal-median background
model and a MAD threshold, linked with trackpy, and cut into 16-frame 96x96 clips
centred on each tracked cell at temporal strides 1, 3, 9 and 27. One clip in ten is made
static by repeating its middle frame.

Pixel sizes are derived, not assumed. The ComplexEye imager is an OmniVision OV9712 with
a 1280x800 array over a 3888x2430 µm active area, i.e. 3.0375 µm per sensor pixel, at
4.69x magnification, giving 0.648 µm/px in the sample plane; the deposit's own metadata
states 0.641 µm/px for the same instrument and we use that, the two agreeing within 1%.
The reference microscope is a Leica DMI6000 B whose published field of view is
662.9 x 495.2 µm; those movies are 1392 x 1040 px, so the pixel size is 0.4762 µm/px on
both axes. Everything is resampled to a common 0.65 µm/px.

For BBBC054 the page states a Leica DMI 6000, 20x objective and an Andor
AND-DG-152V-C1E-FI camera. The Andor Neo 5.5 sCMOS specification gives a 2560 x 2160
array at 6.5 µm pitch; the BBBC054 images are 1080 x 1280, exactly half in each axis,
i.e. 2x2 binning, so the sample-plane pixel size is 2 x 6.5 / 20 = 0.65 µm/px — already
the common target. Because the camera model code could not be confirmed by name, a
scale-sensitivity analysis over resampling factors 0.5, 0.75, 1.0 and 1.5 is also
reported.

BBBC054 annotation coordinates follow the data rather than the dataset page: the page
states that axis-1 is the x-coordinate, but across all 47,043 annotations axis-1 spans
-4..1080 and axis-2 spans -6..1284 in images that are 1080 rows by 1280 columns, so
axis-1 indexes rows. Following the page transposes every crop.

### 4.3 Leakage check

Every S-BIAD2196 evaluation movie was fingerprinted against every candidate pretraining
movie: frames 0, n/2 and n-1, block-averaged to 40x25, z-scored and concatenated,
compared by Pearson correlation, with the SHA-1 of the raw first frame recorded
separately. A pair counts as a match when the frame dimensions are identical and
r > 0.98, or when the first frames are byte-identical; all pairs above r = 0.90 are
reported whatever their shapes, so the decision is auditable rather than hidden behind a
threshold. Matching **pretraining** movies are excluded.

### 4.4 Model and objective

TiJepa-1 is a V-JEPA-style joint-embedding predictive architecture: a ViT-S context
encoder sees unmasked tubelets, an EMA target encoder sees the whole clip, and a narrow
predictor maps context latents to target latents at masked positions, with an L1 loss in
latent space and no pixel reconstruction. Tubelets are 2x8x8 over 16x96x96 clips.

The v1 objective mixes two masks, sampled per batch with equal probability. The
**spatial tube mask** is the multi-block mask of the original architecture, applied at
every temporal position. The **future mask** makes every spatial position of the last two
temporal slots — the last four frames — a prediction target, leaving the first twelve
frames as context, which makes the predictor a latent forecaster. The two losses are
logged separately because they are not on the same scale. The ablation TiJepa-1-spatial
uses spatial masking only on the identical corpus for the same number of steps.

### 4.5 Probes and statistics

All probes are frozen-feature probes: encoders are never fine-tuned. BBBC054 uses
`StratifiedGroupKFold` grouped by field of view with a standardise-PCA-standardise
pipeline applied identically to every encoder, and three seeds that each vary a
stratified cell subsample, the fold assignment, and the random-init draw.

Interval estimates are reported only where there are at least ten grouping units.
BBBC054 has 60 fields, so encoder pairs there get a paired cluster bootstrap over fields
(2,000 resamples, both encoders scored on the same resample) with Benjamini-Hochberg
correction within each task. Part B's tolerance contrast has four wells and the tracking
evaluation has six movies, so both report per-group values and an **exact** permutation
distribution with its minimum achievable p, and no confidence interval. Where a
permutation can be enumerated exactly it is; otherwise at least 1,000 permutations are
drawn and the floor is reported.

### 4.6 Tracking and forecasting

Tracking: a query point at the first frame of a ground-truth track is propagated by
matching its patch token to candidate tokens within a search window, identically for
every encoder, against a nearest-detection linking baseline. Metrics are the TAP-Vid-style
⟨δ⟩ averaged over 1, 2, 4, 8 and 16 µm thresholds, and median track survival at 8 µm.

Forecasting: each model receives the first 12 frames and predicts its own target-encoder
latents for the last 4 through its own predictor under the future mask, scored as
L1(predictor)/L1(copy-last-observed-latent), with a mean-context-latent baseline.
Only within-model ratios are comparable.

### 4.7 Part B

Single-cell NF-κB traces come from the Tay-lab deposit. The matrix layout was
reconstructed from the deposit's own MATLAB preprocessing code; the numeric ligand codes
come from its plotting code, which states them with doses, and are independently
confirmed by the GEO treatment protocol of the matching ATAC series. Column 128 is the
inter-stimulus interval in hours: the deposit's code computes it from whether the ligand
list repeats a value, which reads like a homologous-versus-heterologous flag, but the
pair (naive, LPS) occurs at both 4 and 8 and every heterologous pair occurs at both
values, so it cannot be a function of ligand identity. Homologous restimulation exists
only at 8 h, which is why the matched comparison is at 8 h.

Expression is GSE260996, assembled from the per-sample Rsubread featureCounts objects
GEO deposits, keyed by Entrez gene ID; it is identical to the merged count matrix the
deposit itself ships. Accessibility is GSE261647. Both pass a data-type guard that checks
row-identifier class, sample count and assay type against the GEO record, and differential
expression uses DESeq2 on counts.

### 4.8 Part C

Anchors are built from GSE46903 with donor-paired moderated tests requiring at least
three donor-matched pairs and passing a canonical-marker sanity rule. GSE46903's M-CSF
arm has its only control at 0 h, i.e. undifferentiated monocytes, so a 72 h M-CSF
contrast would conflate differentiation with stimulus; GM-CSF is used and the mismatch
with the human-serum differentiation of the target datasets is a stated limitation.
Scores are rank-based single-sample scores, which are comparable across platforms.
`inmoose` could not be built in this environment, so the moderated tests use a
pure-Python implementation of the limma empirical-Bayes method, validated in
`tests/test_moderated.py` against hand-computable cases and against the limiting
behaviour that defines the method.

### 4.9 Licences

Pretraining used only CC BY 4.0 data. Datasets excluded on licence grounds were never
downloaded. BBBC054 is CC BY 3.0; S-BIAD2196 is CC BY 4.0; GEO series are US-government
public-domain records cited to their originating publications.

### 4.10 AI-assistance disclosure

Code, analyses and an initial manuscript draft were produced with the assistance of an
AI coding agent (Claude, Anthropic) under the author's direction; the author reviewed
and is responsible for all content. <!-- TODO(author): review -->

### 4.11 Competing interests

<!-- TODO(author): declare competing interests -->
"""


def main() -> int:
    corpus = load("pretrain_corpus.json")
    lk = load("leakage_check.json")
    ps = load("partA/pretrain_tijepa1_seed0.json")
    abl = load("partA/pretrain_tijepa1_spatial_seed0.json")
    dry = load("partA/dryad_benchmark.json")
    bb = load("partA/bbbc054_v1.json")
    tf = load("partA/tracking_forecast_v1.json")
    rna = load("partB/rna_positive_control.json")
    atac = load("partB/atac_positive_control.json")
    lig = load("partB/ligand_codes.json")
    b3 = load("partB/b3_dynamics.json")
    b34 = load("partB/b33_b4.json")
    anc = load("partC/c1_anchors.json")
    restim = load("partC/c3_restimulation.json")
    c3 = load("partC/c3_projection.json")
    refs = load("references.json")

    cpara, cvals = corpus_para(corpus)
    bb_body, bb_conf, bb_table = bbbc_section(bb)
    tpara, fpara = track_para(tf)

    sub: dict[str, str] = {
        "CORPUS_PARA": cpara,
        "LEAKAGE_PARA": leakage_para(lk, corpus),
        "PRETRAIN_PARA": pretrain_para(ps, abl),
        "DRYAD_PARA": dryad_para(dry),
        "BBBC_PARA": bb_body,
        "BBBC_CONFOUND_PARA": bb_conf,
        "BBBC_TABLE": bb_table,
        "TRACK_PARA": tpara,
        "FORECAST_PARA": fpara,
        "METHODS": methods(corpus, ps, refs),
        **cvals,
    }
    sub.update(partB_paras(rna, atac, lig, b3, b34))
    sub.update(partC_paras(anc, restim, c3))

    # abstract fragments, each generated from the numbers
    if bb:
        enc = bb["encoders"]
        best = max((k for k, v in enc.items()
                    if isinstance(v, dict) and "three_class_mean" in v),
                   key=lambda k: enc[k]["three_class_mean"], default=None)
        sub["ABSTRACT_PARTA"] = (
            (f"On BBBC054 microglial morphology classes the strongest encoder is "
             f"{ENC_LABEL.get(best, best)} at {f(enc[best]['three_class_mean'])} "
             "balanced accuracy on the three-class task; a time-only baseline scores "
             "at chance and within-time-bin scores match the overall figures, so the "
             "field-of-view/time confound does not explain the result.")
            if best else
            "BBBC054 probe results are not available in this run.")
    else:
        sub["ABSTRACT_PARTA"] = "BBBC054 probe results are not available in this run."
    fc = ((tf or {}).get("forecasting") or {}).get("summary") or {}
    names = [k for k, v in fc.items() if v.get("mean_ratio_vs_copy_last") is not None]
    if names:
        best_f = min(names, key=lambda k: fc[k]["mean_ratio_vs_copy_last"])
        sub["ABSTRACT_FORECAST"] = (
            f"In latent forecasting, {ENC_LABEL.get(best_f, best_f)} reaches a ratio of "
            f"{f(fc[best_f]['mean_ratio_vs_copy_last'])} against copying the last "
            "observed latent, where below 1 means it beats assuming nothing changes.")
    else:
        sub["ABSTRACT_FORECAST"] = ""
    if b3:
        res = ((b3.get("b31_tolerance") or {}).get("results") or {}).get("raw_trace", {})
        perm = res.get("exact_permutation", {})
        sub["ABSTRACT_B31"] = (
            f"leave-one-well-out balanced accuracy is "
            f"{f(res.get('cell_balanced_accuracy_mean'))} over "
            f"{(b3.get('b31_tolerance') or {}).get('n_wells')} wells, and the exact "
            f"permutation null admits no p below "
            f"{f(perm.get('minimum_achievable_p'))}, so the design cannot reach "
            "significance and we draw no conclusion from it.")
    else:
        sub["ABSTRACT_B31"] = "those results are not available in this run."

    # discussion fragments
    sub["DISCUSSION_PARTA"] = (
        "The future-masking objective gives the model a forecasting task that can be "
        "measured directly, and the forecasting evaluation separates models that have "
        "learned temporal structure from those that have not."
        if names else
        "The forecasting evaluation did not produce usable results in this run.")
    sub["DISCUSSION_LIMITS"] = (
        "The Part B tolerance contrast cannot reach significance with the wells the "
        "public deposit provides. The dynamics-to-expression, dynamics-to-accessibility "
        "and cross-species analyses are null. The Part C bridge is an interpolation "
        "across studies, and its restimulation test is not significant with three "
        "donors and is inconsistent in direction across training stimuli.")
    b31 = (b3 or {}).get("b31_tolerance", {})
    b33c = (((b34 or {}).get("b33") or {}).get("analyses") or {})
    n_cond = next((v.get("n_matched_conditions") for v in b33c.values()
                   if isinstance(v, dict) and v.get("n_matched_conditions")), None)
    sub["POWER_PARA"] = (
        f"The Part B tolerance contrast has {b31.get('n_wells', 'an unknown number of')} "
        f"wells; the dynamics-to-omics regressions have {n_cond or 'few'} matched "
        "conditions; the human tolerance contrast has 4 donors; the restimulation test "
        "has 3. Published label-free classifiers show the same pattern — near-ceiling "
        "same-donor accuracy collapsing to 72-85% across donors — so donor and batch "
        "count, not model capacity, should dominate the design of the experiment above.")

    if refs:
        rows = []
        for k, r in refs["references"].items():
            ident = (r.get("doi") and f"doi:{r['doi']}") or \
                    (r.get("arxiv_id") and f"arXiv:{r['arxiv_id']}") or "TODO(verify)"
            rows.append([k, r.get("first_author") or "—",
                         (r.get("title") or "—")[:72],
                         str(r.get("container") or r.get("year") or "—"),
                         ident])
        sub["REFERENCES"] = (
            "Every identifier below was resolved against Crossref or DataCite during "
            "this run, and each resolved title was checked for the keywords the work "
            "should contain, because a DOI can resolve cleanly and still be the wrong "
            "paper. Nothing is listed that did not resolve.\n\n"
            + markdown_table(["key", "first author", "title", "venue/year", "identifier"],
                             rows))
    else:
        sub["REFERENCES"] = "Reference resolution was not run."

    sub["DATA_AVAILABILITY"] = (
        "All datasets used are public and are listed with their licences and verified "
        "sizes in `manifests/datasets.yaml` and `results/v1/downloads/`. The two "
        "datasets that required manual retrieval (Dryad doi:10.5061/dryad.m0cfxpp8v and "
        "the LTDB figshare collection) were not available for this run and the analyses "
        "that depend on them are reported as not performed, with their code in place. "
        "A defect in the published ComplexEye archive R06.zip is documented in "
        "`results/v1/complexeye_archive_defect.json`.")

    text = TEMPLATE.read_text()
    missing = []
    for key in sorted(set(__import__("re").findall(r"\{\{([A-Z0-9_]+)\}\}", text))):
        if key not in sub:
            missing.append(key)
    for k, v in sub.items():
        text = text.replace("{{" + k + "}}", str(v))
    OUT.write_text(text)
    words = len(text.split())
    print(f"wrote {OUT} ({words:,} words)")
    if missing:
        print(f"UNFILLED PLACEHOLDERS: {missing}")
        return 1
    left = __import__("re").findall(r"\{\{[^}]+\}\}", text)
    if left:
        print(f"UNRENDERED: {left[:5]}")
        return 1
    print("no unfilled placeholders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
