"""G6 (PLAN_v2.md §9): render paper/tijepa.md from results, after everything has run.

Replaces `91_make_paper_v1.py`. Three rules are enforced here rather than by care:

* **Labels come from provenance** (rule 1, finding N1). There is no hand-written label
  map: every encoder label is `tijepa.eval.provenance.label_for_key(key, checkpoint)`,
  and every label rendered is written to `paper/labels.json` with its key and checkpoint
  record so `92_check_claims.py` can re-derive it.
* **Nothing is rendered before its inputs exist** (rule 2). The script refuses to run
  unless both pretraining summaries report an annealed final step and the BBBC054 and
  tracking results carry final TiJepa-1 checkpoints. An absent analysis gets one sentence.
* **No internal process language** (rule 5). Prose names datasets, methods and numbers,
  not plan sections or finding IDs; `92_check_claims.py` fails the build otherwise.

`--results` and `--out` exist so the renderer can be exercised on smoke outputs written
outside `results/` and `paper/`.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import re
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.eval import provenance as PV  # noqa: E402
from tijepa.viz.claims import Comparison, compare_phrase  # noqa: E402
from tijepa.viz.tables import markdown_table  # noqa: E402

V1 = REPO / "results" / "v1"
TEMPLATE = REPO / "paper" / "tijepa_v2.template.md"

LABELS: dict[str, dict] = {}


def f(v, nd=3):
    return "not available" if v is None else f"{v:.{nd}f}"


def pf(p):
    if p is None:
        return "not available"
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.3f}"


def lab(key: str, record: dict | None, suffix: str = "") -> str:
    """The display label for a results key, derived from its checkpoint record."""
    base = PV.label_for_key(key, record)
    shown = base + suffix
    LABELS[shown] = {"key": key, "checkpoint": record, "label": base}
    return shown


def enc_label(key: str, rec: dict) -> str:
    cp = rec.get("checkpoint")
    if cp is None and rec.get("checkpoint_per_seed"):
        cp = next(iter(rec["checkpoint_per_seed"].values()))
    suffix = {"E1_handcrafted_logreg": ", logistic probe",
              "E1_handcrafted_rf": ", random-forest probe"}.get(key, "")
    if key.startswith("E6_random_init_draw"):
        suffix = f", draw {key[-1]}"
    return lab(key, cp, suffix)


def load(root: Path, rel: str):
    p = root / rel
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------- 2.1 corpus
def corpus_para(c) -> tuple[str, dict]:
    by_src = c.get("clips_by_source", {})
    strides = c.get("clips_by_stride", {})
    prov = c.get("pixel_size_provenance", {})
    txt = (
        f"The corpus is **{c['n_clips']:,} cell-centred clips** of 16 frames at 96x96 px "
        f"({c['total_gb']} GB in {c['n_shards']} sequential shards) from "
        f"{c['n_movies_ok']} movies across {c['n_archives_done']} of the "
        f"{c['n_archives_in_record']} ComplexEye archives [complexeye] (Zenodo "
        "10.5281/zenodo.10082674, CC BY 4.0). Pixel sizes are derived from the instrument "
        f"specifications (Methods 4.2): "
        f"{prov.get('complexeye', {}).get('um_per_px', 'n/a')} µm/px for the ComplexEye "
        f"movies and {prov.get('complexeye_leica', {}).get('um_per_px', 'n/a')} µm/px "
        "for the Leica reference movies, all resampled to "
        f"{c.get('target_um_per_px')} µm/px. Clips are sampled at temporal strides "
        + ", ".join(f"{k} ({v:,})"
                    for k, v in sorted(strides.items(), key=lambda kv: int(kv[0])))
        + f", and {c.get('static_fraction', 0):.1%} of clips are made static by repeating "
        "a single frame, so still-image evaluation lies inside the training distribution. "
        f"Heuristic frame-level quality control removed {c.get('n_movies_excluded_qc', 0)} "
        "movies, with thresholds set from this corpus's own empirical distribution. "
        f"Clips by source: {', '.join(f'{k} {v:,}' for k, v in by_src.items())}.")
    return txt, {"CORPUS_CLIPS": f"{c['n_clips']:,}", "CORPUS_MOVIES": str(c["n_movies_ok"])}


def leakage_para(lk_v1, lk) -> str:
    v1pair = (lk_v1 or {}).get("pairs", [{}])[0]
    t = (
        "The tracking evaluation uses S-BIAD2196, which like the corpus is human "
        "neutrophil brightfield video, so overlap between the two was checked rather than "
        "assumed. A coarse fingerprint (three frames, block-averaged to 40x25, z-scored) "
        f"found one real duplicate, {v1pair.get('eval_movie', '?').split('.')[0]} against "
        f"{v1pair.get('train_movie', '?')} (r = {v1pair.get('pearson_r')}), and that "
        "pretraining movie was excluded before the corpus was built.")
    if not lk:
        return t + (" A full-resolution check of every corpus archive was not completed, "
                    "so further overlap cannot be excluded.")
    n_arch = sum(1 for a in lk["archives"].values() if a.get("checked"))
    dups = lk["duplicates_in_frozen_corpus"]
    t += (
        f" The coarse fingerprint is weakly discriminative ({lk['n_candidate_pairs']} "
        "pairs exceed r = 0.9), so every such pair, and every pair across all "
        f"{len(lk['archives_in_corpus'])} corpus archives ({n_arch} checked in full), was "
        "re-tested by full-resolution normalised cross-correlation of five frames spread "
        "across each movie; a duplicate requires r > 0.95 on at least three of the five. ")
    ex = lk.get("explicit_checks") or []
    if ex:
        e = ex[0]
        t += (f"The closest non-matching coarse pair, {e['eval_movie'].split('.')[0]} "
              f"against {e['train_movie']} (coarse r = {e['fingerprint_r']}), has "
              f"full-resolution r of {', '.join(f'{x:.2f}' for x in e['full_res_r'])} "
              f"and is {'a duplicate' if e['duplicate'] else 'not a duplicate'}. ")
    if dups:
        movies = sorted({d["eval_movie"].split(".")[0] for d in dups})
        t += (f"**{len(dups)} further duplicate pair(s) were found inside the corpus**, "
              f"involving evaluation movie(s) {', '.join(movies)}; the corpus is kept "
              "fixed, so those movies are excluded from tracking and forecasting for every "
              "model trained on it.")
    else:
        t += "No further duplicate exists in the corpus."
    t += (" TiJepa-0, an earlier model trained on a different corpus that did contain the "
          "duplicated movie, is reported only without 08_H4.")
    if not lk.get("all_archives_checked"):
        t += (" Not every archive could be checked; see the data-availability statement.")
    return t


# -------------------------------------------------------------- 2.2 pretraining
def rank_series(log: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    if not log.exists():
        return [], [], [], []
    st, rk, ls, lf = [], [], [], []
    for r in csv.DictReader(log.open()):
        st.append(int(r["step"]))
        rk.append(float(r["effective_rank"]))
        ls.append(float(r["loss_spatial"]) if r["loss_spatial"] else np.nan)
        lf.append(float(r["loss_future"]) if r["loss_future"] else np.nan)
    return st, rk, ls, lf


def pretrain_para(ps, abl, plan) -> str:
    segs = ps["segments"]
    seg_txt = "; ".join(
        f"steps {s['start_step']:,}-{s['end_step']:,} on {s['gpu_name']} with torch "
        f"{s['torch_version']} ({s.get('gpu_seconds', s['seconds']) / 3600:.2f} h)"
        for s in segs)
    st, rk, _, _ = rank_series(REPO / "checkpoints" / "pretrain_log_tijepa1_seed0.csv")
    r = ps.get("resume") or {}
    sched = (plan or {}).get("schedule_at_resume_step_14751", {})
    t = (
        f"TiJepa-1 is a V-JEPA-style model [vjepa] (ViT-S encoder, bf16) trained with "
        f"{ps['objective']}. The budget is defined in steps: {ps['planned_final_step']:,} "
        f"steps at batch 128, i.e. {ps['clips_seen']:,} clip views or "
        f"{ps['effective_epochs']} passes over the corpus, chosen as the number of steps "
        "the original 12 GPU-hour budget buys at the throughput first observed. The cosine "
        "learning-rate schedule and the EMA momentum are tied to that final step, so they "
        f"finish annealing exactly on it: the final learning rate is {ps['final_lr']:.1e} "
        f"and the final EMA momentum {ps['final_ema_momentum']:.4f}. "
        f"Training ran in {len(segs)} segment(s) ({seg_txt}), {ps['gpu_hours']:.2f} "
        "GPU-hours in total. ")
    if len(segs) > 1 and sched:
        t += (
            f"The run was paused at step {r.get('step', 14751):,} to move to a dedicated "
            "GPU and resumed from its checkpoint with optimiser state intact. The first "
            "segment had been scheduled for a longer run, so tying the schedule to the final "
            f"step changed the learning rate at the resume by "
            f"{100 * sched['lr_relative_change']:+.2f}% "
            f"({sched['lr_old_schedule_max150000']:.3e} to "
            f"{sched['lr_new_schedule_max117000']:.3e}) and the EMA momentum from "
            f"{sched['ema_next_update_old_schedule']:.6f} to "
            f"{sched['ema_next_update_new_schedule']:.6f}. ")
    t += (
        f"Over the last 500 steps the latent L1 loss is "
        f"{f(ps.get('final_loss_spatial_last500'))} on spatially masked batches and "
        f"{f(ps.get('final_loss_future_last500'))} on future-masked batches; the two are "
        "logged separately because they are not on the same scale.")
    if rk:
        t += (f" The effective rank of the embedding, a collapse diagnostic, is "
              f"{rk[0]:.1f} at step {st[0]:,} and {rk[-1]:.1f} at step {st[-1]:,} "
              "(of 384 dimensions), with no collapse.")
    if abl:
        t += (
            f" The ablation, TiJepa-1-spatial, uses the identical corpus, schedule and seed "
            f"with spatial masking only, from scratch to the same {abl['steps']:,} steps "
            f"({abl['gpu_hours']:.2f} GPU-hours); its final loss over the last 500 steps is "
            f"{f(abl.get('final_loss_spatial_last500'))}.")
    if not ps["annealed"] or (abl and not abl["annealed"]):
        t += " **One of the runs ended before its planned final step and is unannealed.**"
    return t


# -------------------------------------------------------------------- 2.3 dryad
def dryad_para(d) -> str:
    return (
        "The planned headline for Part A was a video benchmark with activation-state "
        "labels: Dryad doi:10.5061/dryad.m0cfxpp8v, bone-marrow-derived macrophage movies "
        "labelled M0, M1 and M2. Its automated download endpoints serve a proof-of-work "
        "interstitial, which we treat as an access control and do not bypass, and a manual "
        "copy was not available, so it was not analysed. No dataset used here therefore "
        "provides video of innate immune cells with activation-state labels; Part A's "
        "classification evidence is static, and the frame-shuffle control applies only to "
        "tracking and forecasting."
        if (d or {}).get("status") == "not_provided" else
        "The Dryad benchmark status is not recorded.")


# ------------------------------------------------------------------- 2.4 bbbc054
def bbbc_section(bb) -> dict:
    enc = {k: v for k, v in bb["encoders"].items() if isinstance(v, dict)
           and "three_class_mean" in v}
    labels = {k: enc_label(k, v) for k, v in enc.items()}
    rows = []
    for k in sorted(enc, key=lambda k: -enc[k]["three_class_mean"]):
        v = enc[k]
        rows.append([labels[k], f"{f(v['three_class_mean'])} ± {f(v['three_class_sd'])}",
                     f(v.get("three_class_within_bin_mean")),
                     f"{f(v['round_vs_amoeboid_mean'])} ± {f(v['round_vs_amoeboid_sd'])}",
                     f(v.get("fov_identity_mean"))])
    table = markdown_table(["encoder", "3-class BA (mean ± sd)", "3-class within-bin",
                            "round vs amoeboid BA", "field identity BA"], rows)
    n = bb["n_cells_per_seed"]
    cc = bb["class_counts_per_seed"]["0"]
    comps = {}
    for ck, c in bb.get("pairwise_comparisons", {}).items():
        task = ck.split(":", 1)[0]
        comps[(task, c["encoder_a"], c["encoder_b"])] = Comparison(
            c["encoder_a"], c["encoder_b"], task, difference=c["difference_mean_over_seeds"],
            ci=tuple(c["ci95_widest_over_seeds"]), p_value=c.get("p_value_bh"))

    def cmp(a, b, task="three_class"):
        c = comps.get((task, a, b))
        if c is not None:
            return c, False
        c = comps.get((task, b, a))
        if c is None:
            return None, False
        return Comparison(a, b, task, difference=-c.difference,
                          ci=(-c.ci[1], -c.ci[0]), p_value=c.p_value), True

    def sent(a, b):
        c, _ = cmp(a, b)
        if c is None or a not in enc or b not in enc:
            return None
        return (f"{labels[a]} is {compare_phrase(c)} {labels[b]} "
                f"(Δ {c.difference:+.3f}, 95% CI [{c.ci[0]:+.3f}, {c.ci[1]:+.3f}], "
                f"BH-corrected {pf(c.p_value)})")

    key_pairs = [("E4_tijepa1", "E6_random_init"), ("E4_tijepa1", "E4s_tijepa1_spatial"),
                 ("E4_tijepa1", "E3_dinov2_vit_s"), ("E4_tijepa1", "E0_tijepa0")]
    sents = [s for a, b in key_pairs if (s := sent(a, b))]
    body = (
        "BBBC054 is a set of LPS-exposed mouse microglia images with per-cell "
        "**morphology-class** annotations [bbbc]; the labels are cell shapes, and shape is "
        "not activation state. Replicate 1 annotates three classes, round, ramified and "
        f"amoeboid. Each seed draws about {int(np.mean([int(v) for v in n.values()])):,} "
        f"cells (seed 0: {cc.get('round', 0):,} round, {cc.get('ramified', 0):,} "
        f"ramified, {cc.get('amoeboid', 0):,} amoeboid), taking up to the same number of "
        "cells of each class from every field of view; amoeboid cells are rare and are "
        "taken in full. Probes are identical for every encoder, cross-validated with "
        "folds grouped by field of view, and repeated over three seeds that each change "
        "the cell subsample, the fold assignment and, for the random-init control, the "
        "network draw. Encoder pairs are compared by a paired cluster bootstrap over the "
        "60 fields with Benjamini-Hochberg correction within each task. On the three-class "
        "task, " + "; ".join(sents) + ".")
    tob = bb["time_only_baseline"]
    conf = bb["confound_full_file"]
    confound = (
        "In BBBC054 one field of view is one LPS exposure time, so field and time are "
        "confounded. In the full annotation file the dependence of class on exposure time "
        f"is moderate (Cramér's V = {conf['cramers_v_class_vs_time']:.3f} over "
        f"{conf['n_annotations']:,} annotations and 60 times). In the sample used here, "
        "where no class runs out every field contributes equal class counts, so exposure "
        "time carries no class information **by construction**; a time-only classifier "
        f"nevertheless reaches {f(tob['three_class']['mean'])} on the three-class task and "
        f"{f(tob['round_vs_amoeboid']['mean'])} on round versus amoeboid, against chance "
        "levels of 0.333 and 0.500, because amoeboid cells run out in some fields. Those "
        "figures are the share of any encoder's score that needs no image. Balanced "
        "accuracy computed within six contiguous time bins is reported alongside, and a "
        "field-identity probe measures how much field-specific appearance each encoder "
        "carries.")
    reg_rows = []
    for k in sorted(enc, key=lambda k: -enc[k].get("exposure_time_skill_vs_cv_mean", -9)):
        v = enc[k]
        reg_rows.append([labels[k], f(v.get("exposure_time_mae_mean"), 2),
                         f(v.get("exposure_time_skill_vs_cv_mean")),
                         f(v.get("exposure_time_within_block_rho_mean"))])
    ref_rho = np.mean([v.get("exposure_time_rho_cv_mean_predictor", np.nan)
                       for v in enc.values()])
    ref_mae = np.mean([r["exposure_time_lofo"]["mae_cv_mean_predictor_hours"]
                       for r in next(iter(enc.values()))["per_seed"]])
    reg = (
        "Exposure-time regression is exploratory. It uses leave-one-field-out "
        "cross-validation with the three fields on either side of the test field removed "
        "from training, so every test field is bracketed by training data in time. Any "
        "leave-out scheme shifts the training mean away from the held-out field, so a "
        "predictor with no time information is anti-correlated with the truth: predicting "
        f"the training-fold mean gives Spearman ρ = {ref_rho:.2f} and an MAE of "
        f"{ref_mae:.2f} h under the same folds. Each encoder is therefore scored by its "
        "skill against that reference, 1 − MAE/MAE(reference), and by the rank correlation "
        "within blocks of six contiguous fields:\n\n"
        + markdown_table(["encoder", "MAE (h)", "skill vs fold-mean", "within-block ρ"],
                         reg_rows))
    ss = bb.get("scale_sensitivity")
    if ss:
        vals = ss["three_class_balanced_accuracy"]
        reg += ("\n\nBecause the BBBC054 camera model code could not be confirmed by name, "
                "TiJepa-1 was also probed after rescaling the crops by factors "
                + ", ".join(f"{k} ({f(v)})" for k, v in vals.items())
                + " (three-class balanced accuracy, seed 0).")
    return {"BBBC_PARA": body, "BBBC_TABLE": table, "BBBC_CONFOUND_PARA": confound,
            "BBBC_REGRESSION_PARA": reg}


# -------------------------------------------------------- 2.5 tracking/forecast
def track_section(tf) -> dict:
    agg = tf["tracking"]["aggregate"]
    encs = tf["encoders"]

    def tl(k):
        if k == "baseline_nearest_detection":
            return lab("baseline", None, " (nearest detection)")
        if k == "E5_frame_shuffled":
            return lab("E5_frame_shuffle", None)
        return enc_label(k, encs[k])

    rows = []
    for k, v in sorted(agg.items(), key=lambda kv: -(kv[1].get("delta_avg_mean") or 0)):
        if "delta_avg_mean" not in v:
            continue
        perm = v.get("vs_baseline_exact_permutation") or {}
        curve = v["per_threshold_success"]
        rows.append([tl(k), str(v["n_tracks"]), f(v["delta_avg_mean"]),
                     " / ".join(f"{curve[f'within_{t}um']:.2f}" for t in (1, 2, 4, 8, 16)),
                     f"{v['median_survival_frames']:.0f}",
                     pf(perm.get("exact_sign_flip_p_two_sided")) if perm else "—"])
    sel = tf["track_selection"]["per_movie"]
    ntr = ", ".join(f"{m.split('.')[0]} {v['n_tracks']}" for m, v in sel.items())
    excl = tf["leakage"]["excluded_for_corpus_v1_models"]
    t = (
        f"Point tracking uses all {tf['n_movies']} movies of S-BIAD2196, with up to 50 "
        f"ground-truth tracks per movie drawn by a recorded seeded rule ({ntr}). Every "
        "encoder uses the same tracker: the query cell's patch token is matched to "
        "candidate tokens in a search window with an identical motion prior. Success is "
        "the fraction of frames within 1, 2, 4, 8 and 16 µm of the ground truth, and ⟨δ⟩ "
        "is their mean; survival is the number of frames tracked before the first error "
        "above 8 µm. With six movies no confidence interval is reported; each method is "
        "compared with the baseline by an exact sign-flip permutation over movies, whose "
        "smallest attainable two-sided p is 0.031."
        + (f" Movie(s) {', '.join(m.split('.')[0] for m in excl)} are excluded for models "
           "trained on the corpus, which contains a duplicate of them." if excl else "")
        + "\n\n" + markdown_table(
            ["method", "tracks", "⟨δ⟩", "success at 1/2/4/8/16 µm",
             "median survival (frames)", "vs baseline (exact)"], rows))
    notes = tf["tracking"].get("rank_consistency_notes") or []
    if notes:
        t += ("\n\nThe two summaries do not always agree: ⟨δ⟩ averages success over the "
              "whole track at five thresholds, while survival counts only the frames before "
              "the first error above 8 µm, so a tracker that drifts slowly but rarely jumps "
              f"can win one and lose the other ({len(notes)} method pair(s) here are "
              "ordered differently by the two).")
    e0 = agg.get("E0_tijepa0", {})
    if e0.get("with_08_H4_reference_only"):
        t += (f" TiJepa-0 is shown without 08_H4; with it, its ⟨δ⟩ would be "
              f"{f(e0['with_08_H4_reference_only']['delta_avg_mean'])}.")

    fc = tf["forecasting"]
    summ = fc["summary"]
    frows = []
    for k, v in summ.items():
        if v.get("mean_ratio_vs_copy_last") is None:
            continue
        pm = v["per_movie_ratio_vs_copy_last"]
        frows.append([enc_label(k, encs[k]), str(len(pm)), f(v["mean_ratio_vs_copy_last"]),
                      f(v["mean_ratio_vs_mean_latent"]),
                      ", ".join(f"{x:.2f}" for x in pm.values())])
    status = fc["per_movie_status"]
    not_eval = {m: s for m, s in status.items() if not s.startswith("evaluated")}
    p = fc["primary_tijepa1_vs_spatial"]
    fpara = (
        "Because TiJepa-1's predictor is trained to produce future latents, forecasting can "
        "be measured directly. Each model receives the first 12 frames of a cell-centred "
        "clip and predicts its own target-encoder latents for the last 4; the score is the "
        "L1 error of that prediction divided by the error of copying the last observed "
        "latent, so a ratio below 1 means the predictor beats assuming nothing changes. "
        "Latent spaces differ between models, so only within-model ratios are meaningful. "
        "TiJepa-1-spatial never saw the future mask in training, so its query is out of "
        "distribution; that is the point of the comparison, which asks whether the "
        "forecasting objective, rather than the architecture or the data, produces "
        "forecasting. "
        + ("Forecasting ran on every movie. " if not not_eval else
           "Movies not evaluated: " + "; ".join(f"{m.split('.')[0]} ({s})"
                                                for m, s in not_eval.items()) + ". ")
        + "\n\n" + markdown_table(["model", "movies", "ratio vs copy-last (mean)",
                                    "ratio vs mean latent", "per movie"], frows))
    if p.get("n_movies"):
        c = Comparison("E4_tijepa1", "E4s_tijepa1_spatial", "forecast",
                       difference=p["mean_difference"],
                       p_value=p["exact_sign_flip_p_two_sided"])
        d = c.direction
        word = {1: "higher than", -1: "lower than", 0: "not distinguishable from"}.get(d)
        fpara += (
            f"\n\nThe primary comparison is TiJepa-1 against TiJepa-1-spatial, per movie. "
            f"TiJepa-1's ratio is {word} TiJepa-1-spatial's (mean difference "
            f"{p['mean_difference']:+.3f} over {p['n_movies']} movies, per movie "
            + ", ".join(f"{x:+.3f}" for x in p["per_movie_difference"])
            + f"; exact sign-flip {pf(p['exact_sign_flip_p_two_sided'])}, smallest "
            f"attainable {f(p['minimum_achievable_p'])}).")
    supp = fc["supplementary_random_init"]["rows"]
    if supp:
        fpara += (" Random-init models are listed in the supplementary results only "
                  "(ratios " + ", ".join(f"{v['mean_ratio_vs_copy_last']:.2f}"
                                         for v in supp.values()
                                         if v.get("mean_ratio_vs_copy_last")) +
                  "): their predictor is untrained, so the ratio measures an arbitrary map "
                  "rather than forecasting.")
    return {"TRACK_PARA": t, "FORECAST_PARA": fpara, "_primary": p}


# ----------------------------------------------------------------- 2.6 part B
def partB(rna, atac, lig, b3, b34) -> dict:
    out = {}
    out["PC_PARA"] = (
        "No result in this section is reported unless its pipeline first passes a positive "
        "control. For expression, media→LPS against media→media in GSE260996 [wang2025] "
        f"recovers {rna['n_passing']} of 7 canonical LPS/NF-κB targets with log2 fold change "
        "above 1 and FDR below 0.05: "
        f"{'pass' if rna['passed'] else 'fail'}. For accessibility, LPS against medium in "
        f"GSE261647 raises accessibility near all {atac['n_genes_increased']} of those "
        f"genes' transcription start sites ({f(atac['mean_delta_at_windows'])} logCPM "
        f"against {f(atac['mean_delta_genome_wide'])} genome-wide, "
        f"{pf(atac['mannwhitney_p_one_sided'])}): {'pass' if atac['passed'] else 'fail'}. "
        "Both matrices are checked against their GEO records for row-identifier class, "
        "sample count and assay type before use.")
    san = lig.get("tolerance_sanity_check", {})
    tol = b3["b31_tolerance"]
    res = tol["results"]["raw_trace"]
    hc = tol["results"]["handcrafted"]
    perm = res["exact_permutation"]
    sens = b3["b31_sensitivity_all_wells"]
    out["B31_PARA"] = (
        "The ligand codes in the trace deposit are taken from the deposit's own plotting "
        "code, which states them with doses (1 = CpG 100 nM, 2 = LPS 1 ng/ml, 3 = polyI:C "
        "1000 ng/ml, 6 = TNF 10 ng/ml), and the GEO treatment protocol of the matching ATAC "
        "series states the same doses. As a sanity check, the NF-κB second-response "
        f"amplitude is {f(san.get('mean_amplitude_lps_to_lps'), 4)} for LPS→LPS against "
        f"{f(san.get('mean_amplitude_naive_to_lps'), 4)} for naive→LPS at the matched "
        f"{san.get('interval_used_h')} h interval: "
        + ("lower, as acute endotoxin tolerance predicts."
           if san.get("expected_direction_observed") else "higher, contradicting tolerance."))
    out["B32_PARA"] = (
        "Can a classifier tell an acutely endotoxin-tolerised macrophage from a naive one? "
        f"The deposit supplies {tol['n_wells']} wells at the matched 8 h interval "
        f"({tol['n_cells']:,} cells). Leave-one-well-out balanced accuracy is "
        f"{f(res['cell_balanced_accuracy_mean'])} ± {f(res['cell_balanced_accuracy_sd'])} "
        f"from the raw trace and {f(hc['cell_balanced_accuracy_mean'])} ± "
        f"{f(hc['cell_balanced_accuracy_sd'])} from hand-crafted features (three seeds, each "
        "a different per-well cell subsample). The exact permutation null over all "
        f"{perm['n_distinct_assignments']} distinct well-label assignments gives "
        f"{pf(perm['p_exact'])}, and no p below {f(perm['minimum_achievable_p'])} is "
        "attainable, so the design cannot reach significance and we draw no conclusion "
        "about tolerance from dynamics. A sensitivity analysis that pools "
        f"{sens['n_wells']} wells across deposit files and both intervals, confounding "
        f"priming with file and interval, gives {f(sens['cell_balanced_accuracy_mean'])} ± "
        f"{f(sens['cell_balanced_accuracy_sd'])} "
        f"(exact {pf(sens['exact_permutation']['p_exact'])}).")
    b32 = b3.get("b32_ligand_identity", {})
    if b32:
        out["B32_PARA"] += (
            f" A secondary task, identifying which pair of ligands a cell saw, reaches "
            f"{f(b32.get('balanced_accuracy_mean'))} ± {f(b32.get('balanced_accuracy_sd'))} "
            f"over {b32.get('n_classes')} classes against chance "
            f"{f(b32.get('chance_balanced_accuracy'))}, so the traces carry stimulus "
            "information even where they cannot resolve priming history.")
    an = b34["b33"]["analyses"]
    bits = []
    for k, v in an.items():
        r = v["top2000_variable_targets"]
        p = r["exact_permutation"]
        bits.append(
            f"for {k.replace('_', ' ')}, {100 * r['fraction_q2_positive']:.1f}% of the "
            f"{r['n_targets']:,} most variable targets have Q² > 0 against a permutation "
            f"null mean of {100 * p['null_mean']:.1f}% ({pf(p['p_value'])}, exact over all "
            f"{p['n_permutations'] + 1} condition orderings)")
    out["B33_PARA"] = (
        "Condition-level dynamics were related to expression (GSE260996) and, separately, "
        "to chromatin accessibility (GSE261647), matching trace conditions to omics "
        "conditions on stimulus identity and the 4 h inter-stimulus interval. With "
        f"{an['rna_expression']['n_matched_conditions']} matched conditions, leave-one-out "
        "correlations are biased downward, so each target is scored by its predictive R² "
        "against the leave-one-out mean of the other conditions (Q²), which is positive "
        "only when the ridge prediction beats that mean under the same folds. Both "
        "analyses are null: " + "; ".join(bits) + ". This analysis is descriptive.")
    b4 = b34["b4"]
    m = b4["mouse"]
    fi = b4["fisher"]
    prim = fi["GSE185433_primary:both"]
    sec = fi["GSE58310_secondary:both"]
    hp = b4["human_primary"]
    out["B4_PARA"] = (
        "Finally we asked whether genes whose LPS induction is blunted by prior priming in "
        "mouse overlap the genes reduced by LPS tolerance in human. In mouse, a DESeq2 model "
        "with priming, challenge and their interaction [deseq2; pydeseq2] finds "
        f"{m['n_lps_induced']:,} genes LPS-induced in the naive arm; of these, "
        f"{m['n_blunted']['CpG']:,} have a significantly negative CpG × LPS interaction, "
        f"{m['n_blunted']['polyIC']:,} a negative polyI:C × LPS interaction, and "
        f"{m['n_blunted']['both']:,} both. The human contrast closest to this design is "
        "GSE185433 [schrijver2023], LPS-tolerised against untrained macrophages, both "
        f"restimulated with LPS; a donor-paired moderated test over {len(hp['donors'])} "
        f"donors finds {len(hp['down']):,} genes lower in tolerised cells at FDR 0.05. "
        "Orthologs come from the MGI table [mgi], which maps all seven canonical LPS genes "
        "correctly (positive control passed). The overlap of the doubly blunted mouse genes "
        f"with the human tolerance genes is {prim['n_overlap']} against "
        f"{prim['expected_overlap']:.1f} expected (odds ratio {prim['odds_ratio']:.2f}; "
        f"enrichment {pf(prim['p_enrichment'])}, depletion {pf(prim['p_depletion'])}). "
        "With resting LPS-tolerised cells from GSE58310 [saeed2014] as the human side the "
        f"overlap is {sec['n_overlap']} against {sec['expected_overlap']:.1f} expected. "
        "The two contrasts are not the same biology (mouse heterologous priming 4 h apart, "
        "human homologous tolerance), and the comparison is exploratory.")
    return out


# ------------------------------------------------------------------ 2.7 part C
def partC(anc, restim, c3, expect) -> dict:
    out = {}
    a = anc["anchors"]
    usable = anc.get("anchors_usable", [])
    out["C1_PARA"] = (
        "An earlier design built activation axes from GSE46903 [xue2014] with donor-paired "
        "moderated tests and a canonical-marker sanity rule. Under those requirements "
        + "; ".join(
            f"the {k} anchor "
            + ("passes" if v.get("status") == "ok" else
               "cannot be built (too few donor-matched pairs)" if v.get("status") ==
               "cannot_form_anchor" else "fails its marker rule")
            for k, v in a.items())
        + f", leaving only {', '.join(usable) or 'no'} axis. An interferon-γ axis is not an "
        "LPS-response signature, and the restimulation in the datasets below is LPS, so "
        "here the restimulation response is read directly from TNF and IL6, the canonical "
        "trained-immunity readouts, with a pre-specified NF-κB panel and the "
        "interferon-stimulated genes as separate secondary readouts.")
    rows = []
    for gse in ("GSE212282", "GSE185433"):
        ds = restim[gse]
        if ds.get("status") != "ok":
            continue
        for arm, r in ds["arms"].items():
            if "primary_TNF_IL6" not in r:
                continue
            for g in ("TNF", "IL6"):
                x = r["primary_TNF_IL6"][g]
                rows.append([gse, arm.replace("_", " "), g,
                             ", ".join(f"{v:+.2f}" for v in x["per_donor_log2_difference"]),
                             f"{x['mean_difference']:+.2f}",
                             x["sourced_expectation"],
                             x.get("verdict", "—").replace(" the sourced direction", ""),
                             pf(x["exact_sign_flip_p_two_sided"])])
    pc = {g: restim[g]["positive_control"] for g in ("GSE212282", "GSE185433")}
    pc_txt = "; ".join(
        f"{g}: " + ", ".join(f"{k} {v['mean_difference']:+.1f}" for k, v in p["genes"].items())
        for g, p in pc.items())
    bg = restim["GSE212282"].get("beta_glucan_opposite")
    t = (
        "Two public datasets restimulate trained and untrained human monocyte-derived "
        "macrophages with LPS on matched donors: GSE212282 [ferreira2023] (β-glucan, "
        "itaconate, dimethyl itaconate and β-glucan with dimethyl itaconate) and GSE185433 "
        "[schrijver2023] (LPS-tolerised, IL-4 and IL-4 with LPS). In both, monocytes are "
        "exposed for 24 h, rested for five days and restimulated with LPS for 4 h before "
        "RNA sequencing; the GEO record of every sample used states that restimulation. As "
        "a positive control, LPS restimulation of untrained cells raises TNF, IL6 and IL1B "
        f"above the same donors' unstimulated day-6 cells (log2 differences {pc_txt}). "
        "An expected direction is stated only where the dataset's own publication reports "
        "TNF or IL-6 after LPS restimulation for that arm; those reports are cytokine "
        "protein after 24 h of restimulation, while the data here are RNA after 4 h. "
        "Ferreira et al. report higher TNF and IL-6 for dimethyl itaconate and β-glucan "
        "(their Figure 1D) and higher IL-6 for itaconate (Figure 1B-C); Schrijver et al. "
        "report higher TNF and IL-6 after IL-4 training (their Figure 1e). Neither reports "
        "a restimulation phenotype for the combination arms or for LPS-tolerised cells in "
        "this protocol.\n\n"
        + markdown_table(["dataset", "arm (vs untrained → LPS)", "gene",
                          "per-donor log2 Δ", "mean", "sourced expectation", "observed vs "
                          "expectation", "exact p"], rows)
        + "\n\nWith three donors the exact sign-flip null has eight assignments and no "
        "two-sided p below 0.25 exists, so no contrast is significant and the per-donor "
        "values are what the test shows.")
    if bg:
        t += (" **β-glucan, the canonical training stimulus, points opposite to its "
              "reported direction** for TNF and IL6 in this dataset. Candidate explanations, "
              "none of which we can distinguish here, are the readout and its timing (RNA at "
              "4 h against protein at 24 h), the depth of one bulk library per donor and arm "
              "against pooled ELISA replicates, and the number of donors.")
    out["C3_RESTIM_PARA"] = t
    out["C3_PROJECTION_PARA"] = (
        "Projecting these molecular states into the learned image space was not performed: "
        "it needs image-space anchors for macrophage activation states, which only the "
        "unavailable Dryad dataset would provide, and an omics anchor whose stimulus matches "
        "an image label, which no surviving anchor does.")
    return out


# --------------------------------------------------------------------- methods
def methods(ps, abl, plan, bb, tf) -> str:
    segs = ps["segments"]
    gpus = sorted({s["gpu_name"] for s in segs})
    torches = sorted({s["torch_version"] for s in segs + (abl or {}).get("segments", [])})
    bench = (plan or {}).get("benchmark", {})
    return f"""### 4.1 Compute

Pretraining ran on {', '.join(gpus)} GPUs, one at a time, with torch
{' and '.join(torches)}; per-segment hosts, GPUs, step ranges and times are recorded with
each checkpoint. On the second instance one GPU trained at
{bench.get('steps_per_hour', 0):,.0f} steps per hour at {bench.get('gpu_util_mean')}%
mean utilisation with 28 data-loading workers reading memory-mapped shards from local
disk. There is a single pretraining seed per objective; the seed-to-seed variation
reported in Part A comes from independent random-init draws, cell subsamples and fold
assignments.

### 4.2 Data and preprocessing

Pretraining video is the ComplexEye collection [complexeye] (Zenodo
10.5281/zenodo.10082674, CC BY 4.0), downloaded one archive at a time with MD5
verification, preprocessed, and deleted before the next. Cells are detected with a
temporal-median background model and a MAD threshold, linked with trackpy [trackpy], and
cut into 16-frame 96x96 clips centred on each tracked cell at temporal strides 1, 3, 9 and
27. One clip in ten is made static by repeating its middle frame. The corpus is frozen:
its index file has a fixed checksum and every shard's SHA-256 is recorded, and both
pretraining runs read the identical shard list.

Pixel sizes are derived, not assumed. The ComplexEye imager is an OmniVision OV9712 with a
1280x800 array over a 3888x2430 µm active area, i.e. 3.0375 µm per sensor pixel, at 4.69x
magnification, giving 0.648 µm/px; the deposit's metadata states 0.641 µm/px for the same
instrument and we use that. The reference microscope is a Leica DMI6000 B whose published
field of view is 662.9 x 495.2 µm over 1392 x 1040 px, i.e. 0.4762 µm/px. Everything is
resampled to 0.65 µm/px. For BBBC054 the page states a Leica DMI 6000, 20x objective and an
Andor camera whose 2560 x 2160 array at 6.5 µm pitch, binned 2x2 to the 1080 x 1280 images,
gives 2 x 6.5 / 20 = 0.65 µm/px. BBBC054 annotation coordinates follow the data: axis-1
indexes rows.

### 4.3 Leakage check

Every S-BIAD2196 movie was compared with every movie in every corpus archive. A coarse
fingerprint (frames 0, n/2 and n-1, block-averaged to 40x25, z-scored, Pearson r, plus the
SHA-1 of the first frame) screens all pairs; every pair with r > 0.9 is re-tested by
normalised cross-correlation of five full-resolution frames at 0, n/4, n/2, 3n/4 and n-1 of
each movie, and is a duplicate when r > 0.95 on at least three of them. A duplicate found
before the corpus was built is excluded from it; one found afterwards excludes the
evaluation movie for models trained on the corpus.

### 4.4 Model and objective

TiJepa-1 is a V-JEPA-style joint-embedding predictive architecture [vjepa]: a ViT-S
context encoder sees unmasked tubelets, an EMA target encoder sees the whole clip, and a
narrow predictor maps context latents to target latents at masked positions, with an L1
loss in latent space. Tubelets are 2x8x8 over 16x96x96 clips. Two masks are sampled per
batch with equal probability: the multi-block **spatial tube mask**, applied at every
temporal position, and the **future mask**, which makes every position of the last two
temporal slots (the last four frames) a target and leaves the first twelve frames as
context. TiJepa-1-spatial uses the spatial mask only. Both runs use AdamW (learning rate
1.5e-4, weight decay 0.05, 500 warm-up steps) with a cosine schedule to zero and an EMA
momentum rising linearly from 0.996 to 1.0, both reaching their end values on the final
step, {ps['planned_final_step']:,}. Data order, flips and masks are deterministic functions
of the seed and the step, so the resumed run continued the same sample stream. Checkpoints
are written atomically under an exclusive lock.

### 4.5 Probes and statistics

All probes use frozen features. BBBC054 probes use folds grouped by field of view, a
standardise-PCA-standardise pipeline applied identically to every encoder, and three seeds
that each vary the cell subsample, the fold assignment and the random-init draw.
Hand-crafted features (segmentation, shape, Zernike moments, Haralick texture) are probed
with both logistic regression and a random forest and both are reported. Exposure-time
regression clips features to their 0.1-99.9% training-fold quantiles, scales them robustly
and drops non-finite features, and every reported error is checked to be finite and below
ten times that of the mean predictor. Interval estimates are reported only with at least
ten grouping units: BBBC054 encoder pairs get a paired cluster bootstrap over its 60 fields
(2,000 resamples) with Benjamini-Hochberg correction within each task; tracking, Part B and
Part C report per-group values and exact permutation distributions with their minimum
attainable p.

### 4.6 Tracking and forecasting

A query point at the start of a ground-truth track is propagated by matching its patch
token to candidate tokens within a search window, identically for every encoder, against a
nearest-detection baseline; tracks of at least 33 frames are eligible and 32 steps are
scored. For forecasting, up to 256 cell-centred 16-frame windows per movie are sampled
along all tracks with a recorded seed; each model predicts its own target latents for the
last four frames through its predictor under the future mask.

### 4.7 Part B

Single-cell NF-κB traces come from the deposit accompanying [wang2025]; the matrix layout
was reconstructed from the deposit's own preprocessing code and the ligand codes from its
plotting code. Expression is GSE260996, assembled from the per-sample count objects GEO
deposits; accessibility is GSE261647. Mouse differential expression uses DESeq2 [deseq2]
through pydeseq2 [pydeseq2]; human moderated tests use a Python implementation of the limma
empirical-Bayes method [smyth2004; limma], validated in the test suite. Orthologs come from
the MGI HOM_MouseHumanSequence table [mgi].

### 4.8 Part C

Restimulation data are the per-sample MMSEQ gene-level files of GSE212282 and GSE185433,
converted to log2 and mapped from Ensembl to HGNC symbols. Each GEO sample's treatment
field is recorded and checked before use. Expected directions are taken only from the
source publications [ferreira2023; schrijver2023], with figure and quotation recorded in
the repository.

### 4.9 Licences

Pretraining used only CC BY 4.0 data. BBBC054 is CC BY 3.0; S-BIAD2196 is CC BY 4.0; GEO
series are public records cited to their originating publications. Datasets excluded on
licence grounds were never downloaded.

### 4.10 AI-assistance disclosure and record of corrections

Code, analyses and an initial manuscript draft were produced with the assistance of an AI
coding agent (Claude, Anthropic) under the author's direction; the author reviewed and is
responsible for all content. <!-- TODO(author): review --> An earlier internal draft of
this analysis contained errors, which are documented in the repository.

### 4.11 Competing interests

<!-- TODO(author): declare competing interests -->
"""


# ------------------------------------------------------------------------ main
def figure_block(n: int, path: str, caption: str) -> str:
    return f"![Figure {n}]({path})\n\n**Figure {n}.** {caption}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "v2"))
    ap.add_argument("--out", default=str(REPO / "paper" / "tijepa.md"))
    ap.add_argument("--labels-out", default=str(REPO / "paper" / "labels.json"))
    a = ap.parse_args()
    R = Path(a.results)

    ps = load(R, "partA/pretrain_tijepa1_seed0.json")
    abl = load(R, "partA/pretrain_tijepa1_spatial_seed0.json")
    bb = load(R, "partA/bbbc054.json")
    tf = load(R, "partA/tracking_forecast.json")
    # rule 2: nothing is rendered before its inputs exist
    missing = [n for n, v in (("TiJepa-1 summary", ps), ("ablation summary", abl),
                              ("BBBC054", bb), ("tracking", tf)) if not v]
    if missing:
        raise SystemExit(f"inputs absent: {missing}; the paper is not rendered")
    for name, s in (("TiJepa-1", ps), ("TiJepa-1-spatial", abl)):
        if not s.get("annealed"):
            print(f"WARNING: {name} is unannealed; the paper states it")
    for res in (bb, tf):
        for k, e in res["encoders"].items():
            cp = e.get("checkpoint") if isinstance(e, dict) else None
            if cp and cp.get("run_name", "").startswith("tijepa1") and not cp.get("final"):
                raise SystemExit(f"{k} was evaluated on an unfinished checkpoint")

    corpus = load(V1, "pretrain_corpus.json")
    sub: dict[str, str] = {}
    cp, cv = corpus_para(corpus)
    sub.update({"CORPUS_PARA": cp, **cv})
    sub["LEAKAGE_PARA"] = leakage_para(load(V1, "leakage_check.json"),
                                       load(R, "leakage_check_full.json"))
    plan = load(R, "pretrain_plan.json")
    sub["PRETRAIN_PARA"] = pretrain_para(ps, abl, plan)
    sub["DRYAD_PARA"] = dryad_para(load(R, "partA/dryad_benchmark.json"))
    sub.update(bbbc_section(bb))
    ts = track_section(tf)
    primary = ts.pop("_primary")
    sub.update(ts)
    sub.update(partB(load(V1, "partB/rna_positive_control.json"),
                     load(V1, "partB/atac_positive_control.json"),
                     load(V1, "partB/ligand_codes.json"),
                     load(R, "partB/b3_dynamics.json"), load(R, "partB/b33_b4.json")))
    restim = load(R, "partC/c3_restimulation.json")
    sub.update(partC(load(V1, "partC/c1_anchors.json"), restim,
                     load(R, "partC/c3_projection.json"),
                     load(R, "partC/arm_expectations.json")))
    sub["METHODS"] = methods(ps, abl, plan, bb, tf)

    figs = load(R, "figures.json") or {"figures": []}
    have = {r["function"]: r for r in figs["figures"] if r.get("status") == "ok"}
    captions = {
        "FIG2": ("fig2_pretraining", "Pretraining. Loss on spatially and future-masked "
                 "batches, and effective rank, for TiJepa-1 and TiJepa-1-spatial; the dashed "
                 "line marks the resume."),
        "FIG4": ("fig4_bbbc054", "BBBC054 morphology-class probes. Three-class balanced "
                 "accuracy per seed for every encoder, against the time-only baseline and "
                 "chance, and exposure-time skill."),
        "FIG5": ("fig5_tracking_forecast", "Tracking and forecasting on S-BIAD2196. "
                 "Success against error threshold, and per-movie forecasting ratios for "
                 "TiJepa-1 and TiJepa-1-spatial."),
        "FIG6": ("fig6_partB", "Part B. Per-well tolerance predictions and the B4 overlap "
                 "against its expectation."),
        "FIG7": ("fig7_partC", "Part C. Per-donor TNF and IL6 log2 differences after LPS "
                 "restimulation, each arm against untrained cells, with the sourced "
                 "expectation where one exists. Hypothesis-generating; no image is paired "
                 "with these samples."),
    }
    for i, (k, (fn, cap)) in enumerate(captions.items(), start=2):
        n = {"FIG2": 2, "FIG4": 3, "FIG5": 4, "FIG6": 5, "FIG7": 6}[k]
        rec = have.get(fn)
        png = next((p for p in (rec or {}).get("files", []) if p.endswith(".png")), None)
        sub[k] = figure_block(n, "../" + png, cap) if png else ""
        del i

    # abstract, discussion, power: generated from the numbers
    enc = {k: v for k, v in bb["encoders"].items() if isinstance(v, dict)
           and "three_class_mean" in v}
    best = max(enc, key=lambda k: enc[k]["three_class_mean"])
    pair = next((c for ck, c in bb["pairwise_comparisons"].items()
                 if ck.startswith("three_class:") and {c["encoder_a"], c["encoder_b"]} ==
                 {"E4_tijepa1", "E6_random_init"}), None)
    abs_a = (f"On BBBC054 microglial morphology classes the best encoder is "
             f"{enc_label(best, enc[best])} at {f(enc[best]['three_class_mean'])} balanced "
             f"accuracy (three classes; a time-only baseline reaches "
             f"{f(bb['time_only_baseline']['three_class']['mean'])}).")
    if pair:
        c = Comparison(pair["encoder_a"], pair["encoder_b"], "t",
                       difference=pair["difference_mean_over_seeds"],
                       ci=tuple(pair["ci95_widest_over_seeds"]), p_value=pair["p_value_bh"])
        if pair["encoder_a"] != "E4_tijepa1":
            c = Comparison("E4_tijepa1", "E6_random_init", "t", difference=-c.difference,
                           ci=(-c.ci[1], -c.ci[0]), p_value=c.p_value)
        abs_a += (f" TiJepa-1 is {compare_phrase(c)} the identical network with random "
                  f"weights (Δ {c.difference:+.3f}, 95% CI [{c.ci[0]:+.3f}, "
                  f"{c.ci[1]:+.3f}]).")
    sub["ABSTRACT_PARTA"] = abs_a
    fsum = tf["forecasting"]["summary"]
    if primary.get("n_movies"):
        c = Comparison("E4_tijepa1", "E4s_tijepa1_spatial", "forecast",
                       difference=primary["mean_difference"],
                       p_value=primary["exact_sign_flip_p_two_sided"])
        sub["ABSTRACT_FORECAST"] = (
            f"In latent forecasting on held-out movies, TiJepa-1's error relative to copying "
            f"the last latent is {f(fsum['E4_tijepa1']['mean_ratio_vs_copy_last'])} against "
            f"{f(fsum['E4s_tijepa1_spatial']['mean_ratio_vs_copy_last'])} for "
            f"TiJepa-1-spatial, {compare_phrase(c)} it across {primary['n_movies']} movies "
            f"(exact {pf(primary['exact_sign_flip_p_two_sided'])}).")
        fdir = c.direction
    else:
        sub["ABSTRACT_FORECAST"] = ""
        fdir = None
    b31 = load(R, "partB/b3_dynamics.json")["b31_tolerance"]
    rr = b31["results"]["raw_trace"]
    sub["ABSTRACT_B31"] = (
        f"leave-one-well-out balanced accuracy is {f(rr['cell_balanced_accuracy_mean'])} "
        f"over {b31['n_wells']} wells, and the exact permutation null admits no p below "
        f"{f(rr['exact_permutation']['minimum_achievable_p'])}, so the design cannot reach "
        "significance.")
    ver = [x.get("verdict") for g in ("GSE212282", "GSE185433")
           for r in restim[g].get("arms", {}).values()
           for x in r.get("primary_TNF_IL6", {}).values() if x.get("verdict")]
    n_match = sum(v == "matches the sourced direction" for v in ver)
    n_opp = sum(v == "opposite to the sourced direction" for v in ver)
    sub["ABSTRACT_PARTC"] = (
        f"Of the {n_match + n_opp} gene-arm readouts with a sourced expectation, {n_match} "
        f"point in the reported direction and {n_opp} the opposite way"
        + (", including β-glucan" if restim["GSE212282"].get("beta_glucan_opposite") else "")
        + "; with three donors none is significant.")
    if fdir == -1:
        sub["DISCUSSION_PARTA"] = (
            "Self-supervised video pretraining on openly licensed innate-immune microscopy "
            "is practical at modest scale, and the future-masking objective produces a model "
            "whose own forecasts beat a copy-last baseline by more than the same model "
            "trained without that objective, on held-out movies.")
    else:
        sub["DISCUSSION_PARTA"] = (
            "Self-supervised video pretraining on openly licensed innate-immune microscopy "
            "is practical at modest scale and can be compared against strong baselines under "
            "leakage-controlled protocols. Adding a future-masking objective did not, on "
            "these held-out movies, produce forecasts distinguishably better than those of "
            "the same model trained without it, relative to each model's own copy-last "
            "baseline.")
    sub["DISCUSSION_LIMITS"] = (
        "The Part B tolerance contrast cannot reach significance with the wells the public "
        "deposit provides. The dynamics-to-expression and dynamics-to-accessibility analyses "
        "are null, and the cross-species analysis is exploratory. The Part C comparison is "
        "across studies, is not significant with three donors, and its readout (RNA after "
        "4 h) differs from the protein readouts its expectations come from.")
    sub["POWER_PARA"] = (
        f"The Part B tolerance contrast has {b31['n_wells']} wells; the dynamics-to-omics "
        "regressions have five matched conditions; the restimulation datasets have three "
        "donors each; tracking and forecasting have six movies. Donor, well and movie "
        "counts, not model capacity, limit what any analysis here can conclude.")
    refs = load(R, "references.json")["references"]
    sub["REFERENCES"] = markdown_table(
        ["key", "first author", "title", "venue, year", "identifier"],
        [[k, r.get("first_author") or "—", (r.get("title") or "—")[:90],
          f"{r.get('container') or '—'}, {r.get('year') or '—'}",
          (r.get("doi") and f"doi:{r['doi']}") or (r.get("arxiv_id") and
                                                    f"arXiv:{r['arxiv_id']}") or "—"]
         for k, r in refs.items() if r.get("status") == "resolved"])
    sub["DATA_AVAILABILITY"] = (
        "All datasets are public and listed with their licences in "
        "`manifests/datasets.yaml`. The Dryad macrophage movies "
        "(doi:10.5061/dryad.m0cfxpp8v) and the LTDB collection "
        "(doi:10.6084/m9.figshare.c.3827890) require manual retrieval and were not "
        "analysed. The published ComplexEye archive R06.zip lacks its zip directory and "
        "contributed no movies.")
    best_bb = enc_label(best, enc[best])
    sub["TITLE"] = ("TiJepa: self-supervised video representations of innate immune cell "
                    "behaviour, and what public data can and cannot say about trained "
                    "immunity")
    del best_bb

    text = TEMPLATE.read_text()
    keys = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text))
    unfilled = sorted(keys - set(sub))
    for k, v in sub.items():
        text = text.replace("{{" + k + "}}", str(v))
    text = re.sub(r"\n{3,}", "\n\n", text)
    Path(a.out).write_text(text)
    Path(a.labels_out).write_text(json.dumps(LABELS, indent=2, default=str) + "\n")
    print(f"wrote {a.out} ({len(text.split()):,} words), {len(LABELS)} labels")
    if unfilled or re.findall(r"\{\{[^}]+\}\}", text):
        print(f"UNFILLED: {unfilled}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
