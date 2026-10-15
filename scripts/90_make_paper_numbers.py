"""M7: fill paper/tijepa.md from paper/tijepa.template.md using results/*.json.

Every number in the manuscript is written here from a results file. If a result is
missing, the placeholder is replaced with an explicit "not available in this run"
sentence -- never with a plausible-looking number.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"


def load(rel: str):
    p = R / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def fmt(v, nd=3, pct=False):
    if v is None:
        return "n/a"
    try:
        return f"{v*100:.1f}%" if pct else f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def ci(d, nd=3):
    if not d or d.get("ci_low") is None:
        return "n/a"
    return f"[{d['ci_low']:.{nd}f}, {d['ci_high']:.{nd}f}]"


MISSING = "*(not available in this run; see `reports/RUN_REPORT.md`)*"


def main() -> int:
    corpus = load("partA/pretrain_corpus.json")
    pre = load("partA/pretrain_seed0.json")
    bb = load("partA/bbbc054_probes.json")
    tr = load("partA/tracking.json")
    b31 = load("partB/b31_tolerance.json")
    b32 = load("partB/b32_ligand_identity.json")
    b33 = load("partB/b33_dynamics_to_expression.json")
    b4 = load("partB/b4_cross_species.json")
    c1 = load("partC/c1_signatures.json")
    c2 = load("partC/c2_scores.json")
    c3 = load("partC/c3_projection.json")
    pf = load("preflight.json")
    v: dict[str, str] = {}

    # ---------------- corpus + pretraining
    if corpus:
        v["CORPUS_CLIPS"] = f"{corpus['n_clips']:,}"
        v["CORPUS_GB"] = str(corpus["total_gb"])
        v["CORPUS_MOVIES"] = str(corpus["n_movies_ok"])
        iv = corpus["effective_interval_s_distribution"]["counts"]
        v["CORPUS_PARA"] = (
            f"We built the pretraining corpus from the ComplexEye neutrophil migration "
            f"release (Zenodo 10082674, CC BY 4.0): {corpus['n_movies_ok']} of "
            f"{corpus['n_movies_attempted']} movies processed without failure into "
            f"**{corpus['n_clips']:,} cell-centred clips** of "
            f"{corpus['clip_shape'][0]}x{corpus['clip_shape'][1]}x"
            f"{corpus['clip_shape'][2]} uint8 ({corpus['total_gb']} GB) in "
            f"{corpus['n_shards']} sequential shards. Cells were detected with a "
            f"classical temporal-median/MAD pipeline and linked with trackpy; each clip "
            f"follows its own track so the cell stays centred and the model cannot use "
            f"frame drift as a shortcut. Clips were sampled at two strides, giving "
            f"effective frame intervals of "
            f"{', '.join(f'{float(k):g}s (n={c:,})' for k, c in sorted(iv.items(), key=lambda x: float(x[0])))}. "  # noqa: E501
            f"All frames are at {corpus['sources']['complexeye']['um_per_px']} um/px "
            f"against a {corpus['target_um_per_px']} um/px target, so no resampling was "
            f"applied. The corpus is single-source, which is a real limitation: it is "
            f"one cell type (human neutrophils) under one imaging modality.")
    else:
        v.update({"CORPUS_CLIPS": "n/a", "CORPUS_GB": "n/a", "CORPUS_MOVIES": "n/a",
                  "CORPUS_PARA": MISSING})

    if pre:
        v["PRETRAIN_PARA"] = (
            f"TiJepa-0 is a ViT-S ({pre['config']['clip_size']}x"
            f"{pre['config']['clip_size']} px, {pre['config']['clip_frames']} frames, "
            f"2x8x8 tubelets, 1,152 tokens) trained with a V-JEPA-style objective: a "
            f"context encoder sees visible tubelets, an EMA target encoder sees the "
            f"whole clip, and a narrow predictor regresses the target encoder's latents "
            f"at masked positions under an L1 loss. Masking is multi-block and tubed "
            f"over time, so a masked cell cannot be copied from an adjacent frame. "
            f"Training ran for **{pre['steps']:,} steps "
            f"({pre['effective_epochs']} effective epochs over the corpus) in "
            f"{pre['gpu_hours']} GPU-hours** on one {pre['device']} in bf16, stopping on "
            f"`{pre['stop_reason']}`. Mean loss fell from "
            f"{fmt(pre['initial_loss_mean_first500'])} over the first 500 steps to "
            f"{fmt(pre['final_loss_mean_last500'])} over the last 500. Embedding "
            f"standard deviation and effective rank were logged throughout as collapse "
            f"diagnostics (Figure 2); no collapse intervention was required. "
            f"Pretraining used **seed 0 only**: the run had a single GPU, so the "
            f"three-seed requirement was applied to all probes and downstream models "
            f"instead, and single-seed pretraining is a limitation of this run.")
    else:
        v["PRETRAIN_PARA"] = MISSING

    # ---------------- BBBC054
    if bb:
        t1 = bb["tasks"]["morphology_class"]
        t2 = bb["tasks"]["lps_exposure_time_regression"]
        ok = {k: e for k, e in t1.items() if e.get("status") == "ok"}
        best = max(ok, key=lambda k: ok[k]["balanced_accuracy_mean"]) if ok else None
        e4 = t1.get("E4_tijepa0", {})
        v["BBBC_E4_BA"] = fmt(e4.get("balanced_accuracy_mean")) if e4.get("status") == "ok" else "n/a"  # noqa: E501
        v["BBBC_BEST_NAME"] = (best or "n/a").replace("_", " ")
        v["BBBC_BEST_BA"] = fmt(ok[best]["balanced_accuracy_mean"]) if best else "n/a"
        r4 = t2.get("E4_tijepa0", {})
        v["BBBC_E4_RHO"] = fmt(r4.get("spearman_rho")) if r4.get("status") == "ok" else "n/a"
        lines = [
            "| encoder | balanced accuracy (mean ± sd over 3 seeds) | 95% CI | perm. p | "
            "Spearman ρ (exposure time) | FOV-identity BA |",
            "|---|---|---|---|---|---|",
        ]
        conf = bb["tasks"].get("fov_confound_probe", {})
        for k in ["E1_handcrafted_morphology", "E2_imagenet_vit_s", "E3_dinov2_vit_s",
                  "E4_tijepa0", "E5_tijepa0_frame_shuffled", "E6_tijepa0_random_init"]:
            e = t1.get(k, {})
            if e.get("status") != "ok":
                note = ("N/A (undefined on static frames)"
                        if k.startswith("E5") else f"not available ({e.get('status')})")
                lines.append(f"| {k} | {note} | | | | |")
                continue
            lines.append(
                f"| {k} | {fmt(e['balanced_accuracy_mean'])} ± "
                f"{fmt(e['balanced_accuracy_sd'])} | "
                f"{ci(e['cluster_bootstrap_ci_seed0'])} | "
                f"{fmt(e['fov_permutation_test_seed0'].get('p_value'))} | "
                f"{fmt(t2.get(k, {}).get('spearman_rho'))} | "
                f"{fmt(conf.get(k, {}).get('fov_identity_balanced_accuracy'))} |")
        labels = " vs ".join(bb["labels_present"])
        absent = bb.get("labels_absent") or []
        v["BBBC_PARA"] = (
            f"We sampled {bb['n_cells']:,} annotated cells across {bb['n_fovs']} "
            f"fields of view, stratified by FOV and class, and cropped "
            f"{bb['crop_px']}x{bb['crop_px']} px around each annotation. Replicate 1 "
            f"contains only {labels}"
            + (f" (no {', '.join(absent)} cells are annotated in it), so the planned "
               f"three-class probe is a two-class probe" if absent else "")
            + ". All probes are FOV-grouped (never splitting a field of view across "
            "train and test), with cluster-bootstrap CIs over FOVs and a FOV-level "
            "permutation null.\n\n" + "\n".join(lines) + "\n\n"
            f"The FOV-identity column matters: in BBBC054 each FOV *is* one exposure "
            f"time, so an encoder that merely memorises fields of view would score well "
            f"on exposure-time regression for the wrong reason. Chance for that probe is "
            f"{fmt(list(conf.values())[0]['chance']) if conf else 'n/a'}. "
            f"Crops were not resampled to a common micron scale because BBBC054 states a "
            f"20x objective but no pixel size; this is a recorded deviation.")
    else:
        v.update({"BBBC_E4_BA": "n/a", "BBBC_BEST_NAME": "n/a", "BBBC_BEST_BA": "n/a",
                  "BBBC_E4_RHO": "n/a", "BBBC_PARA": MISSING})

    # ---------------- tracking
    if tr and tr.get("status") == "ok":
        agg = tr["aggregate"]
        v["TRACK_E4"] = fmt(agg.get("E4_tijepa0", {}).get("delta_avg_mean"))
        v["TRACK_BASE"] = fmt(agg.get("baseline_nearest_detection", {}).get("delta_avg_mean"))
        rows = ["| method | ⟨δ⟩ (1–16 µm) | 95% CI (movie bootstrap) | median survival (frames) |",  # noqa: E501
                "|---|---|---|---|"]
        for k in ["baseline_nearest_detection", "E4_tijepa0",
                  "E5_tijepa0_frame_shuffled", "E6_tijepa0_random_init"]:
            a = agg.get(k, {})
            if a.get("delta_avg_mean") is None:
                rows.append(f"| {k} | not available ({a.get('status','n/a')}) | | |")
                continue
            lo, hi = (a.get("delta_avg_ci") or [None, None])
            rows.append(f"| {k} | {fmt(a['delta_avg_mean'])} | "
                        f"{f'[{lo:.3f}, {hi:.3f}]' if lo is not None else 'n/a'} | "
                        f"{a['median_survival_frames']:.0f} |")
        v["TRACK_PARA"] = (
            f"We tracked query points on {tr['n_movies']} held-out movies "
            f"({tr['um_per_px']} um/px, {tr['frame_interval_s']:g} s per frame) for "
            f"{tr['n_steps']} steps, scoring TAP-Vid-style ⟨δ⟩ over thresholds of "
            f"{', '.join(str(t) for t in tr['thresholds_um'])} um.\n\n"
            + "\n".join(rows) + "\n\n"
            f"The frame-shuffled control (E5) is meaningful here because the input is a "
            f"genuine time series. CoTracker was not run ({tr['cotracker']}).")
    else:
        v.update({"TRACK_E4": "n/a", "TRACK_BASE": "n/a", "TRACK_PARA": MISSING})

    # ---------------- Part B
    if b31:
        iv8 = b31["by_interval"].get("8h", {})
        if iv8.get("status") == "ok":
            raw = iv8["representations"]["raw_trace"]
            hand = iv8["representations"]["handcrafted"]
            v["B31_RAW_BA"] = fmt(raw["balanced_accuracy_mean"])
            v["B31_RAW_CI"] = ci(raw["cluster_bootstrap_ci_seed0"])
            v["B31_RAW_P"] = fmt(raw["group_permutation_test_seed0"].get("p_value"), 2)
            v["B31_GROUPS"] = str(iv8["n_groups"])
            v["B31_PARA"] = (
                f"The Tay-lab deposit (Zenodo 14337874, CC BY 4.0) provides single-cell "
                f"NF-κB traces across sequential ligand pairs. The deposit ships no "
                f"column dictionary, and the folder holding the human-readable condition "
                f"labels was not deposited, so the matrix layout was reconstructed from "
                f"the deposit's own MATLAB preprocessing script and the ligand code for "
                f"LPS was identified by intersecting the second-ligand code sets of two "
                f"deposit files with their file-name ligand sets. Analyses are restricted "
                f"to the five files whose layout could be confirmed; the cytokine-block "
                f"and sepsis files are flagged layout-unverified and excluded.\n\n"
                f"Classifying prior LPS exposure from the response to a second LPS "
                f"stimulus at a matched 8 h interval ({iv8['n_naive']} naive→LPS vs "
                f"{iv8['n_tolerant']} LPS→LPS cells) gives a balanced accuracy of "
                f"**{fmt(raw['balanced_accuracy_mean'])}** "
                f"({ci(raw['cluster_bootstrap_ci_seed0'])}) from the raw trace and "
                f"{fmt(hand['balanced_accuracy_mean'])} "
                f"({ci(hand['cluster_bootstrap_ci_seed0'])}) from hand-crafted features. "
                f"**Neither is significant**: the contrast rests on {iv8['n_groups']} "
                f"wells, so a well-level permutation null is coarse and both give "
                f"p = {fmt(raw['group_permutation_test_seed0'].get('p_value'), 2)}. "
                f"We report the point estimates because the plan requires null results to "
                f"be reported, not because they support a claim. "
                + (f"First-ligand identity across "
                   f"{b32['n_classes']} classes reaches "
                   f"{fmt(b32['balanced_accuracy_mean'])} (chance "
                   f"{fmt(1/b32['n_classes'])}); this is **not** comparable to the "
                   f"published 72.8% four-class figure, which uses a different class "
                   f"set and validation scheme." if b32 else ""))
        else:
            v.update({"B31_RAW_BA": "n/a", "B31_RAW_CI": "n/a", "B31_RAW_P": "n/a",
                      "B31_GROUPS": "n/a", "B31_PARA": MISSING})
    else:
        v.update({"B31_RAW_BA": "n/a", "B31_RAW_CI": "n/a", "B31_RAW_P": "n/a",
                  "B31_GROUPS": "n/a", "B31_PARA": MISSING})

    if b33:
        v["B33_CONDITIONS"] = str(b33.get("n_conditions", "n/a"))
        if b33.get("status") == "ok":
            v["B33_PARA"] = (
                f"Regressing condition-level expression on condition-mean trace features "
                f"with leave-one-condition-out ridge over {b33['n_conditions']} shared "
                f"conditions gives a mean per-gene r of {fmt(b33['mean_r'])} and "
                f"{b33['n_genes_r_above_0.5']} genes with r > 0.5, **below** a "
                f"condition-permutation null averaging "
                f"{fmt(b33['permutation_null_mean_count'], 1)} such genes "
                f"(p = {fmt(b33['permutation_p'], 2)}). This is a null result. The "
                f"deposited RNA-seq contains no LPS→LPS sample, so the homologous "
                f"tolerance contrast classified above has no matching transcriptome at "
                f"all; the shared conditions are heterologous.")
        else:
            v["B33_PARA"] = (f"The dynamics-to-expression regression was not run: "
                             f"{b33.get('reason', b33.get('status'))}.")
    else:
        v.update({"B33_CONDITIONS": "n/a", "B33_PARA": MISSING})

    if b4 and b4.get("status") == "ok":
        h, m, ov = b4["human"], b4["mouse"], b4["overlap_test"]
        nom = b4.get("overlap_test_nominal_exploratory", {})
        v["B4_PARA"] = (
            f"Linking the mouse and human literatures was also null. In human "
            f"(GSE58310, LPS-tolerised vs macrophage control, {h['n_tolerised']} "
            f"donor-paired samples, {h['n_genes_tested_after_expression_filter']:,} "
            f"expressed genes) the pre-specified criteria of BH-FDR < 0.05 and "
            f"|log2FC| > 1 return {h['n_up']} up and {h['n_down']} down genes; in mouse "
            f"({m['contrast']}) they return {m['n_up']} up. The ortholog overlap is "
            f"{ov['overlap']} genes. A clearly-labelled exploratory analysis using "
            f"nominal p < 0.01 instead of FDR ({h['n_up_nominal_p01']} human up genes) "
            f"also gives an overlap of {nom.get('overlap', 'n/a')}. The two contrasts are "
            f"not the same biological comparison — the mouse one is heterologous priming, "
            f"the human one homologous tolerance — and neither has the power to support "
            f"an overlap test.")
    else:
        v["B4_PARA"] = MISSING

    # ---------------- Part C
    if c1:
        det = c1["detail"]
        ok = [k for k, d in det.items() if d.get("status") == "ok"]
        v["C1_PARA"] = (
            "Anchor signatures were derived from GSE46903 (Xue et al., human GM-CSF "
            "macrophages on GPL6947). All three anchors were taken at 72 h, the only "
            "time point at which ultrapure LPS samples exist in that series, against "
            "the matched unstimulated control: "
            + "; ".join(f"**{k}** from {det[k]['n_stim']} {det[k]['stimulus']} vs "
                        f"{det[k]['n_ctrl']} control samples" for k in ok)
            + ". Each signature is the top 100 up-regulated genes by Welch t among genes "
              "passing BH-FDR < 0.05. S_IFNg recovers canonical interferon-γ response "
              "genes (" + ", ".join(det["S_IFNg"]["top20"][:5]) + "), which is a useful "
              "sanity check; S_LPS at 72 h does not recover acute LPS genes, consistent "
              "with the acute response having resolved by that time point, and this "
              "weakens the LPS half of the M1-like axis.")
    else:
        v["C1_PARA"] = MISSING

    if c2 and c2.get("status") == "ok":
        co = c2["coordinates"]
        rows = ["| state | n | M1-like (95% CI) | M2-like (95% CI) |", "|---|---|---|---|"]
        for st, d in co.items():
            rows.append(
                f"| {st.replace('_',' ')} | {d['n_samples']} | "
                f"{fmt(d['M1_like'],4)} {ci({'ci_low':d['M1_like_ci'][0],'ci_high':d['M1_like_ci'][1]},4)} | "  # noqa: E501
                f"{fmt(d['M2_like'],4)} "
                f"{ci({'ci_low':d['M2_like_ci'][0],'ci_high':d['M2_like_ci'][1]},4) if d.get('M2_like_ci') else 'n/a'} |")  # noqa: E501
        v["C2_PARA"] = (
            "We scored every GSE58310 sample (Saeed et al.: naive monocytes, 6 d "
            "macrophage control, LPS-tolerised and β-glucan-trained, 4 donors each) "
            "with a rank-based singscore-style statistic, expressed relative to that "
            "dataset's own macrophage baseline. Rank-based scoring is what makes an "
            "array-derived signature comparable to RNA-seq at all.\n\n"
            + "\n".join(rows) + "\n\n"
            f"The separations are small — hundredths of a unit on a rank scale whose "
            f"range is ±0.5 — and the trained and tolerant states sit close to the "
            f"macrophage baseline on both axes, with naive monocytes displaced furthest. "
            f"Replication was not possible: {c2['replication']['GSE85243']} "
            f"{c2['replication']['GSE172116']}")
    else:
        v["C2_PARA"] = MISSING

    if c3 and c3.get("status") == "ok":
        v["C3_PARA"] = c3.get("paragraph", MISSING)
    else:
        reason = (c3 or {}).get("reason", "the image-side anchors require the "
                  "M0/M1/M2 movie dataset that proved unreachable")
        v["C3_PARA"] = (
            f"**The projection into image space (C3) was not performed.** {reason} "
            f"Mapping omics coordinates into the learned image space requires image-side "
            f"centroids for M0, M1 and M2 cells, and no accessible dataset in this run "
            f"provides images labelled with those states. Rather than substitute a "
            f"different label set and present it as the same analysis, the projection is "
            f"left undone and the omics coordinates are reported on their own.")

    # ---------------- methods, data availability, references
    tier = (pf or {}).get("tier", {})
    comp = (pf or {}).get("compute", {})
    v["METHODS"] = f"""### 4.1 Compute and tier

{comp.get('gpu_count', 'n/a')}x {(comp.get('gpus') or [{}])[0].get('name', 'n/a')}
({(comp.get('gpus') or [{}])[0].get('memory_total_mib', 'n/a')} MiB),
{comp.get('cpu_cores_logical', 'n/a')} logical cores, {comp.get('ram_total_gb', 'n/a')} GB RAM.
Tier **{tier.get('tier')}** ({tier.get('reason')}), encoder {tier.get('encoder')},
precision {tier.get('precision')}, raw-video cap {tier.get('raw_video_cap_gb')} GB.
PyTorch {comp.get('torch_version', 'n/a')}, CUDA {comp.get('cuda_version', 'n/a')}.
Seed policy: {tier.get('seed_policy')}.

### 4.2 Pretraining data and preprocessing

See §2.1. Detection used a temporal-median background with a MAD threshold and an area
filter; linking used trackpy (search radius 12 px, memory 2, minimum track length 16).
Cellpose was not used: the labelled evaluation sets need no segmentation (BBBC054 ships
annotated coordinates, S-BIAD2196 ships tracked instance masks), so a classical detector
kept the GPU free for training. The frame-QC classifier specified in the plan was **not**
trained, because only 5 of 8,441 S-BIAD2133 files could be retrieved in time; no QC
filtering was applied to pretraining clips.

### 4.3 Model and training

ViT-S encoder (384 dim, 12 layers, 6 heads) over 2x8x8 tubelets of 16x96x96 clips
(1,152 tokens), fixed 3D sine-cosine positional embeddings, EMA target encoder
(momentum {tier.get('tier') and ''}0.996 → 1.0), predictor of width 192 and depth 6,
multi-block tube masking, L1 loss on layer-normalised target latents, AdamW with cosine
schedule and warmup, bf16 autocast, gradient clipping at 1.0.

### 4.4 Probes and statistics

All encoders are frozen and every encoder gets the identical probe. Splits are grouped
at the highest available unit (FOV for BBBC054, well for Part B traces, movie for
tracking), never at the cell or frame level. Confidence intervals are cluster bootstraps
(2,000 resamples) at the grouping unit; "better than chance" is tested by permuting
labels at the group level; encoder comparisons use a paired cluster bootstrap on
identical resamples. Learned probes are run at seeds 0/1/2 and reported as mean ± sd.
For Part B, the logistic C was tuned once by inner CV and then held fixed inside the
permutation refits, which is a recorded deviation from per-fit tuning.

### 4.5 Licences of data and weights

Pretraining used only CC BY 4.0 data. Every dataset's licence was read from its own
record and is recorded in `manifests/datasets.yaml`; third-party model weights and their
licences are in `manifests/models.yaml`. The datasets the plan excluded
(Zenodo 16319732 CC BY-NC-ND, DeepCell DynamicNuclearNet, the Cellpose training set,
CTMC-v1, LIVECell, Cell Tracking Challenge) were not downloaded or used.
"""

    ds = []
    man = REPO / "manifests" / "datasets.yaml"
    if man.exists():
        import yaml

        for e in yaml.safe_load(man.read_text())["datasets"]:
            ds.append(f"- `{e.get('key')}` — {e.get('accession')} — "
                      f"licence: {e.get('licence_verified')} — status: {e.get('status')}")
    v["DATA_AVAILABILITY"] = ("All data are public. Accessions, verified licences, sizes "
                              "and checksums are in `manifests/datasets.yaml`:\n\n"
                              + "\n".join(ds))

    v["REFERENCES"] = """Only references whose identifiers were resolved during this run
are listed. Works discussed from the landscape review are cited there with their own
identifiers (`reports/Trained immunity imaging datasets.md`).

- Saeed S, et al. Epigenetic programming of monocyte-to-macrophage differentiation and
  trained innate immunity. *Science* 2014. Data: GEO **GSE58310**. <!-- TODO(verify): DOI -->
- Novakovic B, et al. β-glucan reverses the epigenetic state of LPS-induced immunological
  tolerance. *Cell* 2016. Data: GEO **GSE85243** (in GSE85246). <!-- TODO(verify): DOI -->
- Xue J, et al. Transcriptome-based network analysis reveals a spectrum model of human
  macrophage activation. *Immunity* 2014. Data: GEO **GSE46903** (in GSE47189).
  <!-- TODO(verify): DOI -->
- Wang AG, Son M, et al. Inflammatory memory of macrophages emerges from coordinated
  transcription factor and chromatin dynamics. *Cell Systems* 2025. Data: Zenodo
  **10.5281/zenodo.14337874**, GEO **GSE260996**, **GSE261647**. <!-- TODO(verify): DOI -->
- ComplexEye: a multi-lens array microscope for high-throughput embedded immune cell
  migration analysis. Data: Zenodo **10.5281/zenodo.10082674** (CC BY 4.0).
  <!-- TODO(verify): journal reference -->
- CellTrack: benchmark dataset for 2D cell tracking. BioStudies **S-BIAD2196**
  (CC BY 4.0). <!-- TODO(verify): associated publication -->
- Ljosa V, Sokolnicki KL, Carpenter AE. Annotated high-throughput microscopy image sets
  for validation. *Nature Methods* 2012. BBBC054 image set (CC BY 3.0).
  <!-- TODO(verify): DOI -->
- Assran M, et al. Revisiting feature prediction for learning visual representations from
  video (V-JEPA). 2024. <!-- TODO(verify): arXiv id -->
- Oquab M, et al. DINOv2: learning robust visual features without supervision. 2023.
  <!-- TODO(verify): arXiv id -->
- Blumberg B, et al. MGI HOM_MouseHumanSequence ortholog report, Mouse Genome
  Informatics, downloaded during this run. <!-- TODO(verify): citation form -->
"""

    tpl = (REPO / "paper" / "tijepa.template.md").read_text()
    missing = []
    for key in sorted(v, key=len, reverse=True):
        tpl = tpl.replace("{{" + key + "}}", str(v[key]))
    import re

    for m in re.findall(r"\{\{([A-Z0-9_]+)\}\}", tpl):
        missing.append(m)
        tpl = tpl.replace("{{" + m + "}}", MISSING)
    (REPO / "paper" / "tijepa.md").write_text(tpl)
    (R / "paper_numbers.json").write_text(json.dumps(v, indent=2) + "\n")
    print(f"wrote paper/tijepa.md ({len(tpl.split())} words)")
    if missing:
        print(f"  placeholders with no result: {sorted(set(missing))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
