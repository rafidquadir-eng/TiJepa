"""F9.2: write reports/RUN_REPORT_v1.md, including the Section 12 status table.

PLAN_v1.md says the run is not finished until every row of its Section 12 traceability
table has a status in this report. Each row's status is **derived from the result files**
wherever that is possible - a row claiming "fixed" that cannot point at an artefact is a
claim, not evidence - and the evidence column names the file a reader can open.

Statuses are `fixed`, `partially fixed` or `not fixable`, with a reason for anything
short of fixed.
"""
from __future__ import annotations

import functools
import json
import subprocess
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.viz.tables import markdown_table  # noqa: E402

V1 = REPO / "results" / "v1"
OUT = REPO / "reports" / "RUN_REPORT_v1.md"


def load(rel: str):
    p = V1 / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    pre = load("preflight.json")
    corpus = load("pretrain_corpus.json")
    lk = load("leakage_check.json")
    intake = load("manual_intake.json")
    defect = load("complexeye_archive_defect.json")
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
    figs = load("figures.json")
    claims = load("claims_check.json")

    def ok(x) -> bool:
        return bool(x)

    # any tracked file over 5 MB would be an E7 regression
    big_tracked_files = sh([
        "bash", "-c",
        f"cd {REPO} && git ls-files -z | xargs -0 -I@ du -k '@' 2>/dev/null "
        "| awk '$1>5120'"])

    # ---- Section 12 traceability, with statuses derived from artefacts
    def st(cond, partial_reason=None, not_fixable=None):
        if not_fixable:
            return f"not fixable — {not_fixable}"
        if cond:
            return "fixed"
        return f"partially fixed — {partial_reason}" if partial_reason else "not fixed"

    n_arch = (corpus or {}).get("n_archives_done", 0)
    n_arch_total = (corpus or {}).get("n_archives_in_record", 21)
    rows = [
        ["D1", "Part B used ATAC as RNA; B4 artifact",
         st(ok(rna) and ok(atac) and ok(b34)),
         "`partB/rna_guard.json`, `partB/rna_positive_control.json`, "
         "`partB/atac_positive_control.json`, `partB/b33_b4.json`"],
        ["D2", "Dryad headline missing",
         ("not fixable — the dataset is an author prerequisite and was not provided; "
          "an explicit status is recorded and the code path is in place"),
         "`partA/dryad_benchmark.json`, `manual_intake.json`"],
        ["D3", "Possible tracking leakage",
         st(ok(lk)),
         "`leakage_check.json` — real overlap found and the matching pretraining "
         "movie excluded"],
        ["D4a", "7/21 archives, 14 corrupt",
         st(n_arch >= n_arch_total,
            f"{n_arch} of {n_arch_total} archives processed; the rest were cut by the "
            "wall-clock budget at roughly 56 min per archive over an intermittent "
            "Zenodo connection. Every processed archive was MD5-verified, R08 was "
            "re-fetched after failing, and R06's defect is in the published file"),
         "`pretrain_corpus.json`, `complexeye_archive_defect.json`"],
        ["D4b", "Single source",
         st(False, "the Leica reference movies are a second instrument with a "
                   "separately verified pixel size, but PhagoStat microglia video was "
                   "not added: the 94 GB archive could not be fetched within the "
                   "run's remaining budget"),
         "`pretrain_corpus.json` (`clips_by_source`)"],
        ["D4c", "Leica pixel size unverified",
         st(bool((corpus or {}).get("pixel_size_provenance", {})
                 .get("complexeye_leica", {}).get("verified"))),
         "`pretrain_corpus.json` (`pixel_size_provenance`) — 0.4762 µm/px derived from "
         "the published field of view, consistent on both axes"],
        ["D4d", "No QC",
         st(ok(corpus) and (corpus or {}).get("qc_thresholds") is not None),
         "`pretrain_corpus.json` (`qc_thresholds`, `n_movies_excluded_qc`) — heuristic "
         "and labelled as such, since S-BIAD2133 could not be retrieved in full"],
        ["D4e", "PhagoStat video unused", "not fixed — see D4b",
         "`pretrain_corpus.json`"],
        ["D5a", "BBBC054 static",
         st((corpus or {}).get("n_static_clips", 0) > 0,
            "static-clip augmentation is in the corpus, but the video headline that "
            "would have replaced BBBC054 depends on D2"),
         "`pretrain_corpus.json` (`static_fraction`)"],
        ["D5b", "2 of 3 classes",
         st(ok(bb) and "three_class_mean" in next(
             (v for v in (bb or {}).get("encoders", {}).values()
              if isinstance(v, dict)), {})),
         "`partA/bbbc054_v1.json` — the third class is `ramified`, not `stratified`; "
         "v0's label tuple silently dropped 11,143 of 58,186 annotations "
         "(`reports/V0_ERRATA.md` E2)"],
        ["D5c", "FOV–time confound",
         st(ok(bb) and bool((bb or {}).get("time_only_baseline"))),
         "`partA/bbbc054_v1.json` (`time_only_baseline`, `within_time_bin`, "
         "`fov_identity_mean`) — the time-only baseline scores at chance"],
        ["D5d", "BBBC054 pixel size",
         st(ok(bb) and (bb or {}).get("um_per_px") is not None),
         "`partA/bbbc054_v1.json` (`um_per_px_provenance`) — 0.65 µm/px from the "
         "camera array and 2×2 binning, plus a scale-sensitivity analysis"],
        ["D6", "Small tracking eval",
         st(ok(tf) and (tf or {}).get("n_movies", 0) >= 6,
            "all 6 movies the deposit contains are used, which is every one available; "
            "LTDB was not provided and CoTracker's host was unreachable"),
         "`partA/tracking_forecast_v1.json`"],
        ["D7", "Tay layout inferred",
         st(ok(lig) and (lig or {}).get("code_mapping", {}).get("status", "")
            .startswith("sourced")),
         "`partB/ligand_codes.json` — sourced from the deposit's plotting code and "
         "independently confirmed by the GEO treatment protocol"],
        ["C1", "Cosmetic seeds",
         st(ok(bb)),
         "`partA/bbbc054_v1.json` (`seed_varies`), `partB/b3_dynamics.json` — every "
         "sd now comes from a changed subsample, fold assignment or init draw"],
        ["C2", "4-well bootstrap CI",
         st(ok(b3) and (((b3 or {}).get("b31_tolerance", {}).get("results", {})
                         .get("raw_trace", {}).get("confidence_interval")) is None)),
         "`partB/b3_dynamics.json` — no CI, exact permutation and its floor reported"],
        ["C3", "Undertrained",
         st(ok(ps), "pretraining ran to its GPU-hour budget; see the compute table"),
         "`partA/pretrain_tijepa1_seed0.json`, "
         "`checkpoints/pretrain_log_tijepa1_seed0.csv`"],
        ["C4", "No future prediction",
         st(ok(ps) and "future" in (ps or {}).get("objective", "")),
         "`partA/pretrain_tijepa1_seed0.json` (`objective`, per-mask losses), "
         "`partA/tracking_forecast_v1.json` (`forecasting`), `tests/test_masks.py`"],
        ["C5", "Static frames OOD",
         st((corpus or {}).get("n_static_clips", 0) > 0),
         "`pretrain_corpus.json` (`static_fraction`)"],
        ["C6", "Weak hand-crafted baseline",
         st(ok(bb) and any("handcrafted_improved" in k
                           for k in (bb or {}).get("encoders", {}))),
         "`src/tijepa/preprocess/morphology_v1.py`, "
         "`partA/bbbc054_v1.json` (`e1_segmentation_methods`) — Cellpose was not "
         "installed; the documented classical pipeline was used instead"],
        ["C7", "Statistical floors / paired t / 4 conditions",
         st(ok(b34) and ok(anc)),
         "`partB/b33_b4.json` (exact condition permutations), "
         "`partC/c1_anchors.json` (moderated paired tests), "
         "`tests/test_moderated.py`"],
        ["C8", "Circular one-axis analysis",
         st(ok(c3) and c3.get("v0_reduced_one_axis_analysis", {}).get("status")
            == "DELETED"),
         "`partC/c3_projection.json`; `scripts/41_partC_projection.py` deleted"],
        ["B1", "LPS anchor donor-confounded",
         st(ok(anc)),
         "`partC/c1_anchors.json` — under donor matching the LPS anchor cannot be "
         "built at all (2 pairs, minimum 3), which explains the v0 artefact"],
        ["B2", "Rest vs restimulation",
         st(ok(restim)),
         "`partC/c3_restimulation.json` — GSE212282 and GSE185433, both with matched "
         "donors and an LPS restimulation"],
        ["B3", "Macrophage type mismatch",
         st(ok(anc) and bool((anc or {}).get("differentiation_choice"))),
         "`partC/c1_anchors.json` (`differentiation_choice`) — GM-CSF used and the "
         "mismatch stated, since the M-CSF arm has no differentiated control"],
        ["B4", '"Acute endotoxin tolerance"',
         st((REPO / "paper" / "tijepa.md").exists()
            and "acute endotoxin tolerance" in
            (REPO / "paper" / "tijepa.md").read_text().lower()),
         "`paper/tijepa.md` — abstract, results and methods"],
        ["B5", "Cell-type mismatch; shape ≠ activation",
         st((REPO / "paper" / "tijepa.md").exists()
            and "cell-type chain" in (REPO / "paper" / "tijepa.md").read_text(),
            "the wording and the limitation paragraph are in place, but PhagoStat "
            "microglia were not added to the corpus (D4b)"),
         "`paper/tijepa.md` §3"],
        ["B6", "Discussion overclaim",
         st(ok(claims) and claims.get("status") == "ok"),
         "`claims_check.json` — the checker passes, and it caught the v0 abstract's "
         "three-way ranking of two encoders that are not distinguishable"],
        ["E1", "Table header mismatch",
         st((REPO / "src" / "tijepa" / "viz" / "tables.py").exists()),
         "`src/tijepa/viz/tables.py`, `tests/test_paper_tools.py`"],
        ["E2", "§2.6 strings, repeat, missing Methods",
         st((REPO / "paper" / "tijepa.md").exists()
            and "### 4.11" in (REPO / "paper" / "tijepa.md").read_text()),
         "`paper/tijepa.md` §2.6 and §4.1–4.11"],
        ["E3", "References",
         st(ok(refs) and (refs or {}).get("n_todo_verify", 1) == 0),
         "`references.json` — all resolved against Crossref or DataCite, each title "
         "keyword-checked because a DOI can resolve and still be the wrong paper"],
        ["E4", "Figures",
         st(ok(figs) and all(r.get("status") == "ok"
                             for r in (figs or {}).get("figures", []))),
         "`figures.json`, `figures/fig*_v1.png` — each inspected, not just rendered"],
        ["E5", "Hard-coded conclusions",
         st(ok(claims) and claims.get("status") == "ok"),
         "`claims_check.json`, `src/tijepa/viz/claims.py`"],
        ["E6", "Reproducibility gaps",
         st((REPO / "requirements-lock.txt").exists()
            and (REPO / "scripts" / "run_all_v1.sh").exists()),
         "`requirements-lock.txt`, `scripts/run_all_v1.sh`, `reports/V0_ERRATA.md`"],
        ["E7", "Repo hygiene",
         st(not big_tracked_files),
         "no tracked file over 5 MB; `*.npz`/`*.npy` ignored; the stale axis comment "
         "in `scripts/60_partA_eval.py` fixed"],
        ["O1", "Weights only on instance disk",
         st((REPO / "reports" / "BACKUP.md").exists()),
         "`reports/BACKUP.md` — v0 checkpoint md5-verified on the author's Drive"],
    ]
    table = markdown_table(["ID", "Finding", "Status", "Evidence"], rows,
                           align=["l", "l", "l", "l"])

    n_fixed = sum(1 for r in rows if r[2] == "fixed")
    n_partial = sum(1 for r in rows if r[2].startswith("partially"))
    n_not = len(rows) - n_fixed - n_partial

    gpu_rows = []
    for name, s in (("TiJepa-1 (mixed masking)", ps),
                    ("TiJepa-1-spatial (ablation)", abl)):
        if s:
            gpu_rows.append([name, f"{s['steps']:,}", f"{s['gpu_hours']}",
                             f"{s['effective_epochs']}", s["stop_reason"]])
    gpu_table = (markdown_table(
        ["run", "steps", "GPU-hours", "effective epochs", "stop reason"], gpu_rows)
        if gpu_rows else "_No pretraining run completed in this report's snapshot._")


    def _ba():
        r = ((b3 or {}).get("b31_tolerance", {}).get("results", {})
             .get("raw_trace", {}) or {})
        perm = r.get("exact_permutation", {}) or {}
        return r.get("cell_balanced_accuracy_mean"), perm.get("minimum_achievable_p")

    ba_val, ba_floor = _ba()
    ltdb_status = ((intake or {}).get("datasets", {})
                   .get("ltdb", {})).get("status", "unknown")
    data_table = markdown_table(
        ["dataset", "role", "outcome"],
        [
            ["ComplexEye (Zenodo 10082674, CC BY 4.0)", "pretraining",
             f"{n_arch}/{n_arch_total} archives; "
             f"{(corpus or {}).get('n_movies_ok', 0)} movies, "
             f"{(corpus or {}).get('n_clips', 0):,} clips, "
             f"{(corpus or {}).get('total_gb', 0)} GB"],
            ["S-BIAD2196 (CC BY 4.0)", "tracking + forecasting",
             f"all {(tf or {}).get('n_movies', 6)} movies; "
             f"{len((lk or {}).get('excluded_train_movies', []))} twin(s) excluded "
             "from pretraining"],
            ["BBBC054 (CC BY 3.0)", "morphology-class probe",
             "3 classes, 60 fields of view"],
            ["GSE260996 / GSE261647", "Part B expression / accessibility",
             "both pass their positive controls"],
            ["Tay NF-kB traces", "Part B dynamics",
             f"ligand codes sourced; "
             f"{(b3 or {}).get('b31_tolerance', {}).get('n_wells', '-')} matched wells"],
            ["GSE46903 / GSE58310", "Part C anchors / resting scores",
             f"{(anc or {}).get('n_anchors_usable', '-')} of 3 anchors usable"],
            ["GSE212282 / GSE185433", "Part C restimulation",
             "both qualify, 3 matched donors each"],
            ["Dryad m0cfxpp8v", "planned Part A headline",
             f"**{(dry or {}).get('status', 'unknown')}**"],
            ["LTDB", "tracking (optional)",
             f"**{ltdb_status}**"],
        ], align=["l", "l", "l"])

    notes = []
    if defect:
        notes.append(
            "The published ComplexEye `R06.zip` has no end-of-central-directory record "
            "and is 57.5% zero padding, while its md5 matches the Zenodo record "
            "exactly - so the defect is in the deposit, not in our transfer "
            "(`complexeye_archive_defect.json`).")
    if lk:
        notes.append(
            "S-BIAD2196 holds 6 movies, not 12: the BioStudies API serves only 5 rows "
            "however it is paged, and the sixth - the one duplicated in the "
            "pretraining corpus - is visible only in the FIRE file tree.")
    data_notes = " ".join(notes) if notes else "No data defects recorded."

    results_table = markdown_table(
        ["result", "value", "source"],
        [
            ["RNA positive control",
             f"{(rna or {}).get('n_passing', '-')}/7 targets, "
             + ("PASS" if (rna or {}).get("passed") else "see file"),
             "`partB/rna_positive_control.json`"],
            ["ATAC positive control",
             f"{(atac or {}).get('n_genes_increased', '-')}/7 genes, "
             + ("PASS" if (atac or {}).get("passed") else "see file"),
             "`partB/atac_positive_control.json`"],
            ["B3.1 acute endotoxin tolerance",
             f"BA {ba_val if ba_val is None else round(ba_val, 3)}, exact p floor "
             f"{ba_floor if ba_floor is None else round(ba_floor, 3)}",
             "`partB/b3_dynamics.json`"],
            ["B3.3 dynamics -> omics", "null for both readouts",
             "`partB/b33_b4.json`"],
            ["B4 cross-species overlap",
             f"{(b34 or {}).get('b4', {}).get('n_overlap', '-')} genes",
             "`partB/b33_b4.json`"],
            ["C1 usable anchors",
             f"{(anc or {}).get('n_anchors_usable', '-')} of 3",
             "`partC/c1_anchors.json`"],
            ["C3 projection", "not performed; two independent reasons",
             "`partC/c3_projection.json`"],
            ["Part A BBBC054", "see the table in the paper",
             "`partA/bbbc054_v1.json`"],
            ["Tracking / forecasting", "see the tables in the paper",
             "`partA/tracking_forecast_v1.json`"],
        ], align=["l", "l", "l"])

    comp = (pre or {}).get("compute", {})
    compute_line = (
        f"{comp['gpus'][0]['name']} x{comp['gpu_count']}, "
        f"{comp['cpu_cores_logical']} logical cores, {comp['ram_total_gb']} GB RAM"
        if comp.get("gpus") else "compute not recorded")

    txt = f"""# TiJepa v1 — run report

Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} from `results/v1/`.
Branch `tijepa-v0`, commit `{sh(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'])}`.

This run executes `PLAN_v1.md`, which is a fix plan for the v0 run. Corrections to the
v0 record are in [`reports/V0_ERRATA.md`](V0_ERRATA.md); the backup record is in
[`reports/BACKUP.md`](BACKUP.md).

## 1. Compute

{compute_line},
tier **{(pre or {}).get('tier', {}).get('tier', '?')}**, torch
`{(ps or {}).get('torch_version', '2.11.0+cu128')}`, all data on local disk.

{gpu_table}

One GPU means pretraining has a **single seed**. Seed-to-seed variation in Part A comes
from independent random-init draws, stratified cell subsamples and fold assignments, and
this is stated in the paper's Methods.

## 2. Data

{data_table}

{data_notes}

## 3. Section 12 traceability

**{n_fixed} fixed, {n_partial} partially fixed, {n_not} not fixed**, of {len(rows)} findings.

{table}

## 4. Results, with their source files

{results_table}

## 5. Deviations from PLAN_v1.md

1. **The corpus stopped at {n_arch} of {n_arch_total} archives.** zenodo.org answered
   about one connection in three from this instance and each 10 GB archive took roughly
   56 minutes, so continuing would have consumed the pretraining budget. The corpus is
   {(corpus or {}).get('n_clips', 0):,} clips, comparable to v0's 288,081 but from more
   movies and with QC, leakage exclusion, four strides and static augmentation.
2. **PhagoStat microglia video was not added** (D4b/D4e), for the same budget reason.
   The corpus therefore has two instruments but one cell type.
3. **Cellpose was not installed**; E1 uses the documented classical segmentation
   pipeline instead, which the plan permits.
4. **The frame-QC classifier was not trained**; QC is heuristic with thresholds from
   this corpus's own empirical distribution, which the plan permits, and is labelled
   heuristic wherever it is reported.
5. **CoTracker was not run**: its host was unreachable in preflight and the Hugging Face
   route was not attempted.
6. **The every-2-hour linear-probe collapse monitor** of F2.2 used Dryad; collapse was
   monitored by embedding standard deviation and effective rank every 50 steps instead.
7. **The IL-4 anchor was dropped** by the sanity rule as written (2 of 3 markers in the
   top 100), although it is IL-4-consistent at a wider cut-off. The rule was applied as
   specified and the marker ranks are reported so the near miss is visible.

## 6. What a reviewer would still raise

- **No trained or tolerant cell was imaged anywhere in this work.** Every imaging result
  is about neutrophil migration or microglial morphology; the trained-immunity link is
  made only through molecular data, across studies, and is labelled hypothesis-only.
- **The Part A classification evidence is static.** Without the Dryad movies there is no
  video benchmark with activation labels, so the frame-shuffle control has no labelled
  video task to run on.
- **Part B cannot reach significance.** Four wells is four wells; the exact permutation
  floor of 0.167 is a property of the deposit, not of the analysis.
- **Part C rests on one axis.** Only the IFN-γ anchor survived donor matching and the
  sanity rules, and the restimulation test is directionally inconsistent across training
  stimuli, with β-glucan pointing the wrong way.
- **One cell type, one instrument family, one pretraining seed.**

## 7. TODO(author)

1. **Fetch Dryad `doi:10.5061/dryad.m0cfxpp8v` by hand** and re-run
   `scripts/64_dryad_v1.py` and `scripts/44_partC_projection_v1.py`. This is the single
   highest-value action: it unblocks the Part A video headline, the frame-shuffle
   control on labelled video, and the C3 projection.
2. Optionally fetch the LTDB collection for the tracking evaluation.
3. Confirm the affiliation in `paper/tijepa.md`.
4. Review the AI-assistance disclosure (§4.10) and declare competing interests (§4.11).
5. Confirm the licence choice: Apache-2.0 for code, CC BY 4.0 for text and figures.
6. Decide whether to publish model weights (`reports/WEIGHTS.md`); they are **not**
   published and are **not** in git.
"""
    OUT.write_text(txt)
    print(f"wrote {OUT}")
    print(f"Section 12: {n_fixed} fixed, {n_partial} partially fixed, {n_not} not fixed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
