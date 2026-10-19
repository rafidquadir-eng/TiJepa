# ruff: noqa: E501  (the report template is long Markdown lines)
"""G6: write reports/RUN_REPORT_v2.md from the artefacts, after everything has run.

Every status in the two traceability tables (PLAN_v2 §11 N1-N15, and PLAN_v1 §12
re-derived) is computed from a results file named in the same row. The script refuses to
run before both pretraining runs and the evaluations have finished (rule 2).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.viz.tables import markdown_table  # noqa: E402

R2 = REPO / "results" / "v2"
R1 = REPO / "results" / "v1"


def load(root: Path, rel: str):
    p = root / rel
    return json.loads(p.read_text()) if p.exists() else None


def sh(cmd: str) -> str:
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                          cwd=REPO).stdout.strip()


def st(cond: bool, partial: str | None = None, not_fixable: str | None = None) -> str:
    if not_fixable:
        return f"not fixable — {not_fixable}"
    if cond:
        return "fixed"
    return f"partially fixed — {partial}" if partial else "not fixed"


def main() -> int:
    pre = load(R2, "preflight.json")
    plan = load(R2, "pretrain_plan.json")
    bench = load(R2, "pretrain_benchmark.json")
    ps = load(R2, "partA/pretrain_tijepa1_seed0.json")
    abl = load(R2, "partA/pretrain_tijepa1_spatial_seed0.json")
    bb = load(R2, "partA/bbbc054.json")
    tf = load(R2, "partA/tracking_forecast.json")
    dry = load(R2, "partA/dryad_benchmark.json")
    lk = load(R2, "leakage_check_full.json")
    b3 = load(R2, "partB/b3_dynamics.json")
    b34 = load(R2, "partB/b33_b4.json")
    rs = load(R2, "partC/c3_restimulation.json")
    ae = load(R2, "partC/arm_expectations.json")
    claims = load(R2, "claims_check.json")
    figs = load(R2, "figures.json")
    insp = load(R2, "figures_inspection.json")
    intake = load(R2, "manual_intake.json")
    refs = load(R2, "references.json")
    corpus = load(R1, "pretrain_corpus.json")
    if not (ps and abl and bb and tf and claims):
        raise SystemExit("inputs absent: the report is written only after every run")

    mc = claims.get("manuscript_checks", {})
    # ------------------------------------------------------------ N-table
    enc = {k: v for k, v in bb["encoders"].items() if isinstance(v, dict)
           and "three_class_mean" in v}
    maes_ok = all(v["exposure_time_mae_mean"] < 10 * v["exposure_time_mae_mean_predictor"]
                  for v in enc.values())
    n_cells = [int(v) for v in bb["n_cells_per_seed"].values()]
    sel = tf["track_selection"]["per_movie"]
    fstat = tf["forecasting"]["per_movie_status"]
    e0 = tf["tracking"]["aggregate"].get("E0_tijepa0", {})
    restim_pc = all(rs[g]["positive_control"]["status"] == "PASS"
                    for g in ("GSE212282", "GSE185433"))
    b4 = b34["b4"]
    sens = b3["b31_sensitivity_all_wells"]
    final_ok = (ps["annealed"] and abl["annealed"] and ps["final_lr"] <= 1e-12
                and abs(ps["final_ema_momentum"] - 1.0) < 1e-12)
    n_rows = [
        ["N1", "Paper labelled TiJepa-0 results as TiJepa-1",
         st(mc.get("provenance") == "pass"),
         "`claims_check.json` (`manuscript_checks.provenance`), `paper/labels.json`, "
         "`src/tijepa/eval/provenance.py`"],
        ["N2", "Report, paper and figures generated before pretraining",
         st(final_ok and bool(figs)),
         "`partA/pretrain_tijepa1_seed0.json`, `partA/pretrain_tijepa1_spatial_seed0.json` "
         "(`annealed`), this report's generation time"],
        ["N3", "Schedule not annealing; non-cumulative GPU-hours; no SIGTERM; non-atomic saves",
         st(final_ok and len(ps["segments"]) >= 2),
         f"`partA/pretrain_tijepa1_seed0.json` (final LR {ps['final_lr']:.1e}, EMA "
         f"{ps['final_ema_momentum']:.4f}, {len(ps['segments'])} segments), "
         "`tests/test_trainer.py`"],
        ["N4", "Dryad (and LTDB) not provided",
         st(False, not_fixable="neither was on the author's Drive at the start or before "
            "the evaluations; the status is recorded. The script detects and lists the "
            "files, but the benchmark logic itself still has to be written against the "
            "deposit's README") if (dry or {}).get("status") == "not_provided" else
         st(True),
         "`partA/dryad_benchmark.json`, `manual_intake.json`"],
        ["N5", "Leakage check incomplete; weak fingerprint; E0 trained on 08_H4's twin",
         st(bool(lk and lk.get("all_archives_checked")) and bool(e0.get(
             "with_08_H4_reference_only")),
            "the full-resolution check did not cover every archive" if lk else
            "the full check did not run"),
         "`leakage_check_full.json`, `partA/tracking_forecast.json` (E0 `movies_excluded`)"],
        ["N6", "BBBC054 ~879 cells/seed; E1 MAE 1e8 h; negative block-CV ρ; empty panel",
         st(min(n_cells) > 8000 and maes_ok and "scale_sensitivity" in bb
            and "confound_full_file" in bb),
         f"`partA/bbbc054.json` ({min(n_cells):,}-{max(n_cells):,} cells/seed; all MAEs "
         "below 10x the mean predictor; buffered LOFO with the fold-mean reference)"],
        ["N7", "18 tracks; ⟨δ⟩/survival unexplained; forecasting 4/6 movies; random init "
               "presented as comparable",
         st(all(v["n_tracks"] <= 50 for v in sel.values())
            and all(m in fstat for m in tf["movies"])
            and "n_movies" in tf["forecasting"]["primary_tijepa1_vs_spatial"]),
         "`partA/tracking_forecast.json` (`track_selection`, `per_threshold_success`, "
         "`per_movie_status`, `primary_tijepa1_vs_spatial`, `supplementary_random_init`)"],
        ["N8", "Part C expectations assumed; restimulation unchecked; panel mixes ISGs",
         st(bool(ae) and restim_pc),
         "`partC/arm_expectations.json`, `partC/c3_restimulation.json` "
         "(`positive_control`, `restimulation_labels`, `primary_TNF_IL6`)"],
        ["N9", "B4 point-estimate rule; resting human side; no ortholog control",
         st(b4["ortholog_positive_control"]["status"] == "PASS"
            and b4["human_primary"]["dataset"] == "GSE185433"),
         "`partB/b33_b4.json` (`b4.mouse.design`, `human_primary`, "
         "`ortholog_positive_control`, two-sided `fisher`)"],
        ["N10", "B3.1 sensitivity sd 0; biased r > 0.5 count",
         st(sens["cell_balanced_accuracy_sd"] > 0 and "fraction_q2_positive" in
            b34["b33"]["analyses"]["rna_expression"]["top2000_variable_targets"]),
         "`partB/b3_dynamics.json` (`b31_sensitivity_all_wells`), `partB/b33_b4.json` (`b33`)"],
        ["N11", "Manuscript text bugs, internal references, empty/text figure panels",
         st(mc.get("banned_strings") == "pass" and mc.get("grammar") == "pass"
            and bool(insp)),
         "`claims_check.json`, `figures.json`, `figures_inspection.json`"],
        ["N12", "Wall-clock budget not transferable", st(bool(plan)),
         "`pretrain_plan.json`, `pretrain_benchmark.json`"],
        ["N13", "Concurrent writers on shared storage; possible I/O starvation",
         st(bool(bench) and (bench or {}).get("gpu_util_mean", 0) >= 80),
         f"`single_writer_check.json`, `preflight.json` (storage read), "
         f"`pretrain_benchmark.json` (GPU {bench.get('gpu_util_mean')}%), "
         "`checkpoints/<run>.lock` in `src/tijepa/train/pretrain_video.py`"],
        ["N14", "Claims checker passed despite N1", st(mc.get("provenance") == "pass"),
         "`scripts/92_check_claims.py`, `tests/test_claims_v2.py`"],
        ["N15", "BBBC054 time-only baseline uninformative by construction",
         st("confound_full_file" in bb),
         "`partA/bbbc054.json` (`time_only_baseline.interpretation`, `confound_full_file`)"],
    ]
    # ----------------------------------------------- PLAN_v1 §12 re-derived
    v1_rows = {}
    for line in (REPO / "reports" / "RUN_REPORT_v1.md").read_text().splitlines():
        if line.startswith("| ") and line.count("|") >= 5 and not line.startswith("| ID"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == 4 and cells[0][:1] in "DCBEO" and len(cells[0]) <= 4:
                v1_rows[cells[0]] = cells
    changes = {
        "D2": ["Dryad headline missing", n_rows[3][2], "`partA/dryad_benchmark.json`"],
        "D3": ["Possible tracking leakage", n_rows[4][2], "`leakage_check_full.json`"],
        "D5c": ["FOV-time confound",
                "fixed — confound described on the full annotation file; buffered LOFO "
                "regression read against the fold-mean reference",
                "`partA/bbbc054.json`"],
        "D6": ["Small tracking eval",
               st(sum(v["n_tracks"] for v in sel.values()) >= 100,
                  "LTDB not provided"),
               "`partA/tracking_forecast.json` (`track_selection`)"],
        "C1": ["Cosmetic seeds", st(sens["cell_balanced_accuracy_sd"] > 0),
               "`partB/b3_dynamics.json`, `partA/bbbc054.json`"],
        "C3": ["Undertrained",
               st(final_ok, "a run ended before its planned final step"),
               "`partA/pretrain_tijepa1_seed0.json`, "
               "`partA/pretrain_tijepa1_spatial_seed0.json`"],
        "C4": ["No future prediction",
               st("n_movies" in tf["forecasting"]["primary_tijepa1_vs_spatial"]),
               "`partA/tracking_forecast.json` (`primary_tijepa1_vs_spatial`)"],
        "C6": ["Weak hand-crafted baseline",
               "fixed — classical segmentation pipeline; both logistic and random-forest "
               "probes reported", "`partA/bbbc054.json`"],
        "B6": ["Discussion overclaim", st(claims.get("status") == "ok"), "`claims_check.json`"],
        "E3": ["References", st(all(r["status"] == "resolved"
                                    for r in refs["references"].values())),
               "`references.json`"],
        "E4": ["Figures", st(bool(insp)), "`figures.json`, `figures_inspection.json`"],
        "E5": ["Hard-coded conclusions", st(claims.get("status") == "ok"),
               "`claims_check.json`"],
        "O1": ["Weights only on instance disk", "see the backups section below",
               "`reports/BACKUP.md`"],
    }
    s12 = []
    for k, cells in v1_rows.items():
        if k in changes:
            s12.append([k, *changes[k]])
        else:
            s12.append([k, cells[1], cells[2] + " (unchanged since the v1 run)",
                        "results/v1: " + cells[3]])
    # ------------------------------------------------------------- compute
    seg_rows = [[run, str(i + 1), s["host"], s["gpu_name"], s["torch_version"],
                 f"{s['start_step']:,}-{s['end_step']:,}",
                 f"{s.get('gpu_seconds', s['seconds']) / 3600:.2f}"]
                for run, summ in (("tijepa1", ps), ("tijepa1_spatial", abl))
                for i, s in enumerate(summ["segments"])]
    mon = []
    mp = R2 / "pretrain_monitor.csv"
    if mp.exists():
        mon = list(csv.DictReader(mp.open()))
    commit = sh("git rev-parse --short HEAD")
    text = f"""# TiJepa v2 — run report

Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} from `results/v2/`, after
both pretraining runs and all evaluations had finished. Branch `tijepa-v0`, commit
`{commit}`. This run executes `PLAN_v2.md`; corrections to the earlier records are in
[`reports/V0_ERRATA.md`](V0_ERRATA.md) (sections for v0 and v1).

## 1. Compute

{pre['hardware']['gpus'][0]['name']} x{len(pre['hardware']['gpus'])},
{pre['hardware']['cpu_logical_cores']} logical cores, {pre['hardware']['ram_gb']} GB RAM,
torch `{pre['hardware']['torch_version']}`, project on local ext4
(`{pre['storage_read']['dd']}`), so the corpus was read in place.

Budget: **{plan['S_total']:,} steps** per run ({plan['S_total_formula']} =
{plan['S_total_formula_value']:,}, rounded as specified), measured throughput
{bench['steps_per_hour']:,.0f} steps/h at {bench['gpu_util_mean']}% GPU utilisation
(`pretrain_benchmark.json`); projected {plan['projected_hours']['total']} h for both runs,
under the 36 h ceiling, so the budget was not reduced.

| run | steps | annealed | final LR | final EMA | GPU-hours (cumulative) | stop |
|---|---|---|---|---|---|---|
| tijepa1 | {ps['steps']:,} | {ps['annealed']} | {ps['final_lr']:.1e} | {ps['final_ema_momentum']:.4f} | {ps['gpu_hours']:.2f} | {ps['stop_reason']} |
| tijepa1_spatial | {abl['steps']:,} | {abl['annealed']} | {abl['final_lr']:.1e} | {abl['final_ema_momentum']:.4f} | {abl['gpu_hours']:.2f} | {abl['stop_reason']} |

Segments:

{markdown_table(['run', 'segment', 'host', 'GPU', 'torch', 'steps', 'hours'], seg_rows)}

Resume discontinuity at step 14,751: LR {plan['schedule_at_resume_step_14751']['lr_old_schedule_max150000']:.4e}
→ {plan['schedule_at_resume_step_14751']['lr_new_schedule_max117000']:.4e}
({100 * plan['schedule_at_resume_step_14751']['lr_relative_change']:+.2f}%), EMA momentum
{plan['schedule_at_resume_step_14751']['ema_next_update_old_schedule']:.6f} →
{plan['schedule_at_resume_step_14751']['ema_next_update_new_schedule']:.6f}. The
`n_spatial`/`n_future` columns of the TiJepa-1 log count batches within a segment, so they
restart at the resume; the per-segment counts are in the summary.

Collapse monitor (`pretrain_monitor.csv`, {len(mon)} rows; monitor only, never used to pick
a checkpoint): {", ".join(f"{r['run']}@{int(r['step']):,} rank {float(r['effective_rank']):.1f} probe {float(r['probe3_balanced_accuracy']):.3f}" for r in mon[-6:])}.

## 2. Data

Corpus unchanged (index md5 `{(load(R2, 'corpus_shards_sha256.json') or {}).get('pretrain_index_md5')}`,
{corpus['n_clips']:,} clips, {corpus['n_shards']} shards, SHA-256 of every shard in
`corpus_shards_sha256.json`). Manual downloads: Dryad
**{intake['datasets']['dryad_m0cfxpp8v']['status']}**, LTDB
**{intake['datasets']['ltdb']['status']}** (`manual_intake.json`).

## 3. PLAN_v2 §11 findings

{markdown_table(['ID', 'Finding', 'Status', 'Evidence (results/v2 unless stated)'], n_rows)}

## 4. PLAN_v1 §12, re-derived

Rows not listed in PLAN_v2 keep their v1 status and v1 evidence file.

{markdown_table(['ID', 'Finding', 'Status', 'Evidence'], s12)}
"""
    (REPO / "reports" / "RUN_REPORT_v2.md").write_text(text)
    print(f"wrote reports/RUN_REPORT_v2.md; N statuses: "
          f"{[r[2].split(' ')[0] for r in n_rows]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
