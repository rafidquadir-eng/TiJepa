# TiJepa — Plan v2: finish pretraining on the dedicated GPU, fix what the v1 review found

> **For the executing agent.** This plan continues the v1 run on a new instance with a
> dedicated GPU that shares storage and Google Drive with the previous instance. Read this
> file end to end, then `RESUME.md`, `PLAN_v1.md` (still in force), `PLAN.md` §1 (still in
> force), `reports/RUN_REPORT_v1.md` and `reports/V0_ERRATA.md`. Execute phases G0 → G8 in
> order. Section 11 lists every new review finding (N1–N15) with its fix and acceptance
> check. **The run is finished only when every N-row and every still-open row of PLAN_v1 §12
> has a status backed by an evidence file.**
>
> Branch: continue on `tijepa-v0` (PR #1). `git pull` first. Never force-push, never push
> to `main`, never rewrite history.

---

## 0. Where things stand (review of commit `94576a3`)

**Done well in v1 and kept:** the RNA/ATAC split with data-type guards and positive controls;
ligand codes sourced from the deposit's own code; the three-class BBBC054 discovery
(`ramified`, 11,143 annotations v0 dropped); verified pixel sizes; the leakage check that
found a real duplicate (`R08:H4` ↔ `08_H4`); honest small-n statistics in Part B; donor-
matched Part C anchors; the restimulation datasets GSE212282/GSE185433; reference resolution
against Crossref/DataCite; V0 errata; 52 passing tests.

**Not done / wrong:**
- **Pretraining is 1.51 of 12 budgeted GPU-hours in** (step 14,751, `checkpoints/tijepa1_T1_seed0.pt`).
  The spatial-only ablation has not started. Every Part A number in the repo comes from the
  **v0** checkpoint.
- The paper, figures and run report were **generated before pretraining finished** and contain
  a critical mislabel (N1) and false statuses (N2).
- Dryad and LTDB were **not provided** (nothing under `TiJepa/manual_downloads/` on Drive).

---

## 1. Rules (in addition to PLAN.md §1 and PLAN_v1.md §1)

1. **Provenance guard.** Every encoder result records the checkpoint file, run name, step and
   SHA-256 it came from. Paper and figure labels are derived from that record, never from a
   hand-written map. A results file whose checkpoint does not match its label is a hard error
   (this is how N1 must never recur).
2. **Nothing is generated before its inputs exist.** Paper, figures and run report are
   regenerated only after pretraining and all evaluations have finished. If an analysis is
   absent, its section says so in one sentence; no placeholder sections.
3. **Single writer on shared storage.** Exactly one process may write a given checkpoint. The
   trainer takes an exclusive lock file (`checkpoints/<run>.lock`, holding hostname, PID,
   start time) and refuses to start if a live lock exists. Checkpoints are written atomically
   (`tmp` file + `fsync` + `os.replace`).
4. **Budget is defined in steps, not wall-clock.** The LR and EMA schedules must anneal fully
   by the last step of every run.
5. **No internal process language in the manuscript.** No review-finding IDs (D1, N3…), plan
   section numbers, "v0/v1/previous version" narrative, file paths into `results/`, or status
   strings in the paper body or figure titles. Version history belongs in
   `reports/V0_ERRATA.md` and one sentence in Methods.
6. **Expected biological directions must be sourced.** Any "as predicted" or "expected
   direction" in Part C cites the publication and the specific result it comes from; otherwise
   no expectation is stated.

---

## 2. Author prerequisites (before or during the run)

1. **Dryad (blocking for the Part A headline and Part C3).** Download
   `https://datadryad.org/dataset/doi:10.5061/dryad.m0cfxpp8v` ("Download dataset") in a
   browser and upload the zip to Drive at `TiJepa/manual_downloads/dryad_m0cfxpp8v/`.
   The agent re-checks Drive at G0 and again before G4; the evaluation code is already in
   place (`scripts/64_dryad_v1.py`).
2. **LTDB (optional):** figshare collection `10.6084/m9.figshare.c.3827890` →
   `TiJepa/manual_downloads/ltdb/`.
3. **Old instance:** make sure no trainer or agent is still running there (it shares
   storage). The agent also checks (G0.3).

---

## 3. Phase G0 — Environment, integrity, safety (before touching the GPU)

### G0.1 Repository
`git pull` on `tijepa-v0`; confirm HEAD includes `94576a3`; working tree clean.

### G0.2 Hardware and environment → `results/v2/preflight.json`
GPU model/count/memory, CUDA driver, CPU cores, RAM, free **local** disk, and the filesystem
type of the project directory (`df -T`, `mount`). Reuse the environment pinned in
`requirements-lock.txt` (torch `2.11.0+cu128`). Verify `torch.cuda.is_available()` and a
bf16 matmul on the new GPU. If this torch build cannot run on the new GPU architecture,
install the closest version that does, record it as a deviation, and note that pretraining
spans two torch versions (both recorded in the training log).

### G0.3 Single-writer check
- `pgrep -af "pretrain|51_pretrain_v1"` on this instance; inspect `checkpoints/*.lock`.
- Record mtime and size of `checkpoints/tijepa1_T1_seed0.pt` and of
  `checkpoints/pretrain_log_tijepa1_seed0.csv`; wait 5 minutes; record again. If either
  changed, **another process is writing**: stop, write `reports/BLOCKED.md`, and do not start
  training.

### G0.4 Integrity of the inputs
- Checkpoint SHA-256 must equal `2774e60461d2c24b9b8c92b71d211c1cfa4b4e6241010d1c61888f6b7d054b22`
  (size 376,538,701 bytes); load it and confirm `step == 14751` and that it holds `model`,
  `optim`, `config`.
- Corpus: `md5(data/processed/v1/pretrain_index.csv) == 43b19b36849e88d29cefa484066e35a1`;
  43 shards present. Compute SHA-256 of every shard once and commit the list as
  `results/v2/corpus_shards_sha256.json` (small) so future moves can be verified.
- Any mismatch: stop and write `reports/BLOCKED.md`. Do **not** rebuild the corpus
  (`RESUME.md` explains why).

### G0.5 Back up the partial checkpoint first
`rclone copy` `checkpoints/tijepa1_T1_seed0.pt` and its log CSV to
`<remote>:TiJepa/checkpoints_v1_partial/` and verify md5 (procedure as in `reports/BACKUP.md`).
Append to `reports/BACKUP.md`. No credentials in any file or log.

### G0.6 Storage throughput
Measure sequential read throughput of one shard from the project directory. If the project
directory is a network filesystem and throughput is below ~1 GB/s, or if GPU utilisation
during G1.3 stays below 80%, copy `data/processed/v1/` to local NVMe, verify against
`corpus_shards_sha256.json`, and point the trainer at the local copy. Record the decision.

### G0.7 Manual downloads
Re-check `<remote>:TiJepa/manual_downloads/`; ingest whatever is present (PLAN_v1 F0.3).
Update `results/v1/manual_intake.json` → write `results/v2/manual_intake.json`.

---

## 4. Phase G1 — Fix the trainer, then resume (fixes N3, N12, N13)

### G1.1 Trainer fixes (`src/tijepa/train/pretrain_video.py`), each unit-tested
1. **Graceful stop:** SIGTERM/SIGINT handler that finishes the current step, saves a
   checkpoint atomically, writes the summary, releases the lock, and exits 0.
2. **Atomic saves** (rule 3) and **lock file** (rule 3).
3. **Cumulative accounting:** the checkpoint stores `gpu_seconds_total`,
   `wall_seconds_total` and a list of `segments` (`host`, `gpu_name`, `torch_version`,
   `start_step`, `end_step`, `seconds`). The summary reports totals and per-segment values.
   Seed the existing checkpoint's history with the known first segment
   (A100-SXM4-40GB, 0 → 14,751, 1.51 h, torch 2.11.0+cu128).
4. **Schedule tied to the planned total:** `max_steps` is the planned final step, so cosine LR
   reaches its floor and EMA momentum reaches 1.0 exactly at the end. `max_hours` becomes a
   safety cap only. If the cap is hit, the run stops and the report states it ended
   unannealed.
5. Log `p_future`, the spatial and future loss separately, LR and EMA momentum (already
   partly logged) every 250 steps.

### G1.2 Decide the step budget (record in `results/v2/pretrain_plan.json`)
- The v1 budget was **12 A100-40GB GPU-hours**. Throughput observed there was
  14,751 steps / 1.51 h ≈ **9,770 steps/h** (batch 128). Convert the budget to steps:
  **S_total = 14,751 + 9,770 × 10.49 ≈ 117,000 steps** (≈ 15.0 M clips, ≈ 53.6 epochs over
  279,647 clips). This is hardware-independent and reproducible.
- Benchmark the new GPU: load a *copy* of the checkpoint into a scratch directory, run 500
  steps without saving, measure steps/h and GPU utilisation. Delete the scratch copy.
- Projected wall-clock = (main: 117,000 − 14,751 steps) + (ablation: 117,000 steps) at the
  measured rate. If the projection exceeds **36 h**, reduce S_total for **both** runs to fit
  36 h (never below 80,000), before resuming, and record the decision. Otherwise keep 117,000.
- Record the LR and EMA momentum at step 14,751 under the old schedule (max_steps 150,000)
  and the new one; the discontinuity is expected to be ~1–2% and is reported, not hidden.

### G1.3 Run (one GPU: sequential; more GPUs: parallel)
| GPUs | Plan |
|---|---|
| 1 | Resume `tijepa1` (p_future 0.5) to S_total; then `tijepa1_spatial` (p_future 0.0) from scratch to the same S_total, same schedule, same seed, same frozen shard list |
| 2 | Resume `tijepa1` on GPU 0; `tijepa1_spatial` on GPU 1 concurrently |
| ≥ 4 | As 2, plus `tijepa1` seeds 1 and 2 from scratch (S_total each) on GPUs 2–3 |

- `--num-workers` = physical cores − 2. Run inside `nohup`/`tmux`; log to `logs/`.
- Check `nvidia-smi` utilisation 10 minutes in; if < 80%, apply G0.6.
- **Monitoring every ~2 h** (does not change training): embedding std, effective rank, the
  two losses; a fast 3-class BBBC054 linear probe on a fixed 1,000-cell subset from the latest
  checkpoint (collapse/regression monitor only; never used to pick a checkpoint). Write to
  `results/v2/pretrain_monitor.csv`.
- **Backups:** copy each run's checkpoint to Drive every ~3 h and at completion
  (`<remote>:TiJepa/checkpoints_v1/`), verified by md5.
- While training runs, do G2, G3 and the non-GPU parts of G4–G6 (CPU-only work, or use the
  GPU only if it is idle).

---

## 5. Phase G2 — Fix evaluation code before any evaluation runs

### G2.1 Provenance guard (fixes N1, N14)
- Delete the label map entry `"E4_tijepa0_only_available": "E4 TiJepa-1"` in
  `scripts/91_make_paper_v1.py` and replace the whole map with labels derived from each
  result's recorded checkpoint (`tijepa0_T1_seed0.pt` → "TiJepa-0"; `tijepa1_*` →
  "TiJepa-1"; `tijepa1_spatial_*` → "TiJepa-1 (spatial-only)"; random init → "random init").
- Every evaluation script records `checkpoint: {file, run_name, step, sha256}` per encoder.
- `scripts/92_check_claims.py` gains (a) a provenance check, (b) a banned-strings check
  (`PLAN_v`, `review finding`, `D5b`, `F3.`, `NOT USABLE`, `not run in this pass`,
  `blocking requirements recorded`, `results/v1/`, `v0`, `previous version` in the paper body,
  and lowercase-mangled identifiers such as ` e5 `, ` bbbc054 `, ` s-biad2196 `), and
  (c) a grammar check for dangling comparatives (`higher than (`, `lower than (`).
  Tests for all three.

### G2.2 BBBC054 (fixes N6, N15)
1. Run with the full sample: `--n-cells 9000` per seed (the v1 numbers came from ~879/seed).
2. **E1 numeric blow-up:** exposure-time MAE of 1.0×10⁸ h means unscaled or outlying
   features reached the regressor. Use `RobustScaler`, clip features to their 0.1–99.9%
   quantiles fitted on the training fold, drop non-finite features, and assert that every
   reported MAE is finite and below 10× the mean predictor.
3. **Exposure-time regression:** leave-contiguous-block CV makes every encoder's ρ strongly
   negative because predictions for a held-out block regress toward the training mean, which
   lies on the other side of the time axis. Replace it with **leave-one-FOV-out with a ±3-FOV
   buffer** removed from training (each test FOV is bracketed by training data on both sides),
   and also report the per-block within-block ρ. Keep it labelled exploratory.
4. **Confound description, honestly:** in the class-balanced per-FOV sample the time-only
   baseline is at chance **by construction**. Say so. Separately describe the confound in the
   full annotation file: class proportions per exposure time and Cramér's V between class and
   time. Drop the sentence "exposure time by itself carries no morphology information".
5. Run `--scale-sensitivity` so the figure has no empty panel.
6. Encoders: E1, E2, E3, E0 TiJepa-0, E4 TiJepa-1, E4s TiJepa-1 spatial-only, E6 random
   init (3 draws). (E5 frame-shuffle stays N/A on static images.)

### G2.3 Tracking and forecasting (fixes N5c, N7)
1. **Tracks:** evaluate up to 50 ground-truth tracks per movie (all if fewer), selected by a
   seeded rule recorded in the results; report `n_tracks` per movie. (v1 evaluated 18 tracks in
   total; v0 had 125.)
2. **Metrics:** report per-threshold curves (1, 2, 4, 8, 16 µm) and define survival in the
   text (first frame with error > 8 µm). Add a short consistency note when ⟨δ⟩ and survival
   rank methods differently.
3. **Leakage for E0:** TiJepa-0's corpus contained `R08:H4`, the twin of `08_H4`. Exclude
   `08_H4` from every comparison involving E0 (report both with and without).
4. **Forecasting:** run on all 6 movies; any movie that cannot be evaluated gets a stated
   reason in the results file. Primary comparison: **TiJepa-1 vs TiJepa-1-spatial**, ratio of
   L1(predictor) to L1(copy-last) per movie, exact sign-flip test over movies. E0 is a
   reference row. Random-init ratios move to a supplementary table with one sentence on why
   they are not comparable (an untrained predictor).
5. If Dryad is provided, add Dryad movies to forecasting.

### G2.4 Dryad (only if provided; fixes N4 when possible)
Run `scripts/64_dryad_v1.py` with E0, E1–E3, E4, E4s, E5, E6, leave-one-batch-out, confound
probes (batch, magnification), as specified in PLAN_v1 F3.1. Then run C3
(`scripts/44_partC_projection_v1.py`) only where the stimulus-matching rule allows; with only
the IFN-γ omics anchor surviving and no IFN-γ image label, C3 remains not performed and the
paper says why in one sentence.

---

## 6. Phase G3 — Leakage check, completed (fixes N5a–b)

1. Run `scripts/14_leakage_check.py` against **every movie in the frozen corpus**, i.e. all
   10 archives in `results/v1/pretrain_corpus.json` → `archives_done` (the v1 check covered 8;
   **R05 and R12 were never checked**). Use the corpus's own decoded frames or the archives;
   if an archive is no longer on disk, re-download it (MD5-verified), check, delete.
2. Make the fingerprint more discriminative: in addition to the current 40×25 z-scored
   fingerprint, compare full-resolution normalised cross-correlation of frames
   {0, n/4, n/2, 3n/4, n−1} for every pair with fingerprint r > 0.9 (v1 had 226 such pairs),
   and the frame count/shape. Declare a duplicate at full-resolution r > 0.95 on ≥ 3 of 5
   frames. Re-examine `03_F2 ↔ R16:E2` (fingerprint r = 0.968) explicitly.
3. If any new duplicate is found **in the frozen corpus**: do not rebuild. Exclude the
   affected evaluation movie(s) from tracking/forecasting for every model trained on that
   corpus, and say so. Record everything in `results/v2/leakage_check_full.json`.

---

## 7. Phase G4 — Part B corrections (fixes N9, N10)

1. **B3.1 sensitivity analysis:** its per-seed values are identical (sd 0). Vary the cell
   subsample and fold assignment per seed, or report a single value without an sd.
2. **B3.3 statistic:** with 5 conditions, leave-one-out ridge produces negatively biased
   per-target correlations (visible in the accessibility analysis). Replace "number of targets
   with r > 0.5" by the per-target **predictive R² (Q²) against the leave-one-out mean
   predictor**, summarised as the fraction of targets with Q² > 0, with the exact condition
   permutation null. Keep both analyses labelled descriptive.
3. **B4 rebuilt properly:**
   - Mouse: fit DESeq2 on GSE260996 counts with `~ priming + challenge + priming:challenge`
     over FF, FL, CF, CL, PF, PL; "priming-blunted LPS genes" = LPS-induced in the naive arm
     (padj < 0.05, log2FC > 1) **and** a significantly negative interaction term (padj < 0.05)
     for CpG and for polyI:C. Report counts for each priming and for both.
   - Human primary: **GSE185433**, LPS-tolerised→LPS vs untrained→LPS, donor-paired moderated
     test (3 donors) — this is homologous tolerance **with** restimulation, a much closer match
     than resting GSE58310, which becomes the secondary.
   - Ortholog positive control: assert that `Tnf, Il1b, Il6, Cxcl2, Ccl4, Nfkbia, Tnfaip3` map
     to their human symbols; fail otherwise.
   - Fisher test in both directions (enrichment and depletion), with the expected overlap
     printed. Keep it exploratory.

---

## 8. Phase G5 — Part C corrections (fixes N8)

1. **Source the expectations (rule 6).** Find the publications behind GSE212282 and
   GSE185433 (GEO "Citation(s)" → PubMed). For every arm (β-glucan, itaconate, 4-OI,
   β-glucan + 4-OI, LPS-tolerised, IL-4, IL-4 + LPS) record what the paper reports for
   TNF/IL-6 (protein or RNA) after LPS restimulation: direction, readout, figure/table, quote.
   Save to `results/v2/partC/arm_expectations.json`. The "as predicted" column uses only these.
   Arms with no reported phenotype get "no sourced expectation". Do not assume that itaconate
   or 4-OI train; they are often reported to blunt inflammatory responses.
2. **Positive control for restimulation.** From the GEO sample characteristics, confirm for
   every GSM that the scored sample was LPS-restimulated (record the field values). If the
   series contains non-restimulated samples, show that LPS restimulation raises TNF/IL6/IL1B in
   the untrained arm. If it does not, report the within-sample percentile of TNF, IL6, IL1B
   (expected high after 4 h LPS). A failed control blocks that dataset's results.
3. **Readouts.** Primary: per-gene TNF and IL6 log-expression differences (the canonical
   trained-immunity readouts); secondary: the pre-specified panel. Report donor-level values.
4. If β-glucan still points opposite to its sourced expectation after 1–3, report it plainly
   and list concrete candidate explanations (restimulation timing, readout depth, donor
   number) without asserting one.
5. Remove plan section numbers from figure titles; show donor-level points.

---

## 9. Phase G6 — Evaluations, figures, paper, report (after pretraining finishes)

Order (from `RESUME.md`, updated):
1. `62_bbbc054_v1.py --n-cells 9000 --scale-sensitivity` (with G2.2 fixes)
2. `63_tracking_forecast_v1.py` (with G2.3 fixes)
3. `64_dryad_v1.py` and `44_partC_projection_v1.py` if Dryad is present
4. Part B and Part C scripts with G4/G5 fixes
5. `81_figures_v1.py`, then `91_make_paper_v1.py`, `92_check_claims.py`, `94_run_report_v1.py`

### Paper fixes (fixes N1, N2, N11)
- Fix the lowercasing in `scripts/91_make_paper_v1.py` (line ~189,
  `c.rstrip(".").lower()`), and the dangling-comparative template phrase.
- §2.2 describes TiJepa-1 training from the summary (steps, schedule, segments and GPUs,
  losses, rank, the resume and its LR discontinuity).
- Remove all internal process language (rule 5). One Methods sentence may say that a
  previous internal version contained errors documented in the repository.
- The abstract is written from final results only, through the claims renderer, with correct
  provenance labels. The title is re-checked against what the results support.
- Discussion: the forecasting sentence only if TiJepa-1 beats TiJepa-1-spatial on the
  forecasting ratio; otherwise state the null.

### Figures
Regenerate all; no empty or placeholder panels (a missing analysis is omitted, not shown as
a red text box); no plan identifiers in titles; labels from provenance; donor-/movie-level
points where n is small. Inspect each PNG before committing.

### Run report
Write `reports/RUN_REPORT_v2.md` from artefacts **after** everything ran: compute per run and
per segment, data, primary/secondary results with source files, the PLAN_v1 §12 table
**re-derived** (C3 and C4 must reflect the finished runs), the Section 11 table below, and
deviations. Replace the `RESUME.md` content with a short "completed" note pointing to the
report.

---

## 10. Phase G7 — Engineering and final steps

1. Tests for: lock file and atomic save; SIGTERM handling (subprocess test with a tiny
   synthetic model); cumulative segment accounting; schedule annealing to the planned step;
   provenance guard; banned strings; E1 finite-MAE assertion; ortholog positive control.
2. `pytest -q` and `ruff check` clean; `scripts/run_all_v1.sh` updated (or `run_all_v2.sh`)
   to reproduce exactly what ran, including the resume.
3. **Backups:** final checkpoints (`tijepa1`, `tijepa1_spatial`, any extra seeds), logs,
   `results/`, `figures/`, `paper/` to `<remote>:TiJepa/`; verify; update `reports/BACKUP.md`
   and `reports/WEIGHTS.md` (paths, sizes, md5, SHA-256).
4. Push after every milestone. Update PR #1 title ("TiJepa v1: corrected computational
   preprint") and body from the v2 run report, including both traceability tables and the
   TODO(author) list.
5. Do not publish weights publicly, post to bioRxiv/Zenodo/Hugging Face, or create a GitHub
   Release.

---

## 11. New findings → fixes (the v2 run report must reproduce this table with a status)

| ID | Finding | Fix | Acceptance check |
|---|---|---|---|
| N1 | **Paper labels TiJepa-0 results as "E4 TiJepa-1"** (`91_make_paper_v1.py` maps `E4_tijepa0_only_available` → "E4 TiJepa-1"); abstract, tables, Fig 4 affected | G2.1, G6 | Every label matches its recorded checkpoint; provenance check in `92_check_claims.py` passes |
| N2 | Run report, paper, figures generated before pretraining; C3 marked "ran to its budget" and C4 inconsistent; §2.2 is a placeholder | Rule 2, G6 | `RUN_REPORT_v2.md` generated after all runs; statuses derived from finished artefacts |
| N3 | Cosine LR and EMA tied to `max_steps=150,000` while the run stops on wall-clock → never anneals; GPU-hours not cumulative across resume; no SIGTERM handler; non-atomic saves | G1.1, G1.2 | Final LR at floor and EMA = 1.0 at the last step; segment table in summary; tests |
| N4 | Dryad (and LTDB) still not provided | §2, G0.7, G2.4 | Dryad results, or explicit `not_provided` with the code path ready |
| N5 | Leakage check covered 8 of the 10 corpus archives (R05, R12 unchecked); fingerprint weakly discriminative (226 pairs r > 0.9; `03_F2↔R16:E2` r = 0.968); E0 was trained on `08_H4`'s twin | G3, G2.3.3 | `leakage_check_full.json` covers all corpus movies; E0 comparisons exclude `08_H4` |
| N6 | BBBC054 v1 used ~879 cells/seed; E1 exposure-time MAE 1.0×10⁸ h; block-CV ρ negative for every encoder (mean-reversion artifact); misleading "no time information" claim; empty scale-sensitivity panel | G2.2 | n ≈ 9,000/seed; finite MAEs; buffered LOFO-CV; confound described on the full file |
| N7 | Tracking used 18 tracks (v0: 125) and flipped direction vs v0; ⟨δ⟩ and survival disagree without explanation; forecasting on 4 of 6 movies with no reason; random-init forecasting ratios presented as comparable | G2.3 | ≤ 50 tracks/movie; per-threshold curves; all movies or stated reasons; TiJepa-1 vs spatial primary |
| N8 | Part C "expected direction" assumed for itaconate, 4-OI, β-glucan + 4-OI and IL-4 arms; no check that scored samples were LPS-restimulated; panel mixes interferon-stimulated genes | G5 | `arm_expectations.json` with citations; restimulation positive control; TNF/IL6 primary readouts |
| N9 | B4 "blunted" = lower point estimate (1,142 of 1,440 induced genes) rather than an interaction test; human side is resting GSE58310 although restimulated tolerance (GSE185433) exists; no ortholog positive control | G4.3 | DESeq2 interaction model; GSE185433 primary; ortholog control passes |
| N10 | B3.1 sensitivity sd = 0 (cosmetic); B3.3 "count r > 0.5" under 5-point LOO is negatively biased | G4.1–2 | Real sd or none; Q²-based statistic |
| N11 | Manuscript text bugs: lowercased identifiers in §2.3, dangling "is higher than (+0.082 …)", review IDs/plan references/"(2 blocking requirements recorded.)"/v0 narrative in the body; Fig 4 empty panel; Fig 7 text panel with plan references | G2.1, G6 | Banned-strings and grammar checks pass; figures inspected |
| N12 | `RESUME.md` budget is wall-clock on a shared A100; not transferable to a different GPU | G1.2 | `pretrain_plan.json` with steps-based budget and measured throughput |
| N13 | Two instances share storage: risk of concurrent writers; memory-mapped shards on a network filesystem may starve the GPU | G0.3, G0.6, rule 3 | Lock file; throughput record; GPU utilisation ≥ 80% or documented |
| N14 | Claims checker passed despite N1 | G2.1 | New provenance, banned-string and grammar checks with tests |
| N15 | BBBC054 time-only baseline and within-bin controls are uninformative by construction of the class-balanced sample | G2.2.4 | Text states this; full-file confound statistics reported |

Still open from PLAN_v1 §12 and carried forward: D2 (Dryad), D4a/D4b/D4e (corpus breadth —
**not** re-opened: the frozen corpus is kept so v1 results stay pinned; stated as a
limitation), C3 and C4 (closed by G1 and G2.3 if the runs complete), D6 (LTDB if provided).

---

## 12. Milestones and time guide

| # | Milestone | Commit prefix | Guide |
|---|---|---|---|
| W0 | G0: integrity verified, partial checkpoint backed up, lock in place | `chore(v2):` | 0.5 h |
| W1 | G1.1 trainer fixes + tests; G1.2 budget decision | `partA(v2):` | 1–1.5 h |
| W2 | G1.3 resume main run (background) | `partA(v2):` | start |
| W3 | G2 + G3 + G4 + G5 code fixes and CPU-side reruns (while training) | `eval(v2):` / `partB(v2):` / `partC(v2):` | 6–10 h |
| W4 | Main run done → ablation run (1 GPU) | `partA(v2):` | per G1.2 |
| W5 | G6 evaluations with final checkpoints | `eval(v2):` | 3–5 h |
| W6 | Figures, paper, report, PR, backups | `paper(v2):` / `chore(v2):` | 2–3 h |

### Stop / degrade rules
- Integrity mismatch or another writer detected → stop, `reports/BLOCKED.md`, no training.
- GPU utilisation stays low after G0.6 → continue, record it, adjust the step budget per G1.2.
- Safety cap hit before S_total → stop gracefully, evaluate that checkpoint, report it as
  unannealed.
- Dryad absent → everything else proceeds; Dryad-dependent results are `not_provided`.
- A positive control fails → that analysis reports the failure, not results.
- Never fill a gap with invented data, numbers or references.
