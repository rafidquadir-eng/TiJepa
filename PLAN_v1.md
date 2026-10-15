# TiJepa — Fix Plan v1 (from the v0 review)

> **For the executing agent.** This plan fixes every problem found in the review of the v0
> run (branch `tijepa-v0`, PR #1). Read it end to end, then `PLAN.md` (still in force), then
> `reports/RUN_REPORT.md` from v0. Execute phases F0 → F9 in order. Section 12 maps every
> review finding to the phase that fixes it and to an acceptance check; **the run is not
> finished until every row in Section 12 has a status in the v1 run report.**
>
> Where this plan and your engineering judgment disagree on a detail, follow your judgment
> and record the deviation. Where this plan states a **rule**, it is not negotiable.

---

## 0. Objective and what changes

Produce **TiJepa v1**: a corrected, re-run, preprint-ready version of the v0 work.

| Area | v0 state | v1 target |
|---|---|---|
| Part A headline | Missing (Dryad blocked) | Dryad M0/M1/M2 **video** benchmark, leave-one-batch-out, all encoders incl. frame-shuffle |
| Pretraining | 1 source, 7/21 archives, 4.4 GPU-h, spatial-only masking, 1 seed | Full ComplexEye + PhagoStat microglia, QC-filtered, leakage-checked, **mixed spatial + future masking**, full budget, seeds per tier, matched-budget ablation |
| Tracking | 5 movies, possible leakage | All non-overlapping S-BIAD2196 movies + LTDB (if provided) + CoTracker (if reachable) + a **forecasting** evaluation |
| Part B | ATAC mislabelled as RNA; B4 artifact | Real RNA-seq (GSE260996) loaded; ATAC analysed and labelled as accessibility; honest small-n statistics; ligand codes confirmed |
| Part C | Donor-confounded LPS axis; resting-state only; C3 not run | Donor-matched, sanity-checked anchors; restimulation-based test; C3 projection with matched stimuli |
| Statistics | Cosmetic seeds, 4-cluster bootstrap, permutation floors | Real seed variance, exact/adequate permutation, CIs only where valid |
| Paper | Table/prose/reference/figure bugs, hard-coded conclusions | Data-driven prose with a claims checker, full Methods, verified references, redesigned figures |
| Engineering | Unpinned deps, absolute paths, thin tests | Lockfile, relative paths, `run_all.sh` = what ran, data-type guard tests |
| Weights | Only on instance disk | Backed up to the author's Google Drive and checksum-verified |

### Branch and PR
Continue on branch **`tijepa-v0`** (PR #1 updates automatically). `git pull` first. Never
force-push, never push to `main`, never rewrite history. The author will **squash-merge**,
which drops the 39 MB blob already in branch history. Rename the PR title to
`TiJepa v1: corrected computational preprint` at the end.

---

## 1. Rules

All of `PLAN.md` §1 remains in force (integrity, licences, hygiene, authorship). Additions:

1. **Data-type guard.** Every omics matrix is checked on load: row-ID pattern (gene symbol /
   Ensembl ID vs genomic interval `chrN.start.end` / `chrN:start-end`), sample count against
   the GEO series, and assay type against GEO `library_strategy`. A mismatch is a hard error.
   This is how the v0 ATAC-as-RNA bug must never recur.
2. **Sanity checks before results.** Every differential-expression or signature step must
   first pass a positive control (Section 7.1). A pipeline that fails its positive control
   produces no reported result; the failure is reported instead.
3. **No CIs from fewer than 10 groups.** With < 10 grouping units, report per-group values,
   the exact permutation distribution and the minimum achievable p, not a bootstrap CI.
4. **Seeds must vary something.** Any "mean ± sd over seeds" must come from runs that differ
   in a real source of randomness (pretraining init, random-init draw, cell subsample, fold
   assignment). If sd is exactly 0, the seed claim is removed.
5. **Prose follows numbers.** Directional words in the paper ("higher", "beats", "worse",
   "indistinguishable") are generated from CIs by code (Section 10.2). Hand-written
   directional claims are forbidden.
6. **Google Drive is the author's private storage** and is authorised for backups. It is not
   "posting". Never print, log, or commit rclone config, tokens, or credentials.
7. **The circular "reduced one-axis" analysis is deleted** from code, results and paper.

---

## 2. Author prerequisites (done by the author before launching the agent)

1. **Instance.** Recommended: 4× A100 or 4× H100 (three pretraining seeds + matched
   ablation in parallel). 1 GPU works with the reduced schedule in Section 5.4. ≥ 1 TB free
   local disk.
2. **Google Drive.** An rclone remote of type `drive` configured on the instance
   (`rclone listremotes --long` shows it).
3. **Manual downloads** (both blocked for automated clients in v0), via a normal browser,
   uploaded to Google Drive:
   - Dryad `doi:10.5061/dryad.m0cfxpp8v` → "Download dataset" → upload the zip to
     `TiJepa/manual_downloads/dryad_m0cfxpp8v/`
   - LTDB figshare collection `10.6084/m9.figshare.c.3827890` → upload to
     `TiJepa/manual_downloads/ltdb/`
   If either is absent when the agent runs, the agent proceeds and records it (Section 11).

---

## 3. Phase F0 — Backup, preflight, manual-download intake

### F0.1 Back up the v0 checkpoint FIRST (before anything else)
1. Locate the v0 checkpoint: `checkpoints/tijepa0_T1_seed0.pt` (v0 path on the old instance:
   `/home/ubuntu/proj2/checkpoints/`). If this is a new instance and it is absent, record
   "v0 checkpoint not on this instance" and skip to F0.2.
2. Find the Drive remote: `rclone listremotes --long`; choose the remote whose type is
   `drive` (if several, prefer one named like `gdrive`/`drive`; record the name only).
   Fallbacks, in order: a mounted Google Drive path visible in `mount` / `df` output; a
   `gdrive` CLI. If none works, write `reports/BACKUP.md` with exact manual commands, flag it
   at the top of the PR body, and continue.
3. Copy and verify:
   ```bash
   rclone copy checkpoints/ <remote>:TiJepa/checkpoints_v0/ --include "*.pt" --include "*.csv" --progress
   rclone check checkpoints/ <remote>:TiJepa/checkpoints_v0/ --one-way --include "*.pt"
   ```
   Also record local `md5sum` and `rclone md5sum` of the remote file; they must match.
4. Write `reports/BACKUP.md`: remote folder path, file names, sizes, md5s, timestamp. No
   credentials.

### F0.2 Preflight
Re-run `scripts/00_preflight.py` (extend it): GPUs, CPU, RAM, disk, reachable hosts
(add `huggingface.co`, `api.crossref.org`, `doi.org`, `www.cellpose.org`), rclone remote
detection, and compute tier (Section 5.4). Output `results/v1/preflight.json`.

### F0.3 Intake of manual downloads
`rclone copy <remote>:TiJepa/manual_downloads/ data/raw/manual/` then:
- Dryad: unzip into `data/raw/dryad_bmdm/`; verify every file against the SHA-256 digests the
  Dryad API publishes (already implemented in `scripts/11_download_labelled.py --only dryad`).
- LTDB: unpack into `data/raw/ltdb/`; record file list and sizes.
Status of each → `results/v1/manual_intake.json`.

**Results layout for v1:** write all new results under `results/v1/` and new figures under
`figures/`. Move v0 results to `results/v0/` (git mv) so nothing is silently overwritten.

---

## 4. Phase F1 — Data fixes

### F1.1 Part B: the real RNA-seq (fixes D1)
1. **Inventory the Tay deposit.** List every file in the Zenodo `14337874` zip (it is the
   `tay-lab/macrophage_memory` code release) with size and a one-line description. Identify
   which files are traces, RNA-seq, ATAC-seq, and which scripts read them.
2. **Classify `logCPM.csv` correctly.** It has 18 samples (9 conditions × 2 reps: CL, CT, MC,
   ML, MM, MP, MT, PL, PT) and row IDs like `chrX.163316160.163316661`, matching **GSE261647
   ATAC-seq** (18 samples, same condition codes). Label it ATAC everywhere.
3. **Load GSE260996 RNA-seq.** 33 samples = 11 conditions × 3 reps (`CPG_media`, `CPG_LPS`,
   `CPG_TNF`, `media_CPG`, `media_media`, `media_LPS`, `media_polyIC`, `media_TNF`,
   `polyIC_media`, `polyIC_LPS`, `polyIC_TNF`), RelA-mVenus BMDMs (same genotype as the
   traces). Processed per-sample files are `GSM*_??NAligned.RData.gz` inside
   `GSE260996_RAW.tar` — **these are processed outputs, not raw reads; download them.**
   - Read with `pyreadr`; if the R objects are lists that `pyreadr` cannot parse, install R
     (`apt-get install -y r-base-core` or conda `r-base`) and convert with a small `Rscript`
     to CSV. Read the Tay-lab RNA processing code to learn the object structure.
   - Build a gene × sample count matrix (or the closest processed quantity available), with
     gene IDs, and save to `data/interim/tay/rnaseq_counts.csv` + a schema note.
   - Read the GEO `treatment_protocol_ch1` to get stimulus timings for matching to traces.
4. **Data-type guard** (Rule 1) on both matrices, plus tests (F8).

### F1.2 Part B: confirm the trace layout and ligand codes (fixes D7)
1. Find, in the deposit's MATLAB/Python scripts, the code that maps numeric ligand codes to
   names (plotting or figure scripts usually do). Record file + line in
   `results/v1/partB/ligand_codes.json`. If found, it overrides the v0 inference.
2. **Biological sanity check** of the LPS code: the NF-κB second-response peak amplitude for
   LPS→LPS must be lower than for naive→LPS (classic tolerance). Report effect size and
   direction. If contradicted, flag the code mapping as unconfirmed in the paper.
3. Try to verify the layout of the 4 excluded files (cytokine-block, sepsis) from the
   deposit's scripts. Include any whose layout is confirmed; search all confirmed files for
   **every** LPS→LPS well at any interval (v0 found tolerant cells only at 8 h in one file).

### F1.3 Leakage check: S-BIAD2196 vs ComplexEye (fixes D3)
1. Download **all 12** S-BIAD2196 movies (v0 had 5).
2. For every S-BIAD2196 movie and every ComplexEye movie: record (frames, H, W) and a
   fingerprint = frames {0, n/2, n−1} downsampled to 40×25, z-scored, flattened.
3. Flag pairs with identical shape and fingerprint Pearson r > 0.98 (also exact SHA-1 of
   frame 0). Write `results/v1/leakage_check.json` (all pairs above r = 0.9, decision).
4. **Exclude every matching ComplexEye movie from pretraining.** S-BIAD2196 movies with a
   match are still usable for evaluation once their twin is excluded. State the outcome in
   the paper either way.

### F1.4 Pretraining corpus v1 (fixes D4a–e)
1. **ComplexEye:** download all 21 archives from Zenodo `10082674`, **one at a time**, with
   resume and MD5 verification against the record; extract → preprocess → delete the zip
   before the next (bounds disk). Re-download any archive whose members failed CRC in v0.
   Target: all archives; hard cap 200 GB downloaded.
2. **Leica movies (`*_leica.zip`):** find the pixel size in the ComplexEye paper (Cibir et
   al., *Nat Commun* 2023, DOI 10.1038/s41467-023-43765-3, Methods) or file metadata. If it
   cannot be verified, **exclude** Leica movies and say so.
3. **PhagoStat microglia video** (Zenodo `10803492`, CC BY 4.0): read the 94.2 GB archive
   with HTTP range requests (`remotezip`); extract a stratified subset of acquisitions/scenes
   (WT and FTD) up to **40 GB**. Get the frame interval and pixel size from the record/paper
   (DOI 10.1038/s41598-024-56081-7); resample to the common 0.65 µm/px. If range reads fail,
   download the full archive only if disk allows; otherwise skip and record.
4. **Frame QC (S-BIAD2133, MIT):** finish the paginated download (`scripts/11_download_labelled.py --only s2133` already paginates). Train the frame-level anomaly classifier
   (clean vs any-anomaly; small CNN or logistic on downsampled frames), pick the threshold
   on its validation split, apply to all pretraining clips, and report the removed fraction
   per source. If S-BIAD2133 still cannot be completed, apply documented heuristic QC
   (focus = variance of Laplacian, saturation fraction, global intensity drift) with
   thresholds set from the empirical distribution, and label it heuristic.
5. **Temporal strides:** sample strides {1, 3, 9, 27} where movie length allows (8 s →
   3.6 min effective interval) so the evaluation frame interval (Dryad 5 min) is closer to
   the training distribution. Record the stride distribution.
6. **Source balance:** sample clips ~70% ComplexEye / ~30% PhagoStat (by sampling weight,
   not duplication). Record final counts.
7. **Static-clip augmentation:** with probability 0.1, replace a training clip by one frame
   repeated 16 times (fixes C5 — static inputs become in-distribution).
8. Output `results/v1/pretrain_corpus.json` (per source: movies, clips, µm/px and its
   provenance, intervals, QC removal, leakage exclusions).

### F1.5 Evaluation data
- **Dryad (headline):** parse README/filenames for condition, batch (5 batches, 20 mice) and
  magnification (10× = 2.155 px/µm, 20× = 4.31 px/µm); resample to 0.65 µm/px; detect and
  track cells (same pipeline as pretraining, or Cellpose if available — record which).
- **BBBC054:** determine µm/px from the instrument (Leica DMI 6000, 20×, Andor camera model
  in the BBBC054 page): pixel size = camera pixel pitch / total magnification. If the pitch
  cannot be verified from a manufacturer or paper source, run a **scale-sensitivity
  analysis** (resample factors 0.5, 0.75, 1.0, 1.5) and report all.
- **CoTracker:** try `facebook/cotracker3` (or `cotracker2`) via Hugging Face; record the
  licence (likely non-commercial — acceptable only if stated in Methods per `PLAN.md`
  §1.2.3). If unreachable, record it.

---

## 5. Phase F2 — Pretraining TiJepa-1

### F2.1 Objective change: mixed spatial + future masking (fixes C4)
Keep the v0 V-JEPA architecture. Per batch, sample the mask type:
- **Spatial tube mask** (v0, p = 0.5): multi-block spatial blocks masked at all times.
- **Future mask** (p = 0.5): all spatial positions of the last 2 temporal tubelet slots
  (= last 4 frames) are targets; the first 6 slots (12 frames) are context, optionally with
  light spatial masking (≤ 15%) in the context.
The predictor receives target positional embeddings as in v0. This makes TiJepa-1 a genuine
latent **forecasting** model, which F3.4 evaluates. Unit-test the mask generator.

### F2.2 Budget and stopping (fixes C3)
Stop by **wall-clock GPU-hours**, not an arbitrary step count. Log loss, embedding std and
effective rank every 250 steps; evaluate a fast linear probe (Dryad, 1 fold) every ~2 h as a
monitor only (never used for model selection beyond "did it collapse"). Checkpoint every
30 min, resume-safe. Back up each **final** checkpoint to Drive (F9.1).

### F2.3 Ablation (isolates the new objective)
`TiJepa-1-spatial`: identical corpus, identical steps, spatial masking only.

### F2.4 Schedule by tier
| GPUs | Runs (in parallel, one per GPU) | Budget per run |
|---|---|---|
| ≥ 4 | TiJepa-1 seeds 0, 1, 2 + TiJepa-1-spatial seed 0 | 16 GPU-h |
| 3 | TiJepa-1 seeds 0, 1 + spatial seed 0 | 16 GPU-h |
| 2 | TiJepa-1 seed 0 + spatial seed 0 | 16 GPU-h |
| 1 | TiJepa-1 seed 0 (12 GPU-h), then spatial seed 0 for the **same number of steps** | ≈ 24 GPU-h total |

ViT-S stays the encoder (ViT-B only if ≥ 4 GPUs and all ViT-S runs finished with time left;
then report it as an extra, not the headline). Keep TiJepa-0 (v0) as a historical reference
row only; it is not a valid ablation because its corpus differs.

---

## 6. Phase F3 — Part A evaluation v1

### Encoders (identical probes for all)
E1 hand-crafted (improved, below) · E2 ImageNet ViT-S · E3 DINOv2 ViT-S/14 · E4 TiJepa-1
(per seed) · E4s TiJepa-1-spatial · E5 TiJepa-1 frame-shuffled · E6 TiJepa-1 architecture
random-init (**three independent random draws**, seeds 0/1/2) · E0 TiJepa-0 (reference).

**E1 improved (fixes C6):** segment with Cellpose (`cyto3`, licence recorded) if installable
and its weights download; otherwise a proper brightfield/phase pipeline (local-variance or
Sobel-magnitude map → threshold → morphological fill → watershed). Features: shape (area,
perimeter, eccentricity, solidity, axes, circularity, Zernike moments) + texture (Haralick on
the cell mask) + for videos, motility (speed, persistence, MSD α, turning angles, fraction
moving). Probe with both logistic regression and random forest; report the better as E1.

### F3.1 Dryad M0/M1/M2 — headline (fixes D2)
- Unit: tracked cell (≥ 16 frames); clips at the native 5-min interval.
- **Leave-one-batch-out** (5 folds); never split a movie. Metrics: balanced accuracy,
  macro-F1, one-vs-rest AUROC at cell and movie level. CIs by cluster bootstrap over the 18
  movies. Permutation: permute condition labels at the **movie** level, ≥ 1,000.
- Controls: E5 frame-shuffle; E6 random-init (3 draws); confound probes predicting batch and
  magnification from embeddings; results stratified by magnification.
- Compare with the accuracy reported in the source paper (Sun & Kesapragada) if stated,
  noting any task differences.

### F3.2 BBBC054 — redesigned (fixes D5)
1. Two-class (round vs amoeboid) is stated plainly; "stratified" is absent from the
   annotated replicate.
2. **Confound controls:** (a) *time-only baseline*: classify morphology from exposure time
   alone; (b) **within-time-bin** evaluation (6 contiguous bins of 10 FOVs; balanced
   accuracy computed within bins and averaged), so time cannot explain the score; (c) the
   FOV-identity probe as in v0.
3. **Exposure-time regression** is demoted to exploratory, with **contiguous block CV**
   (leave 6 consecutive FOVs out) because FOV = time.
4. Scale per F1.5 (resampled or scale-sensitivity).
5. Label it a **morphology-class** task, never "activation state" (fixes B5).

### F3.3 Tracking (fixes D6)
All non-leaked S-BIAD2196 movies (target ≥ 10), LTDB (2D max projections) if provided,
CoTracker if available. Same tracker for all encoders (feature correlation within a search
window + identical motion prior). Metrics as v0 (⟨δ⟩ over 1–16 µm, median survival) with
movie-level cluster bootstrap. Report honestly if TiJepa still does not beat the baseline.

### F3.4 Latent forecasting evaluation (new; fixes C4, supports the "world model" framing)
On held-out movies (non-leaked S-BIAD2196 + held-out ComplexEye + Dryad): give each model the
first 12 frames, predict its own target-encoder latents for the last 4. Compare against that
model's own baselines: **copy-last-observed-latent** and **mean-latent**. Report the ratio
`L1(predictor) / L1(copy-last)` per model (latent spaces differ, so only within-model
ratios are comparable). TiJepa-1-spatial is queried with the same future mask
(out-of-distribution; report as such). Bootstrap over movies.

### F3.5 Seeds (fixes C1)
Variability comes from: pretraining seeds (where available), E6 random-init draws, a
different stratified cell subsample per seed, and `StratifiedGroupKFold(shuffle=True,
random_state=seed)` fold assignment. Report mean ± sd and per-seed values.

---

## 7. Phase F4 — Part B v1

### 7.1 Positive controls (Rule 2) — must pass before any Part B result
- Mouse RNA (GSE260996): `media_LPS` vs `media_media` must show up-regulation of canonical
  LPS/NF-κB targets (*Tnf, Il1b, Il6, Cxcl2, Ccl4, Nfkbia, Tnfaip3*; ≥ 5 of 7 with
  log2FC > 1, FDR < 0.05).
- Mouse ATAC (GSE261647): `ML` vs `MM` must show increased accessibility at peaks within
  ±10 kb of those genes' TSSs (report how many).
- Traces: the tolerance sanity check of F1.2.

### 7.2 B3.1 tolerance from dynamics (fixes C2)
Use every confirmed LPS→LPS vs media→LPS well (F1.2). Grouped CV by well. If ≤ 9 wells:
report per-well predictions, the **exact** permutation distribution over all distinct
well-label assignments, the minimum achievable p, and **no bootstrap CI**. State plainly
that the design cannot reach significance if that is the case. Describe it as **acute
endotoxin tolerance** (LPS → LPS hours apart, mouse BMDM) (fixes B4).

### 7.3 B3.2 ligand identity
Keep, with fold-shuffled seeds (F3.5). Note the class set differs from the published one.

### 7.4 B3.3 dynamics → transcriptome **and** → accessibility (fixes D1, C7)
Match trace conditions to RNA conditions using stimulus identity **and** timing from the GEO
protocols; list the matched conditions. Two analyses, labelled correctly:
(a) dynamics → **RNA expression** (GSE260996); (b) dynamics → **chromatin accessibility**
(GSE261647). Leave-one-condition-out ridge as v0, plus a **pre-specified NF-κB target
panel** (the 7 genes above + *Il12b, Cxcl10, Ifit1, Irf7*) analysed separately. Permutation
null over condition labels (exact if feasible). With few conditions, present as descriptive
and say so.

### 7.5 B4 cross-species, rebuilt (fixes D1, C7)
- Mouse: from **RNA**, define priming-suppressed LPS genes = genes induced in
  `media_LPS` vs `media_media` whose induction is reduced in `CPG_LPS` and `polyIC_LPS`
  (interaction contrast; DESeq2 via `pydeseq2` on counts, or limma-voom if only processed
  values exist).
- Human: GSE58310 LPS-tolerised vs control with a **moderated paired test** (R
  `limma` via Bioconductor if installable; else `inmoose.limma`; else a Python empirical-Bayes
  implementation validated in tests against a known example).
- Map with the MGI ortholog table; Fisher test; report overlap genes.
- Pre-register in the paper that the two contrasts differ (heterologous priming vs
  homologous tolerance; mouse acute vs human 6-day) and treat the test as exploratory.

---

## 8. Phase F5 — Part C v1 (hypothesis-generating, as before)

### 8.1 Anchors (fixes B1, B3)
1. Enumerate GSE46903 samples by stimulus × time × differentiation (GM-CSF / M-CSF) × donor;
   save the table.
2. **Prefer the differentiation closest to GSE58310** (human-serum, M-CSF-like) if Xue
   provides it; otherwise use GM-CSF and state the mismatch.
3. **Donor matching:** build each signature from donor-paired stimulus-vs-control samples
   (paired moderated test, donor as block). If a stimulus has < 3 donor-matched pairs, it
   cannot form an anchor.
4. **Sanity rule:** an LPS anchor must contain ≥ 5 canonical LPS genes among its top 100
   (*TNF, IL1B, IL6, CXCL8, CCL3, CCL4, CCL20, NFKBIA, TNFAIP3, PTGS2, IL1A*), an IFN-γ
   anchor ≥ 5 of (*GBP1, GBP5, CXCL9, CXCL10, IDO1, IRF1, STAT1, TAP1*), an IL-4 anchor ≥ 3 of
   (*CCL17, CCL22, MRC1, CD209, ALOX15, FCER2, CCL26*). An anchor that fails is dropped and
   reported (v0's S_LPS of zinc-finger genes would fail).

### 8.2 Resting-state scores (fixes B2 framing)
Score GSE58310 as v0 with the sanity-passed anchors. State in text and legend that these
are **resting** day-6 cells and that trained immunity is defined by the **restimulation**
response, so proximity to baseline is expected. **Exclude naive monocytes from the bridge**
(their displacement reflects monocyte→macrophage differentiation, not activation) — show
them in a separate panel labelled as such.

### 8.3 Restimulation test (new; the biologically meaningful version)
Search, in order, for a processed public matrix with **naive→LPS and trained→LPS (and/or
tolerant→LPS) on matched donors**: GSE212282, GSE185433, GSE323311, GSE242947, GSE184241
(pseudobulk). Use the first that qualifies (data-type guard applies). Score the M1-like
anchor on restimulated samples; test trained→LPS vs naive→LPS (expect higher) and
tolerant→LPS vs naive→LPS (expect lower), paired by donor. Report whichever direction is
observed. If no dataset qualifies, report that and what was checked.

### 8.4 C3 projection (now possible with Dryad)
Image anchors = TiJepa-1 Dryad centroids. **Stimulus must match** across omics and image:
image M1 = LPS 100 ng/ml, image M2 = IL-4 20 ng/ml. Use only omics axes whose anchor
stimulus matches (LPS↔LPS, IL-4↔IL-4) and passed 8.1; if the LPS omics anchor fails, run
C3 on the IL-4 axis alone and say so. Hypothesis framing everywhere; nearest-cell exemplars
labelled "nearest observed cells to a predicted position — not images of trained cells".

### 8.5 Delete the circular analysis (fixes C8)
Remove `reduced_one_axis_analysis` from code, `results/`, figures and paper.

---

## 9. Phase F6 — Statistics standards (applies to all phases; fixes C1, C2, C7)

| Situation | Requirement |
|---|---|
| ≥ 10 groups | Cluster-bootstrap 95% CI, 2,000 resamples |
| < 10 groups | Per-group values + exact permutation distribution + minimum achievable p; no CI |
| Permutation | Exact enumeration when ≤ 10,000 distinct assignments; otherwise ≥ 1,000 permutations (parallelise with joblib); report the floor |
| Encoder comparisons | Paired bootstrap on identical resamples; BH correction across comparisons within a task |
| Differential expression | Moderated tests (limma / DESeq2); paired/blocked designs where donors are shared |
| Primary endpoints (pre-specified) | (1) Dryad LOBO balanced accuracy, E4 vs E3 and E4 vs E6; (2) forecasting ratio vs copy-last; (3) tracking ⟨δ⟩ vs baseline; (4) restimulation test 8.3. Everything else is secondary/exploratory and labelled so |

---

## 10. Phase F7 — Paper, figures, references

### 10.1 Text fixes (fixes E1, E2, B4, B5, B6)
- Rebuild every table from a single table-writer that asserts header length = row length.
- Section 2.6: no raw status strings; one C3 paragraph; human-readable replication notes.
- Methods complete and numbered: 4.1 compute, 4.2 data & preprocessing, 4.3 leakage check,
  4.4 model & objective, 4.5 probes & statistics, **4.6 tracking & forecasting, 4.7 Part B,
  4.8 Part C**, 4.9 licences, 4.10 AI-assistance disclosure, 4.11 competing interests.
- Wording: "acute endotoxin tolerance" (Part B); "morphology class" (BBBC054); "pretrained on
  neutrophil and microglia video" (not "innate immune activation" in general).
- Discussion: remove "NF-κB dynamics carry information about prior LPS exposure" unless the
  v1 result is significant; otherwise "were not significantly predictive with 4 wells".
- State the cell-type chain explicitly as a limitation: pretraining (neutrophils +
  microglia) → evaluation (macrophages, microglia) → target biology (human monocytes /
  macrophages).

### 10.2 Data-driven prose and claims checker (fixes E5)
- Template placeholders for comparisons render from CIs: `{{CMP:E4:E3:dryad}}` →
  "higher than" / "lower than" / "not distinguishable from".
- `scripts/92_check_claims.py`: parses `paper/tijepa.md` for directional words near encoder
  names and verifies each against `results/v1/`; fails the build on any contradiction. Run
  it in `run_all.sh` and in tests.

### 10.3 Figures (fixes E4) — read the repo's plotting conventions; colour-blind-safe palette
1. **Fig 1 Overview:** a real schematic (data sources → pretraining → evaluations; paired vs
   bridged links drawn differently). No truncated text.
2. **Fig 2 Pretraining:** corpus composition (sources, QC removal, leakage exclusions), loss,
   effective rank, per seed.
3. **Fig 3 Dryad headline:** dot-and-CI plot of balanced accuracy for all encoders, confusion
   matrix, frame-shuffle and random-init controls, batch/magnification confound.
4. **Fig 4 BBBC054:** morphology probe with time-only baseline and within-bin results; scale
   sensitivity.
5. **Fig 5 Tracking + forecasting.**
6. **Fig 6 Part B:** tolerance (per-well), positive controls, dynamics→RNA and →ATAC, B4.
7. **Fig 7 Part C:** anchors with sanity results, resting scores (naive monocytes separate),
   restimulation test, C3 projection — no empty panels; labels do not overlap.
8. **Fig 8 Landscape** (from the review report).
Use dot-and-CI plots or bars starting at 0; never a truncated bar axis. Min font 8 pt.
Render each figure to PNG and inspect it before committing.

### 10.4 References (fixes E3)
Resolve every DOI via Crossref (`api.crossref.org/works/<doi>`) or doi.org before listing;
anything unresolved stays `TODO(verify)`. Known corrections and candidates (verify each):
- V-JEPA: **Bardes A**, Garrido Q, Ponce J, Chen X, Rabbat M, LeCun Y, Assran M, Ballas N.
  *TMLR* 2024; arXiv **2404.08471** (v0 wrongly listed Assran as first author).
- DINOv2: Oquab M, et al. arXiv **2304.07193**.
- ComplexEye: Cibir et al., *Nat Commun* 2023, DOI **10.1038/s41467-023-43765-3**.
- PhagoStat: *Sci Rep* 2024, DOI **10.1038/s41598-024-56081-7**.
- BBBC: Ljosa V, Sokolnicki KL, Carpenter AE, *Nat Methods* 2012, DOI 10.1038/nmeth.2083; plus
  the BBBC054 source publication if the BBBC page names one.
- Saeed S et al., *Science* 2014, DOI 10.1126/science.1251086.
- Novakovic B et al., *Cell* 2016, DOI 10.1016/j.cell.2016.09.034.
- Xue J et al., *Immunity* 2014, DOI 10.1016/j.immuni.2014.01.006.
- Netea MG et al., *Science* 2016, DOI 10.1126/science.aaf1098 (trained immunity review).
- Wang AG, Son M, et al., *Cell Systems* 2025 (PMC12147866) — resolve DOI.
- Sun & Kesapragada et al. (Dryad BMDM source paper) — resolve.
- trackpy (Allan DB et al., Zenodo software DOI for the version used); Cellpose (Stringer C
  et al., *Nat Methods* 2021) if used; CoTracker (Karaev N et al.) if used.
- MGI ortholog report: cite the current MGI database paper (resolve; **remove "Blumberg B"**).
- limma (Ritchie ME et al., *Nucleic Acids Res* 2015) / DESeq2 (Love MI et al., *Genome
  Biol* 2014) / pydeseq2, as used.

---

## 11. Phase F8 — Engineering, reproducibility, hygiene (fixes E6, E7)

1. **Dependencies:** write `requirements-lock.txt` (`pip freeze`) and keep `pyproject.toml`
   ranges. **One torch version for the whole v1 run**; record it in every stage's
   `env.json`. Fix the v0 run-report contradiction (training ran on 2.11 per its log; the
   v0 report claimed an upgrade before training) in a "v0 errata" note.
2. **Paths:** all paths in `results/` are repo-relative; scrub v0 files too.
3. **`run_all.sh`** reproduces exactly what v1 ran (same flags, caps, budgets). No `|| true`;
   optional stages check their inputs and write an explicit `status: skipped, reason` JSON.
4. **Tests** (pytest; synthetic fixtures still only `synthetic_*`; data-dependent tests use
   `@pytest.mark.data` and skip when data are absent):
   - data-type guard: RNA row IDs are genes, ATAC row IDs are intervals, sample counts
     match GEO (33 / 18);
   - BBBC054 coordinate convention: all annotations fall inside image bounds under the chosen
     axis order and a large fraction would fall outside under the transposed order;
   - Tay ligand-code mapping + tolerance sanity direction;
   - mask generator (spatial vs future) shapes and coverage;
   - table writer header/row lengths; claims checker on a synthetic paper.
5. **Hygiene:**
   - `git rm --cached results/downloads/s-biad2133.json` → keep the full listing under
     `data/` (gitignored) and commit a ≤ 50 KB summary;
   - add `*.npz` and `*.npy` to `.gitignore`; add a pre-commit/CI check failing on any
     tracked file > 5 MB;
   - fix the stale comment in `scripts/60_partA_eval.py` ("page states axis-1 = x");
   - the 39 MB blob in history is resolved by squash-merge (note in PR body).

---

## 12. Issue → fix traceability (every v0 finding)

The v1 run report must reproduce this table with a **Status** column
(fixed / partially fixed / not fixable — with reason) and the evidence file.

| ID | Finding | Fix | Acceptance check |
|---|---|---|---|
| D1 | Part B used ATAC as RNA; B4 artifact | F1.1, 7.4, 7.5 | Guard tests pass; RNA matrix 33 samples with gene IDs; B3.3 has RNA and ATAC analyses labelled correctly; B4 uses RNA |
| D2 | Dryad headline missing | §2, F0.3, F3.1 | Dryad LOBO results for all encoders, or an explicit "not provided by author" status |
| D3 | Possible tracking leakage | F1.3 | `leakage_check.json`; matched movies excluded from pretraining |
| D4a | 7/21 archives, 14 corrupt | F1.4.1 | Archive table with MD5 verified; corrupt members re-downloaded or listed |
| D4b | Single source | F1.4.3 | PhagoStat clips in corpus (or documented failure) |
| D4c | Leica pixel size unverified | F1.4.2 | Verified value with source, or Leica excluded |
| D4d | No QC | F1.4.4 | QC classifier (or labelled heuristic) and removal fraction per source |
| D4e | PhagoStat video unused | F1.4.3 | Same as D4b |
| D5a | BBBC054 static | F1.4.7, F3.2 | Static-clip augmentation in pretraining; Dryad is the video headline |
| D5b | 2 of 3 classes | F3.2.1 | Stated plainly (not fixable from data) |
| D5c | FOV–time confound | F3.2.2–3 | Time-only baseline, within-bin results, block CV |
| D5d | BBBC054 pixel size | F1.5 | Verified µm/px or scale-sensitivity results |
| D6 | Small tracking eval | F1.3, F3.3 | ≥ 10 movies or all available; LTDB/CoTracker run or status recorded |
| D7 | Tay layout inferred | F1.2 | Code-mapping source cited, sanity direction reported |
| C1 | Cosmetic seeds | F3.5, Rule 4 | sd > 0 or seed claim removed |
| C2 | 4-well bootstrap CI | 7.2, §9 | No CI with < 10 groups; exact permutation + floor reported |
| C3 | Undertrained | F2.2, F2.4 | GPU-hours used ≈ budget; loss/rank curves |
| C4 | No future prediction | F2.1, F3.4 | Future masking implemented; forecasting ratios reported |
| C5 | Static frames OOD | F1.4.7 | Augmentation on; BBBC054 rerun |
| C6 | Weak hand-crafted baseline | F3 (E1 improved) | Cellpose or proper segmentation; motility features on video |
| C7 | Statistical floors / paired t / 4 conditions | §9, 7.4, 7.5 | Adequate permutations; moderated tests; descriptive framing |
| C8 | Circular one-axis analysis | 8.5 | Absent from code, results, paper |
| B1 | LPS anchor donor-confounded | 8.1 | Donor-matched anchors; sanity rule results reported |
| B2 | Rest vs restimulation | 8.2, 8.3 | Framing in text; restimulation test run or "no qualifying dataset" |
| B3 | Macrophage type mismatch | 8.1.2 | Matched differentiation used or mismatch stated |
| B4 | "Acute endotoxin tolerance" | 7.2, 10.1 | Wording present in abstract, results, methods |
| B5 | Cell-type mismatch; shape ≠ activation | F1.4.3, F3.2.5, 10.1 | PhagoStat added; wording; limitation paragraph |
| B6 | Discussion overclaim | 10.1 | Claims checker passes |
| E1 | Table header mismatch | 10.1 | Table writer assertion |
| E2 | §2.6 strings, repeat, missing Methods | 10.1 | Methods 4.1–4.11 present; no raw status strings (grep test) |
| E3 | References | 10.4 | Every listed DOI resolved, or TODO(verify) |
| E4 | Figures | 10.3 | All 8 figures regenerated and visually inspected; no empty panels |
| E5 | Hard-coded conclusions | 10.2 | Claims checker in `run_all.sh` passes |
| E6 | Reproducibility gaps | 11.1–11.4 | Lockfile, single torch version, relative paths, tests |
| E7 | Repo hygiene | 11.5 | No tracked file > 5 MB; stale comment fixed |
| O1 | Weights only on instance disk | F0.1, F9.1 | `reports/BACKUP.md` with matching md5s |

---

## 13. Phase F9 — Final backup, report, PR

1. **Backup to Drive** (`<remote>:TiJepa/`): all v1 final checkpoints
   (`checkpoints_v1/`), `results/`, `figures/`, `paper/`, the corpus index CSVs, and
   `requirements-lock.txt`. Verify with `rclone check --one-way`; update `reports/BACKUP.md`
   and `reports/WEIGHTS.md` with remote paths and md5s.
2. **`reports/RUN_REPORT_v1.md`**: compute and GPU-hours per run; datasets used/skipped;
   every primary and secondary result with its source file; the Section 12 table with status;
   deviations; weaknesses a reviewer would still raise; TODO(author) list.
3. **Paper:** regenerate `paper/tijepa.md` (and PDF if pandoc + LaTeX available); claims
   checker must pass.
4. `pytest -q` and `ruff check` clean; `run_all.sh` updated.
5. Commit and push in milestones (Section 14). Update PR #1 title and body (summary, results
   table, Section 12 status table, TODO(author), "squash-merge recommended").
6. Do not publish weights publicly, post to bioRxiv/Zenodo/Hugging Face, or create a GitHub
   Release.

---

## 14. Milestones, commits, time guide

| # | Milestone | Commit prefix | Guide |
|---|---|---|---|
| V0 | F0: v0 backup verified, preflight, intake | `chore(v1):` | 0.5 h |
| V1 | F1.1–F1.2 Part B data fixed + guard tests | `data(v1):` | 2–3 h |
| V2 | F1.3 leakage check | `data(v1):` | 1 h |
| V3 | F1.4 corpus v1 (runs in background) | `data(v1):` | 6–12 h wall |
| V4 | F4 Part B v1 and F5 Part C 8.1–8.3 (while corpus builds) | `partB(v1):` / `partC(v1):` | 4–6 h |
| V5 | F2 pretraining (all runs) | `partA(v1):` | 12–24 h wall |
| V6 | F3 evaluations + 8.4 C3 | `eval(v1):` | 4–6 h |
| V7 | F7 paper/figures/refs, F8 engineering | `paper(v1):` | 3–4 h |
| V8 | F9 backup, report, PR | `chore(v1):` | 1 h |

Run downloads, preprocessing and training in the background and do Parts B/C meanwhile.
Push after every milestone.

### Stop / degrade rules
- Missing author-provided data (Dryad, LTDB) → continue; mark the dependent analyses
  `not provided`, keep their code ready, and make it the first TODO(author).
- A download blocked by an access control → do not bypass; record and continue.
- Training over budget → stop at the last checkpoint and evaluate it.
- A positive control fails → do not report that analysis's results; report the failure.
- Never fill a gap with invented data, numbers or references.
