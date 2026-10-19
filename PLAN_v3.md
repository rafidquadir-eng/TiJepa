# TiJepa — Plan v3: from "a careful pipeline" to "a result worth reading"

> **For the executing agent.** This plan supersedes the remaining steps of `PLAN_v2.md`
> (§9 step 5 onward: figures, paper, run report). Everything already produced under
> `results/v2/` is kept and reused. Read this file end to end, then `PLAN_v2.md`,
> `PLAN_v1.md` §1 and `PLAN.md` §1 (their rules still apply). Execute phases H0 → H9 in
> order, respecting the **decision gates** (§2). Section 14 maps every finding (R1–R24) to a
> fix and an acceptance check; the v3 run report must reproduce it with a status and an
> evidence file per row.
>
> **Read §0A first. It is the author's scope decision of 2026-10-18 and overrides any
> conflicting instruction elsewhere in this file.**

---

## 0A. Scope and venue — author decision (overrides the rest of this plan where they conflict)

### 0A.1 Goal of the paper
Use JEPA self-supervised learning on **unlabeled** live-cell video and label-free imaging of
innate immune cells, and test — **with public data only** — how much molecular, functional
and memory state can be inferred from those representations, across as many of the
following modality families as public data allow: transcriptomics, epigenomics,
proteomics/signalling (incl. cytokines), metabolism, immunophenotyping, integrated
multi-omics. Trained immunity is the motivating biology; the paper states plainly that no
public image or video of trained cells exists, so trained immunity itself is addressed
through the closest public proxies (acute priming/tolerance, polarisation, restimulation
datasets) and is never claimed to be detected in images.

### 0A.2 What is in and out of scope
- **In:** every openly accessible dataset. The agent downloads directly to the instance,
  uses it, may delete raw files afterwards, and records accession, licence, files used and
  checksums in `manifests/datasets.yaml` and the paper's data-availability section.
- **Dryad** (`m0cfxpp8v` BMDM movies; `8gtht76q2` human macrophage FLIM) is public but its
  download pages require a browser (bot check — never bypass it). Check
  `data/raw/manual/` and Drive for files the author copied; if the author supplies Dryad API
  credentials, use the official API. Otherwise take the "no" branch for that dataset.
- **Out:** data requests to labs, new wet-lab experiments. **H6 shrinks** to a ≤ 1-page
  "Future work: the experiment that would test this directly" subsection of the paper (no
  separate design document, no power analysis beyond one paragraph).
- **H3.4 (full TiJepa-2 retrain) is removed** for speed. H3.1–H3.3 (validation task,
  collapse diagnosis, 5 × 20k-step sweep in the background) stay; if a variant passes
  G-Recipe, report it as a finding and its 20k-step model as an extra row, not as a new
  headline model.

### 0A.3 New track: modality coverage map (phase H5.4 — do it)
Build one table and one figure that are central to the paper: for each modality family in
0A.1, the best public dataset pairing images/video (or image-derived dynamics) with that
modality, the pairing level (same cell / same condition / cross-study), what was tested,
and the result — including "no public pairing exists" rows. Tests to run (frozen encoders
E2, E3, E4, E4s, E0, E7 where applicable; donor/batch/condition-grouped CV; small-n rules):

| Modality family | Public data (verify licence, download directly) | Test |
|---|---|---|
| Immunophenotype / activation state | Dryad `m0cfxpp8v` (BMDM M0/M1/M2 video) if obtainable; BBBC054 (microglia morphology over LPS time) | Already planned (H1.2, BBBC054) |
| Immunophenotype / cell identity | Zenodo `15657646` (label-free two-photon autofluorescence of macrophages, DCs, neutrophils, T/B cells; cell-type labels) | Cell-type classification from label-free images, grouped by sample/animal |
| Metabolism | Zenodo `14330511` (raw NAD(P)H FLIM incl. macrophages); Dryad `8gtht76q2` (human M1/M2 macrophage FLIM, CD68/CD163 ground truth) if obtainable | Predict FLIM-derived metabolic readouts (mean lifetime, free/bound fraction) from the **intensity** image's frozen features; M1 vs M2 from intensity alone. Read `.sdt` with the `sdtfile` package; compute lifetimes with a documented fit |
| Signalling (proteomics/signalling family) | Tay deposit: single-cell NF-κB traces from live imaging, 125 wells | Ligand-pair decoding (B3.2, promoted); acute tolerance (B3.1, unpowered, stated) |
| Transcriptomics | Tay RNA-seq GSE260996 (condition level, same lab as the imaging); restimulation datasets GSE212282/GSE185433 (no imaging) | Dynamics → RNA (descriptive); restimulation biology incl. β-glucan TNF (H5.1) |
| Epigenomics | Tay ATAC GSE261647 (condition level) | Dynamics → accessibility (descriptive) |
| Cytokines | Restimulation datasets (RNA of TNF/IL6; protein only in the source papers) | H5.1; no imaging pairing — stated |
| Integrated / single-cell multi-omics, spatial, microbiome, clinical, repertoire, Perturb-seq | — | Row states why no public image pairing exists or why it does not apply to in-vitro innate-cell imaging |

Every dataset added here goes through the data-type guard, a positive control appropriate
to it, and the leakage check if it could overlap the pretraining corpus.

### 0A.4 Venue and manuscript format
- **Preprint:** bioRxiv (author posts after review of the draft). Non-anonymous.
- **Primary journal: TMLR (Transactions on Machine Learning Research).** Its acceptance
  criteria are that claims are supported by accurate, convincing evidence and that some of
  its audience would be interested — it does not require state-of-the-art results, which
  suits a rigorous, controlled, partly negative empirical study. Rolling submission.
  Prepare `paper/v3/tmlr/` with the official TMLR LaTeX template, **anonymised for
  double-blind review** (no author name, no affiliation, no company, repository linked via
  an anonymised mirror or "available upon acceptance"), main text focused on: what video
  JEPA learns from innate-immune video, the forecasting controls, the collapse diagnosis,
  baselines incl. a pretrained video model, and the modality coverage map. Biology detail
  goes to appendices.
- **Fallback (life-science venue, if TMLR rejects):** F1000Research Research Article
  (publication after pre-publication checks, open peer review, indexed in PubMed once
  approved) or Frontiers in Immunology (Systems Immunology). Prepare
  `paper/v3/biorxiv/` (non-anonymous, biology-forward ordering) so either can be submitted
  with light edits.
- Do not submit anything; the author submits.

### 0A.5 Speed
Target: complete v3 in ≤ 3 days of agent time. Priorities, in order: H0 → H2 (controls
decide the main claim) → H1 (V-JEPA baseline; Dryad if available) → H5.4 coverage map →
H4 association → H3.1–H3.3 (background) → H5.1–H5.2 → H7 (both manuscript versions) →
H8–H9. If time runs short, cut H4 and H3.3 before cutting anything else.

---

## 0. Executive review (state at commit `cb92e2e`)

### What is solid
- An honest, auditable pipeline: provenance-labelled results, data-type guards, positive
  controls, small-n statistics without fake intervals, sourced biological expectations,
  a complete leakage check (1 duplicate, handled), errata for every past mistake.
- Two models trained to an annealed finish on a frozen corpus: **TiJepa-1** (mixed spatial
  + future masking, 117,000 steps, 11.9 GPU-h) and a matched-step **spatial-only ablation**
  (117,000 steps, 9.2 GPU-h).
- Part C now uses sourced expectations and restimulation positive controls (both pass).

### What the results actually say
| Question | Answer from the data so far |
|---|---|
| Does TiJepa-1 learn better cell-state features than strong image models? | **No.** BBBC054 three-class: DINOv2 0.610, TiJepa-0 0.607, TiJepa-1 0.595, ImageNet 0.590 — not distinguishable. TiJepa-1 beats its spatial-only ablation by only +0.013 (p = 0.03) and random init by +0.041. Round-vs-amoeboid: TiJepa models 0.82 vs DINOv2 0.78 |
| Did 2× more training help over TiJepa-0? | **No.** TiJepa-1 (117k steps) ≈ TiJepa-0 (55k steps) |
| Is the representation healthy? | **Concerning.** Effective rank fell monotonically 15.8 → 8.1 (TiJepa-1) and → 7.6 (ablation); embedding std 0.25 → 0.04; the monitor probe peaked at ~34k steps and then declined. Partial dimensional collapse |
| Does TiJepa-1 forecast? | **Unverified.** Its predictor error is 0.17× "copy last latent" (ablation 0.85×), p = 0.031 over 6 movies — but no control rules out that it is predicting the *temporal-position* component of target latents rather than cell dynamics, and the horizon is 32 s on mostly stride-1 clips |
| Do features help tracking? | **The test cannot tell.** Frame-shuffled (0.635) = TiJepa-1 (0.634) ≈ random init (0.62); median survival 1 frame vs 20 for the baseline |
| Is there an activation-labelled video result? | **No — Dryad has never been provided** (third run in a row). This is the single biggest gap and only the author can close it |
| Part B | Tolerance from dynamics is unpowered by design (4 wells). Ligand-pair identity *is* decodable (0.27 vs chance 0.09, 125 wells) — the one solid Part B positive. Dynamics→omics and cross-species overlap are null |
| Part C | Restimulation datasets pass positive controls. **β-glucan-trained macrophages show lower TNF mRNA after 4 h LPS in all 3 donors (−2.1 log2)**, opposite to the 24 h protein result their source paper reports. Interesting if real; unverified against label swaps and kinetics |

### Strategic verdict
The engineering is now trustworthy; the **science is not yet a story**. Three loosely
connected parts with mostly null results will not read as a contribution. v3 does three
things: (1) closes the gaps that could flip the conclusions (Dryad, forecasting controls,
representation collapse, a proper video-model baseline); (2) turns the two genuinely
interesting threads (forecasting, the β-glucan TNF kinetics) into verified findings or
honest nulls; (3) reframes the paper around what is actually contributed — a public-data
audit, an open, rigorously evaluated video SSL model of innate-cell behaviour, and a
**pre-registered, powered experimental design** for the imaging study the field lacks.

---

## 1. Rules (in addition to PLAN_v2 §1, PLAN_v1 §1, PLAN.md §1)

1. **Selection is never done on reported test sets.** Recipe choices (H3) are made on
   validation data that never appears in a reported result (§6.2). Every reported number
   comes from a model frozen before its test set was touched.
2. **Decision gates are binding** (§2). If a gate fails, take the branch the gate names;
   do not run the more expensive step "to see".
3. **Keep training snapshots.** Any new training run keeps immutable snapshots at every
   10,000 steps (`checkpoints/<run>/step_XXXXXX.pt`), in addition to the rolling resume
   checkpoint.
4. **Cost hygiene.** No background loop may run on the GPU instance once its purpose is
   done (the v2 pretrain monitor is still writing rows at step 117,000 every 2 h). When the
   GPU will be idle for > 2 h, say so in the PR body so the author can stop the instance.
5. **Backups are part of "done".** A milestone that produces a checkpoint is not complete
   until the checkpoint exists in a second, persistent location with a verified hash.

---

## 2. Decision gates

| Gate | Condition | If yes | If no |
|---|---|---|---|
| **G-Dryad** | Dryad files present and checksum-verified (H1) | Dryad becomes the Part A headline (H1.2) | Part A headline stays BBBC054; the paper states the gap in one sentence; nothing else changes |
| **G-Forecast** | TiJepa-1 beats copy-last **and** the slot-conditional baseline **and** loses most of its advantage when its context is swapped/static (H2) | Forecasting is reported as a finding, with the controls | Forecasting is reported as "explained by temporal-position structure"; the paper drops the forecasting claim |
| **G-Recipe** | A recipe variant in the H3 sweep improves the validation score (§6.2) by ≥ 0.02 **and** at least doubles feature effective rank vs the baseline recipe at the same step | Report the variant and its 20k-step model as an extra row (§0A.2: no full TiJepa-2 retrain) | Stop recipe work; report the collapse diagnosis as a limitation; evaluate existing models only |
| **G-BetaGlucan** | The β-glucan TNF result survives sample-identity QC and is consistent in at least one independent dataset or in the source paper's own RNA data (H5) | Report it as a replicated transcript-level kinetics observation | Report it as a single-dataset observation (if QC passes) or withdraw it (if QC fails) |

---

## 3. Author actions (only the author can do these) — see also §0A.2 (the agent downloads all open data itself; only Dryad needs the author)

1. **Dryad — copy it straight to the instance (no Drive needed).** On your laptop, open
   `https://datadryad.org/dataset/doi:10.5061/dryad.m0cfxpp8v`, click *Download dataset*,
   then:
   `scp ~/Downloads/<the dryad zip> ubuntu@<instance-ip>:~/proj2/data/raw/manual/`
2. **Backups.** Either re-authorise rclone (`rclone config reconnect gdrive:` → answer *n*
   to auto config → run the printed `rclone authorize "drive" ...` on a computer with a
   browser → paste back), or allow the agent to treat the Lambda persistent filesystem as
   the second location until Drive works.
3. **PR housekeeping (optional, recommended).** Squash-merge PR #1 into `main` whenever
   convenient; v3 works on a new branch (H0.4), so the merge does not disturb it.

---

## 4. Phase H0 — Safety, cost, housekeeping

1. **Stop the stale monitor loop** (`scripts/53_pretrain_monitor_v2.py` /
   `98_backup_loop_v2.sh` or whatever is still writing `results/v2/pretrain_monitor.csv`);
   deduplicate the repeated step-117,000 rows; commit.
2. **Back up the final checkpoints now** (`tijepa1_T1_seed0.pt`, `tijepa1_spatial_T1_seed0.pt`,
   their log CSVs, `tijepa0_T1_seed0.pt`):
   - to the Lambda persistent filesystem (`LambdaFilesystem/.../tijepa_backups/`), verify
     SHA-256;
   - to Drive via rclone if the author re-authorised it, verify md5;
   - record both in `reports/BACKUP.md`. Never print or commit credentials.
3. **Finish v2 cleanly:** commit any uncommitted v2 outputs. Do **not** render the v2 paper
   or figures (v3 replaces them).
4. **Branch:** create `tijepa-v3` from the current `tijepa-v0` head and push it. All v3 work
   goes there. At the end, open **PR #2** `tijepa-v3 → main`. Never force-push, never push
   to `main`.
5. **Results layout:** new outputs under `results/v3/`, figures `figures/v3/`, paper
   `paper/v3/`.

---

## 5. Phase H1 — Dryad intake and the video headline (inference only)

### H1.1 Intake
Look for the Dryad zip in `data/raw/manual/`, on the Lambda filesystem, and on Drive
(`TiJepa/manual_downloads/`). Unzip to `data/raw/dryad_bmdm/`, verify every file against the
SHA-256 values the Dryad API publishes. Write `results/v3/manual_intake.json`. If absent,
re-check once more before H7 and then take the "no" branch of G-Dryad.

### H1.2 Benchmark (if present)
Run `scripts/64_dryad_v1.py` (generalised to read encoder lists from provenance) with:
E0 TiJepa-0, E1 hand-crafted (shape + **motility** features), E2 ImageNet, E3 DINOv2,
E4 TiJepa-1, E4s spatial-only, E5 TiJepa-1 frame-shuffled, E6 random init (3 draws),
E7 frozen V-JEPA (H1.3). Leave-one-batch-out over 5 batches, cell- and movie-level metrics,
cluster bootstrap over the 18 movies, movie-level permutation, batch and magnification
confound probes, results stratified by magnification. Also report the accuracy the source
paper (Sun & Kesapragada) reports, with any task differences. This is the primary Part A
endpoint if G-Dryad passes.

### H1.3 Frozen pretrained video baseline (E7)
Add a frozen V-JEPA-family encoder (prefer V-JEPA 2 ViT-L; else V-JEPA ViT-L) as E7 on every
Part A benchmark (Dryad, BBBC054, association, forecasting is N/A). **Check and record the
weights licence first** (`manifests/models.yaml`); a non-commercial licence is acceptable for
this non-commercial preprint only if stated in Methods. Inputs: replicate grayscale to 3
channels, resize crops to the model's native resolution, 16 frames (static images: repeat
the frame). If weights cannot be downloaded (host blocked), record and continue. This answers
the first question a reviewer will ask: *does training on cell video beat a general-purpose
video model?*

---

## 6. Phase H2 — Verify (or retract) the forecasting result (inference only)

The current metric compares each model's predictor with "copy the last context latent".
Target latents at future slots also carry **temporal-position** information that the
predictor receives through its positional embeddings, so a model can beat copy-last without
modelling any cell dynamics. Run, on the 6 S-BIAD2196 movies and on held-out ComplexEye
movies (H3.1), for TiJepa-1, the spatial ablation and TiJepa-0:

1. **Slot-conditional mean baseline:** for each future slot, the mean target latent at
   that slot over all evaluation clips (fit on half the movies, scored on the others).
2. **Context-swap control:** run the predictor with the context tokens of a *different*
   clip (same movie, random) and the true clip's targets. A dynamics model must get much
   worse; a position model will not.
3. **Static control:** clips made from one frame repeated 16 times. A dynamics model's
   advantage over copy-last should vanish.
4. **Horizon and stride:** report skill separately for stride 1, 3, 9 and 27 clips (8 s to
   3.6 min frame interval) so the horizon spans 32 s to ~14 min.
5. **Pixel-space sanity:** linear decoder from target latents to the next frame's
   downsampled image (fit on half the movies); compare decoded predictions vs copy-last in
   pixel space.

Report a table: ratio vs copy-last, ratio vs slot-mean, context-swap degradation,
static-clip ratio, per stride. Apply **G-Forecast**.

---

## 7. Phase H3 — Diagnose the collapse; improve the recipe only if a gate says so

### H3.1 Validation data that is never used as a test set
- Download **one ComplexEye archive not in the frozen corpus** (prefer `R01.zip`; MD5-
  verify) → `data/raw/complexeye_val/`. Run the leakage fingerprint against S-BIAD2196 and
  against the corpus; drop any duplicate.
- Build a **label-free validation task**: detect and track cells with the existing
  classical pipeline; for each tracked cell at time t, retrieve the same cell at t+k
  (k = 3, 9, 27 frames) among all cells in frame t+k by embedding cosine similarity.
  Score = top-1 retrieval accuracy (re-identification). Also compute feature effective rank
  on these clips.
- **Validation score** = mean re-ID top-1 over k ∈ {3, 9, 27}. This, plus rank, is the only
  thing used for recipe selection.

### H3.2 Diagnose with what exists (no training)
- Effective rank, per-dimension std and singular-value spectra of **encoder outputs**
  (context encoder and EMA target, pooled and per-token) for TiJepa-0, TiJepa-1, spatial
  ablation and random init, on the validation clips, BBBC054 crops and (if present) Dryad.
- Correlate rank with BBBC054 and validation scores across models.
- Write `results/v3/collapse_diagnosis.json`: is the collapse in the encoder or only in the
  monitored embedding? Is it driven by the EMA reaching 1.0, by the target LayerNorm/L1
  objective, or by the corpus (53 epochs over 280k highly similar clips)?

### H3.3 Short recipe sweep (GPU, ~10–12 h total on one A100)
Each run: 20,000 steps from scratch, frozen corpus, seed 0, snapshots every 5,000 steps,
p_future 0.5 unless stated. Variants:

| ID | Change vs the TiJepa-1 recipe |
|---|---|
| S0 | none (baseline, reproduces the first 20k steps' recipe; its schedule ends at 20k) |
| S1 | EMA momentum 0.996 → **0.9995** (never reaches 1.0) |
| S2 | add a variance–covariance regulariser on the predictor outputs (VICReg-style, small weight, e.g. 0.01–0.04 on variance, 0.001 on covariance) |
| S3 | batch 256 with linearly scaled LR (gradient accumulation if memory requires) |
| S4 | longer forecast horizon in training: future targets drawn preferentially from stride-9/27 clips; predictor depth 8 |

Score every snapshot on the validation score and rank. Apply **G-Recipe** against S0 at
20k steps.

### H3.4 If G-Recipe passes: TiJepa-2 — **REMOVED by §0A.2** (kept below for the record only)
Train the winning recipe to the full step budget (117,000) **and** its matched spatial-only
ablation with the same recipe (one GPU: sequential, ~24 h; two GPUs: parallel). Keep
snapshots. Back up per rule 5. Evaluate both on every Part A benchmark. TiJepa-1 remains in
the paper as the earlier recipe; no cherry-picking: every trained model appears in the
results table.

### H3.5 If G-Recipe fails
No more training. The collapse diagnosis becomes a limitation paragraph with the spectra
figure.

---

## 8. Phase H4 — Replace the tracking evaluation with one that can answer the question

Point tracking by patch correlation is insensitive to the features (shuffled = trained =
random). Replace the **primary** tracking metric with **detection-level association**:

- Use S-BIAD2196 ground-truth instance masks. For frame pairs (t, t+k), k ∈ {1, 3, 9},
  embed every ground-truth cell (crop + clip ending at t), match identities with the
  Hungarian algorithm on (a) embedding cosine only, (b) position only (baseline),
  (c) embedding + position.
- Metric: association accuracy (fraction of correct identity matches) and IDF1 over the
  movie. Exact sign-flip over the 6 movies (no CI below 10 groups).
- Encoders: all of E0–E7. E5 (frame-shuffled) is now meaningful because the clip ending at
  t carries motion history.
- Keep the old point-tracking numbers in a supplementary table, labelled as insensitive to
  the features, with the reason.
- Attempt CoTracker via Hugging Face (`facebook/cotracker3`) as a reference tracker; record
  the licence; skip if unreachable.

---

## 9. Phase H5 — Biology: make Part B and Part C answer something

### H5.1 β-glucan TNF kinetics (GSE212282)
1. **Sample-identity QC:** sex genes (XIST, RPS4Y1, DDX3Y, KDM5D, UTY) per sample to confirm
   donor consistency across arms; sample–sample correlation clustering; check for swapped
   arm labels (does each "β-glucan" sample sit with its donor's other arms, and does any
   arm label look like another arm's profile?). Record in `results/v3/partC/sample_qc.json`.
2. **Source paper's own RNA data:** Ferreira et al. 2023 (Cell Rep 42:112658) Figure 2
   reports RNA-seq of the restimulated macrophages. Extract what it shows for β-glucan vs
   RPMI after LPS (TNF, IL6, or the global LPS-response genes). Quote it.
3. **Replication search:** search GEO for other public datasets with β-glucan (or BCG)
   training, a ≥ 5-day rest, and an LPS (or other TLR) restimulation with bulk RNA at a
   stated timepoint and ≥ 3 donors (candidates in the landscape report: GSE242947,
   GSE85246 family, GSE141656; verify each). Apply the same positive control and data-type
   guard. Score TNF/IL6/IL1B.
4. **Kinetics framing:** the source reports protein at 24 h; the RNA is at 4 h. State the
   hypothesis plainly (trained cells may reach the TNF transcript peak earlier and resolve
   faster) **as a hypothesis**, and name the experiment that tests it (a 0.5–24 h
   restimulation time course with paired RNA and protein).
5. Apply **G-BetaGlucan**.

### H5.2 Part B, refocused
- Make **ligand-pair identity from dynamics** (B3.2) a primary Part B result: add the trace
  JEPA encoder vs hand-crafted features vs raw trace, grouped by well, 125 wells (CIs valid).
- **Power analysis for tolerance-from-dynamics:** from the observed per-well effect and
  between-well variance, compute how many wells per arm would give 80% power at α = 0.05.
  Report it; it is a concrete recommendation to the field.
- **B4 symmetric definition:** the human set should mirror the mouse set — genes LPS-induced
  in untrained cells (GSE185433 untrained restim vs untrained unstimulated, if available)
  **and** blunted in tolerised cells; restrict the universe to orthologs LPS-induced in both
  species. Keep exploratory.

### H5.3 Drop what cannot carry weight
B3.3 (dynamics → omics, 5 conditions) and the resting-state Part C scores move to the
supplement as descriptive. C3 projection stays "not performed" unless Dryad and a
stimulus-matched omics anchor both exist (they do not, today).

---

## 10. Phase H6 — The pre-registered experiment — **REDUCED by §0A.2** to a ≤ 1-page Future-work subsection of the paper; the items below are guidance for that subsection, not a separate deliverable

Write `reports/EXPERIMENT_DESIGN.md` — the imaging study that would actually test whether
trained immunity has an imaging phenotype. It must be specific enough to execute and fund:

1. **System:** primary human CD14+ monocytes; arms: RPMI, β-glucan (training), LPS
   (tolerance), IFN-γ and IL-4 polarisation anchors; 24 h exposure → wash → 5-day rest →
   LPS restimulation.
2. **Imaging:** label-free brightfield/phase time-lapse during rest and 0–24 h after
   restimulation (frame interval chosen from the H2 horizon analysis); NAD(P)H/FAD
   autofluorescence (FLIM if available) at days 0, 1, 3, 6 — the glycolytic shift is the
   best-supported image-visible correlate of training.
3. **Ground truth:** TNF/IL-6 protein (ELISA) at 4 h and 24 h after restimulation, bulk
   RNA-seq at 4 h, ATAC at day 6, lactate.
4. **Power:** donors computed from the observed between-donor SD of TNF log2 differences in
   GSE212282/GSE185433 and from the published drop of label-free classifiers to 72–85% across
   donors; state the effect size assumed and the resulting donor count.
5. **Analysis plan, fixed in advance:** frozen TiJepa (and DINOv2, V-JEPA) features →
   donor-held-out classifiers (naive / trained / tolerant); prediction of 24 h TNF protein
   from pre-restimulation video; primary endpoints, α, and what result would falsify the
   hypothesis.
6. **Deposit plan:** raw images to BioImage Archive (CC BY), sequencing to GEO, cytokines and
   metadata alongside.
7. **Cost and timeline** at the level of donors × arms × imaging hours.

---

## 11. Phase H7 — Paper v3, figures, report

### H7.1 Narrative (rewrite the template around what is contributed)
Working title (adjust to results): *"Can imaging see innate immune memory? A public-data
audit, an open self-supervised video model of innate-cell behaviour, and a pre-registered
test"*. Structure:
1. The gap (landscape audit — short, quantitative).
2. TiJepa: model, data, rigorous evaluation (Dryad headline if present; BBBC054; association;
   forecasting with its controls; comparison to DINOv2 and V-JEPA; collapse diagnosis).
3. What existing molecular data say about memory states that imaging would need to detect
   (ligand decoding from dynamics; restimulation kinetics incl. β-glucan TNF; power).
4. The pre-registered experiment (H6), as the paper's forward-looking contribution.
5. Limitations (cell-type chain, single seed per recipe, deposit-limited power).

Rules: every directional phrase through the claims renderer; provenance-derived labels;
banned-strings check; no internal process language; abstract written last from final
results; null results stated plainly.

### H7.2 Figures (`figures/v3/`)
(1) Gap + design schematic; (2) model + training curves incl. rank/std over steps for all
runs; (3) Part A headline (Dryad or BBBC054) dot-and-CI with all encoders incl. V-JEPA;
(4) association + forecasting with controls; (5) Part B ligand decoding + power curve;
(6) Part C restimulation per donor incl. sample QC and kinetics schematic; (7) the
pre-registered design. Inspect every PNG.

### H7.3 Report
`reports/RUN_REPORT_v3.md` generated after everything ran: compute per run (incl. the sweep),
gates and which branch was taken, results with source files, Section 14 table, deviations,
TODO(author).

---

## 12. Phase H8 — Engineering consolidation

1. **One pipeline, not three.** Create `pipeline/` (or a `tijepa` CLI) with versionless stage
   scripts driven by configs (`configs/v3/*.yaml`); move superseded `*_v1.py`/`*_v2.py`
   scripts that are no longer called to `legacy/` with a README mapping old → new. Results of
   earlier versions stay where they are.
2. `scripts/run_all_v3.sh` reproduces v3 exactly from raw data + frozen corpus + checkpoints.
3. Repo-relative paths in every results file (pretrain summaries still contain
   `/home/ubuntu/proj2/...`).
4. Tests: forecasting controls (synthetic latent sequences where position-only and dynamics
   models are separable), association metric, re-ID validation task, snapshot retention,
   gate logic, sex-gene QC, power calculation.
5. `pytest -q`, `ruff check` clean. No tracked file > 5 MB.
6. `README.md` rewritten for v3: what TiJepa is, headline results table, how to reproduce,
   data licences, limitations, link to the experiment design.

## 13. Phase H9 — Finish
1. Back up every new checkpoint/snapshot and final outputs (rule 5); update
   `reports/BACKUP.md`, `reports/WEIGHTS.md`.
2. Push; open PR #2 `tijepa-v3 → main` with the run-report summary, the gate outcomes,
   Section 14 status table and TODO(author).
3. State in the PR body whether the GPU is idle (rule 4).
4. Do not publish weights publicly, post anywhere, or create Releases.

---

## 14. Findings → fixes (reproduce with Status + Evidence in RUN_REPORT_v3)

| ID | Finding | Fix | Acceptance check |
|---|---|---|---|
| R1 | Dryad never provided; no activation-labelled video result in three runs | §3.1 (scp), H1 | Dryad results for all encoders, or G-Dryad "no" branch recorded |
| R2 | Final checkpoints (TiJepa-1, ablation) not backed up off the instance (rclone token missing) | H0.2 | Two verified locations in `BACKUP.md` |
| R3 | Pretrain monitor loop still running after training (wasted GPU/instance time; duplicate rows) | H0.1 | Process stopped; CSV deduplicated |
| R4 | Partial dimensional collapse: effective rank 15.8 → 8.1, std 0.25 → 0.04; monitor probe peaked at ~34k steps | H3.2–H3.5 | `collapse_diagnosis.json`; G-Recipe decision recorded |
| R5 | No intermediate snapshots kept, so step-wise analysis of existing runs is impossible | Rule 3 | Snapshots present for every new run |
| R6 | Forecasting advantage (0.17× copy-last) lacks controls for temporal-position structure, static input and horizon | H2 | Control table; G-Forecast decision recorded |
| R7 | No pretrained **video** baseline (only image models) | H1.3 | E7 rows on every Part A benchmark or recorded unavailability; licence recorded |
| R8 | TiJepa-1 ≈ DINOv2 ≈ TiJepa-0 on BBBC054; 2× training gave nothing | H3, H7.1 framing | Reported plainly; recipe study outcome |
| R9 | Point-tracking metric insensitive to features (shuffled = trained = random; survival 1 frame) | H4 | Association/IDF1 results; old metric demoted with reason |
| R10 | No validation data separate from test sets for model selection | H3.1, rule 1 | Re-ID validation on a non-corpus, non-test archive |
| R11 | Forecasting horizon effectively 32 s on mostly stride-1 clips | H2.4, S4 | Per-stride results |
| R12 | BBBC054 hand-crafted baseline still uses classical segmentation (Cellpose never installed); scale sensitivity shows accuracy rising with zoom (0.52 → 0.62) | H1.2 E1 with Cellpose if installable; report scale result in Methods | Segmentation method recorded; scale finding stated |
| R13 | β-glucan TNF −2.1 log2 at 4 h in all 3 donors, opposite to the source's 24 h protein; not checked for label swaps or replication | H5.1 | `sample_qc.json`; source RNA quote; replication attempt; G-BetaGlucan |
| R14 | Part B's only solid positive (ligand identity) is treated as secondary | H5.2 | Promoted with trace-JEPA comparison and valid CIs |
| R15 | Tolerance-from-dynamics unpowered; no quantitative statement of what would be needed | H5.2 | Power analysis reported |
| R16 | B4 human set asymmetric to the mouse set (all tolerised-down genes vs LPS-induced-and-blunted) | H5.2 | Symmetric definition and universe |
| R17 | Paper is three disconnected parts; the "bridge" (C3) has never been performed | H7.1 | Restructured narrative; C3 status stated once |
| R18 | No forward-looking contribution that converts the gap into a plan | H6 | `EXPERIMENT_DESIGN.md` with power and pre-specified analysis |
| R19 | v1/v2 scripts proliferate (`62_bbbc054_v1.py` produces v2 results); hard to reproduce | H8.1–2 | Versionless pipeline + `legacy/` map + `run_all_v3.sh` |
| R20 | Absolute paths remain in results (`/home/ubuntu/proj2/...`) | H8.3 | Grep finds none in `results/` |
| R21 | PR #1 still titled "TiJepa-0" and spans four versions | H0.4, H9.2 | PR #2 from `tijepa-v3`; author squash-merges PR #1 |
| R22 | Monitor probe used BBBC054 (a test set) during training | Rule 1, H3.1 | New runs monitor on the validation task only |
| R23 | Cell-type chain (neutrophil pretraining → microglia/macrophage evaluation → human monocyte biology) remains the core conceptual limitation | H1 (macrophage video), H6, H7.1 | Limitation stated; experiment design addresses it |
| R24 | Results depend on one pretraining seed per recipe | H3.4 (if ≥ 2 GPUs add seeds), H7.1 | Seed count stated; extra seeds if hardware allows |

---

## 15. Order, time and GPU budget (one A100)

| # | Milestone | GPU | Guide |
|---|---|---|---|
| X0 | H0 safety, backups, branch | — | 0.5–1 h |
| X1 | H1 Dryad (if present) + V-JEPA baseline | ~2 h | 3–4 h |
| X2 | H2 forecasting controls | < 1 h | 2 h |
| X3 | H3.1–H3.2 validation set + diagnosis | ~1 h | 3 h |
| X4 | H3.3 sweep (5 × 20k steps) | ~10–12 h | background |
| X5 | H4 association benchmark; H5 biology; H6 design (during X4) | light | 8–10 h |
| X6 | Gate G-Recipe → H3.4 (≈ 24 h) or skip | 0 or ~24 h | — |
| X7 | H7 paper/figures/report; H8 consolidation; H9 finish | — | 4–6 h |

Commit and push after every milestone with prefixes `chore(v3):`, `data(v3):`, `eval(v3):`,
`partA(v3):`, `partB(v3):`, `partC(v3):`, `design(v3):`, `paper(v3):`.

### Stop / degrade rules
- A gate fails → take its "no" branch; never run the expensive branch anyway.
- A download is blocked by an access control → do not bypass; record and continue.
- A positive control or sample-identity QC fails → that analysis reports the failure.
- Never fill a gap with invented data, numbers or references.
