## Results summary

- **Part A headline (BBBC054).** V-JEPA 2 ViT-L, frozen, is the best encoder (three-class
  balanced accuracy 0.681). TiJepa-1 (0.595) is not distinguishable from DINOv2, the
  ImageNet ViT or TiJepa-0; future masking adds +0.013 (CI +0.001 to +0.027) over the
  matched spatial-only ablation. Evidence: `results/v3/partA/bbbc054.json`.
- **Forecasting (G-Forecast fails).** TiJepa-1 beats copy-last (0.17x) and the
  slot-conditional mean (0.32x), but keeps 0.79 of its 0.83 advantage on static clips and
  0.51 with another clip's context. The paper drops the forecasting claim and reports the
  predictor as using clip content while its advantage over copy-last is positional.
  Evidence: `results/v3/partA/forecast_controls.json`.
- **Re-identification (validation).** V-JEPA 2 0.667 > ImageNet 0.614 > DINOv2 0.585 > raw
  pixels 0.555 > TiJepa-1 0.501 > spatial-only 0.298 > random init ~0.22.
- **Association.** Position alone (0.883 at a 9-frame gap) is not improved by any embedding;
  V-JEPA 2 + position is level with it (0.875, p = 0.25).
- **Collapse.** TiJepa-1's pooled embedding has effective rank 7.7 of 384 on held-out clips
  (context and target encoders alike; tokens 59.4); rank correlates with downstream scores
  across six networks (rho 0.88 BBBC054, 0.77 validation).
- **Recipe sweep (G-Recipe fails).** Five 20,000-step variants (S0 baseline, S1 EMA cap,
  S2 VICReg, S3 batch 256, S4 long-horizon futures; 13.4 GPU-h) scored only on the
  validation set: score changes -0.011 to +0.018 and rank ratios 0.98 to 1.31 vs S0, so no
  variant meets +0.02 and 2x rank. No run had collapsed by 20,000 steps (rank 44.5 to 59.4),
  so the sweep cannot test the late collapse; it is reported as a limitation. Evidence:
  `results/v3/partA/recipe_sweep.json`.
- **Coverage map.** Same-cell positives only for morphology classes (BBBC054) and stimulus
  identity from NF-kB dynamics; autofluorescence identity and NAD(P)H lifetime are null
  against intensity references; no public pairing for any memory readout.
- **Biology (G-BetaGlucan passes).** Lower TNF transcript at 4 h after restimulation in
  beta-glucan-trained macrophages, GSE212282 (LPS, 3/3 donors, mean -2.09 log2) and
  GSE141656 (M. tuberculosis, 3/3 independent donors, -1.96), against higher 24 h TNF-a
  protein in both source papers. Reported as a transcript-level observation with a kinetic
  hypothesis.

## Deviations from PLAN_v3

1. **Validation archive.** `384-well_plate_movies_n=1_complexeye.zip` (already on the
   instance, MD5-verified, not in the corpus) was used instead of `R01.zip`, avoiding an
   11 GB download over an intermittent Zenodo connection.
2. **Validation leakage positive control.** The self-match control reached NCC > 0.95 in
   55 of 64 movies (minimum 0.836) because stored patches are percentile-clipped to uint8;
   the best corpus match in any movie is 0.776, below every self-match. Stated as a post-hoc
   sensitivity check in `results/v3/validation_set.json`.
3. **Donor identity QC.** The pre-specified nearest-centroid rule misassigned 10 of 75
   GSE212282 samples because HLA class II and GSTM1 expression change with cell state; a
   genotype-style presence/absence call (HLA-DRB5, GSTM1) was added after seeing this and
   is the identity test used. Both are reported in `results/v3/partC/sample_qc.json`.
4. **FLIM controls added post hoc.** The condition-mean reference and the within-field
   regression were added after the first FLIM result showed lifetime differs between arms;
   they reverse the reading of that result (null), and the paper says they were post hoc.
5. **FLIM data.** The macrophage folders of Zenodo 14330511 contain raw TCSPC OME-TIFFs,
   not `.sdt` files; lifetimes were computed from the decay tail (first moment and a
   fixed-lifetime NNLS fit) without the instrument response.
6. **Autofluorescence channel order** is not documented by the deposit; channels are named
   AF-0/AF-1/Dodt.
7. **Trace JEPA** was pretrained (6,000 steps) only on the 118,243 traces from the 495 wells
   that are neither evaluation nor tolerance wells, rather than per fold.
8. **Claims checker.** The v2 sentence-level directional check was made position- and
   task-aware for the longer v3 sentences (`scripts/98_check_claims_v3.py`).
9. **Legacy.** 30 superseded scripts moved to `legacy/scripts/` with a map; a full
   config-driven `pipeline/` package was not built (the numbered stage scripts and
   `scripts/run_all_v3.sh` are the single pipeline).
10. **CoTracker** reference tracker (H4) was not attempted; association uses ground-truth
    cells, which removes the need for a reference tracker.
11. **TiJepa-0 summary overwritten** by a smoke run on 2026-10-16 (see
    `reports/V0_ERRATA.md`, V2-E1); facts come from the checkpoint and the v0 report.
12. **One commit** lacks the co-author trailer (V2-E2); history not rewritten.

## TODO(author)

1. **Drive backups:** run `rclone config reconnect gdrive:`; then re-run the Drive copy of
   `LambdaFilesystem/transfer/proj2/tijepa_backups/` and verify md5 (`reports/BACKUP.md`).
2. **Dryad** (`m0cfxpp8v`, `8gtht76q2`): copy the zips to `data/raw/manual/` if you want them
   in a later round; both manuscripts state they were not obtained.
3. **bioRxiv author block:** add your affiliation (and ORCID) to
   `paper/v3/biorxiv/main.tex`; confirm the e-mail.
4. **Code availability:** decide whether the repository becomes public; then replace the
   placeholder sentences (`\CodeAvailability` in both `main.tex`). For TMLR prepare an
   anonymised code supplement (no names, no repository URL).
5. **Read both PDFs** (`paper/v3/tmlr/main.pdf`, `paper/v3/biorxiv/main.pdf`) end to end;
   the titles and framing are yours to adjust.
6. **Licence of TiJepa weights:** weights are not published; decide if and how.
7. **Squash-merge PR #1** into `main` when convenient, then review PR #2.
8. **Submission:** nothing has been submitted or posted; the author submits.
