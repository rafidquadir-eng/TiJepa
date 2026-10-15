# TiJepa-0 run report

Generated 2026-10-15 21:34 UTC from `results/` at commit `062ed68082` by `scripts/91_run_report.py`. Every number below is read from a results file.

## 1. Compute

| item | value |
|---|---|
| GPU | 1x NVIDIA A100-SXM4-40GB (40960 MiB) |
| CPU | 30 logical cores |
| RAM | 226.8 GB |
| Free disk at start | 511.1 GB (all data local) |
| Tier | **T1** - exactly 1 GPU detected (40960 MiB); brief: 1 GPU -> T1 |
| Encoder / precision | ViT-S / bf16 |
| Raw-video cap | 120.0 GB (brief cap 120 GB) |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| Seed policy | single GPU: pretraining uses seed 0 only; seeds 0/1/2 used for all probes and downstream models (must be stated in Methods) |
| Pretraining | 55,000 steps, **4.43 GPU-hours**, 24.44 effective epochs, stop: `max_steps` |

## 2. Data used

| dataset | accession | licence (read from record) | role | status |
|---|---|---|---|---|
| bbbc054 | None | CC BY 3.0 Unported (commercial use allowed) | partA_eval_morphology_and_lps_time | ok |
| complexeye | 10082674 | cc-by-4.0 | pretrain | ok |
| geo_GSE172116 | GSE172116 | NCBI GEO public domain data (no licence asserted); cite the originating publication | partB_human_endotoxin_tolerance | ok |
| geo_GSE260996 | GSE260996 | NCBI GEO public domain data (no licence asserted); cite the originating publication | partB_rnaseq_mouse_bmdm_ligand_pairs | ok |
| geo_GSE261647 | GSE261647 | NCBI GEO public domain data (no licence asserted); cite the originating publication | partB_atac_mouse_bmdm_ligand_pairs | ok |
| geo_GSE46903 | GSE46903 | NCBI GEO public domain data (no licence asserted); cite the originating publication | partC_human_macrophage_activation_anchors | ok |
| geo_GSE58310 | GSE58310 | NCBI GEO public domain data (no licence asserted); cite the originating publication | partBC_human_trained_tolerant_saeed2014 | ok |
| geo_GSE85243 | GSE85243 | NCBI GEO public domain data (no licence asserted); cite the originating publication | partBC_human_trained_tolerant_novakovic2016 | ok |
| phagostat_annot | 10803492 | cc-by-4.0 | pretrain_annot | ok |
| s-biad2133 | S-BIAD2133 | MIT | frame_qc_classifier | ok |
| s-biad2196 | S-BIAD2196 | CC BY 4.0 | pretrain_and_tracking_eval | ok |
| tay_traces | 14337874 | cc-by-4.0 | partB_traces | ok |
| dryad_bmdm | doi:10.5061/dryad.m0cfxpp8v | CC0-1.0 (https://spdx.org/licenses/CC0-1.0.html), read from the Dryad API | partA_eval_headline (PLANNED, NOT OBTAINED) | unreachable |

Pretraining corpus: **288,081 clips** (42.48 GB, 42 shards) from 288/302 movies; 14 movies failed on corrupt zip members (bad CRC / magic number) in archives whose transfer had been interrupted, and were excluded by the integrity check rather than partially decoded.

Datasets the plan excluded on licence grounds (Zenodo 16319732 CC BY-NC-ND, DeepCell DynamicNuclearNet, the Cellpose training set, CTMC-v1, LIVECell, Cell Tracking Challenge) were never downloaded or used.

## 3. Headline results

| analysis | result | source file |
|---|---|---|
| BBBC054 morphology, E1_handcrafted_morphology | BA 0.697 +/- 0.000, 95% CI [0.678, 0.715], perm p 0.005 | `results/partA/bbbc054_probes.json` |
| BBBC054 morphology, E2_imagenet_vit_s | BA 0.755 +/- 0.000, 95% CI [0.738, 0.772], perm p 0.005 | `results/partA/bbbc054_probes.json` |
| BBBC054 morphology, E3_dinov2_vit_s | BA 0.769 +/- 0.000, 95% CI [0.755, 0.786], perm p 0.005 | `results/partA/bbbc054_probes.json` |
| BBBC054 morphology, E4_tijepa0 | BA 0.828 +/- 0.000, 95% CI [0.814, 0.843], perm p 0.005 | `results/partA/bbbc054_probes.json` |
| BBBC054 morphology, E6_tijepa0_random_init | BA 0.772 +/- 0.000, 95% CI [0.756, 0.790], perm p 0.005 | `results/partA/bbbc054_probes.json` |
| BBBC054 LPS-time, E1_handcrafted_morphology | Spearman rho -0.000, MAE 8.03 h (mean-predictor 7.89 h) | `results/partA/bbbc054_probes.json` |
| BBBC054 LPS-time, E2_imagenet_vit_s | Spearman rho 0.197, MAE 7.64 h (mean-predictor 7.89 h) | `results/partA/bbbc054_probes.json` |
| BBBC054 LPS-time, E3_dinov2_vit_s | Spearman rho 0.277, MAE 7.41 h (mean-predictor 7.89 h) | `results/partA/bbbc054_probes.json` |
| BBBC054 LPS-time, E4_tijepa0 | Spearman rho 0.083, MAE 7.96 h (mean-predictor 7.89 h) | `results/partA/bbbc054_probes.json` |
| BBBC054 LPS-time, E6_tijepa0_random_init | Spearman rho 0.100, MAE 7.89 h (mean-predictor 7.89 h) | `results/partA/bbbc054_probes.json` |
| Tracking, baseline_nearest_detection | <delta> 0.527 [0.417, 0.637], median survival 17 frames | `results/partA/tracking.json` |
| Tracking, E4_tijepa0 | <delta> 0.463 [0.325, 0.602], median survival 4 frames | `results/partA/tracking.json` |
| Tracking, E5_tijepa0_frame_shuffled | <delta> 0.474 [0.338, 0.607], median survival 3 frames | `results/partA/tracking.json` |
| Tracking, E6_tijepa0_random_init | <delta> 0.436 [0.306, 0.566], median survival 4 frames | `results/partA/tracking.json` |
| B3.1 tolerance (handcrafted) | BA 0.598, 95% CI [0.566, 0.633], perm p 0.30 **not significant**, 4 wells | `results/partB/b31_tolerance.json` |
| B3.1 tolerance (raw_trace) | BA 0.701, 95% CI [0.643, 0.752], perm p 0.30 **not significant**, 4 wells | `results/partB/b31_tolerance.json` |
| B3.2 ligand identity | BA 0.338 (10 classes, chance 0.100) | `results/partB/b32_ligand_identity.json` |
| B3.3 dynamics -> expression | 86 genes r>0.5 vs null mean 333.9, p 0.49 - **null** | `results/partB/b33_dynamics_to_expression.json` |
| B4 cross-species overlap | human 0 up / mouse 1 up, overlap 0 - **null** | `results/partB/b4_cross_species.json` |
| C2 LPS_tolerised | M1-like 0.0014, M2-like -0.0010 (hypothesis) | `results/partC/c2_scores.json` |
| C2 beta_glucan_trained | M1-like -0.0092, M2-like -0.0042 (hypothesis) | `results/partC/c2_scores.json` |
| C2 control | M1-like 0.0000, M2-like 0.0000 (hypothesis) | `results/partC/c2_scores.json` |
| C2 naive_monocyte | M1-like 0.0191, M2-like -0.0269 (hypothesis) | `results/partC/c2_scores.json` |
| C3 projection | **not_performed** - no M0/M1/M2 image anchors | `results/partC/c3_projection.json` |

## 4. Deviations from PLAN.md

Each item states what the plan asked for, what was done instead, and why.

| # | Plan | What happened | Why |
|---|---|---|---|
| 1 | A4.1 headline benchmark on Dryad `m0cfxpp8v` (M0/M1/M2 BMDM movies) | **Not obtained.** Part A evaluation rests on BBBC054 instead | All official Dryad endpoints return 401/403; the file-stream endpoint serves an Anubis proof-of-work interstitial. Solving it is bypassing an access control (PLAN §2, §9) |
| 2 | A4 six-encoder comparison including E5 frame-shuffle | E5 is **N/A** on BBBC054 and is run only on the tracking task | BBBC054 is static: shuffling identical frames is a no-op. With Dryad unreachable, no dataset in this run has activation-labelled video |
| 3 | C3 projection into image space | **Not performed**; a clearly-labelled reduced one-axis analysis is reported instead | C3 needs image-space M0/M1/M2 centroids, which only the unreachable Dryad set provides. Substituting BBBC054's microglial classes would not be the same analysis |
| 4 | A1.2 segmentation with Cellpose `cyto3` | Classical detector (temporal-median background, MAD threshold, area filter) + trackpy | The labelled sets need no segmentation (BBBC054 ships coordinates, S-BIAD2196 ships tracked masks), so this kept the GPU free for training. Permitted by A1.2 |
| 5 | A1.4 frame-QC classifier from S-BIAD2133 | **Not trained**; no QC filtering applied, removed-fraction reported as N/A | Only 5 of 8,441 files retrieved: the BioStudies files API returns ~5 rows per call regardless of `pageSize`, needing ~85 paginated calls plus thousands of small-file downloads |
| 6 | A4.1 "C tuned by inner CV" | C tuned once on the full design, then held fixed inside permutation refits | Re-tuning inside each of 200 group-permutation refits is computationally prohibitive; recorded in the results JSON |
| 7 | 8.1 permutation tests at 1,000 permutations | 200 permutations | Runtime. The binding limit on these tests is the number of distinct group-label assignments, not the permutation count (see #9) |
| 8 | A1.1 resample everything to 0.65 um/px | ComplexEye/S-BIAD2196 at 0.641 um/px used unresampled (factor 0.986); **BBBC054 crops not resampled at all** | BBBC054 states a 20x objective but no pixel size, so no factor could be computed. Recorded in the results JSON |
| 9 | B3.1 significance testing | Reported, but **not significant** | The contrast has 4 wells, so a well-level permutation null admits only a handful of distinct assignments and p cannot fall below ~1/6 regardless of effect size. This is a design limit of the deposit, not of the test |
| 10 | B4 "DESeq2-style via pydeseq2" | Paired t-test on donor-matched samples + BH, with an expressed-gene prefilter | The deposited matrices are processed log-scale values, not raw counts, so a count-based model does not apply. GSE58310 is donor-paired and an unpaired test discards that |
| 11 | B3.3 / B4 on LPS tolerance | Cover **heterologous** priming only | The deposited Tay RNA-seq has no LPS->LPS sample, so the homologous contrast B3.1 classifies has no matching transcriptome |
| 12 | 4.1 pretraining data cap 120 GB | 7 of 21 ComplexEye archives (44.23 GB) | Transfer time. The cap and the 60%-of-free-disk limit were never reached |
| 13 | Pretraining seeds 0/1/2 | **Seed 0 only** | Single GPU, per the run brief. Seeds 0/1/2 are used for all probes and downstream models |
| 14 | Weights to a draft GitHub Release | **Not published** | Publishing is outward-facing and was not authorised for this run; `reports/WEIGHTS.md` has instructions |
| 15 | A4.3 CoTracker and LTDB | Neither run | `dl.fbaipublicfiles.com` and `ndownloader.figshare.com` were unreachable in preflight |
| 16 | 4.1 ComplexEye record `10.5281/zenodo.7962144` | Used v2 record `10082674` | The 7962144 API endpoint returns no JSON; all licence and size checks were read from 10082674 |
| 17 | Environment | torch upgraded 2.11.0+cu128 -> 2.14.1+cu130 mid-run | Installing the imaging stack pulled a newer torch. CUDA was re-verified on the A100 before training |

## 5. Weaknesses a reviewer would raise

1. **The pretraining corpus is single-source**: one cell type (human neutrophils), one modality, one lab. Claims about "innate immune activation" representations should be read with that in mind.
2. **No activation-labelled video was evaluated.** The only labelled probe is static, so the central claim that temporal modelling helps is tested only on tracking, not on state classification.
3. **Part B's tolerance result is not significant** and rests on 4 wells.
4. **Parts B3.3 and B4 are null**, and both are underpowered by construction.
5. **Part C is a cross-species, cross-platform interpolation between three anchor conditions** with effect sizes in the hundredths of a rank-scale unit.
6. **S_LPS does not recover canonical acute LPS genes** at 72 h, weakening the LPS half of the M1-like axis (S_IFNg does recover canonical genes).
7. **No QC filtering** was applied to pretraining clips.
8. **Single pretraining seed.**

## 6. For the author - TODO list

- [ ] `TODO(author)`: confirm the affiliation in `paper/tijepa.md`.
- [ ] `TODO(author)`: review the AI-assistance disclosure.
- [ ] `TODO(author)`: declare competing interests.
- [ ] `TODO(author)`: choose the repository licence (currently Apache-2.0 for code, CC BY 4.0 for text/figures); say so if a non-commercial licence is preferred.
- [ ] `TODO(verify)`: resolve the DOIs/arXiv ids marked in `paper/tijepa.md` references before posting.
- [ ] Fetch Dryad `m0cfxpp8v` by hand (`reports/MANUAL_DOWNLOADS.md`) and re-run `scripts/60_partA_eval.py` and `scripts/41_partC_projection.py` to obtain the planned headline benchmark and the C3 projection.
- [ ] Optionally fetch LTDB and complete S-BIAD2133 for the QC classifier.
- [ ] Publish weights per `reports/WEIGHTS.md` if desired.
- [ ] **Nothing has been posted to bioRxiv, Zenodo or Hugging Face**, per PLAN §1.4.3.

