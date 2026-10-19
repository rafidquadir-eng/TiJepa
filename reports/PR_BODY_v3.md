TiJepa v3: a public-data audit with a video JEPA — two manuscripts (TMLR, anonymised; bioRxiv, named), figures, and every analysis behind them. Full report: `reports/RUN_REPORT_v3.md`.

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
- **Coverage map.** Same-cell positives only for morphology classes (BBBC054) and stimulus
  identity from NF-kB dynamics; autofluorescence identity and NAD(P)H lifetime are null
  against intensity references; no public pairing for any memory readout.
- **Biology (G-BetaGlucan passes).** Lower TNF transcript at 4 h after restimulation in
  beta-glucan-trained macrophages, GSE212282 (LPS, 3/3 donors, mean -2.09 log2) and
  GSE141656 (M. tuberculosis, 3/3 independent donors, -1.96), against higher 24 h TNF-a
  protein in both source papers. Reported as a transcript-level observation with a kinetic
  hypothesis.

## Decision gates

| gate | condition | outcome | branch taken |
|---|---|---|---|
| G-Dryad | Dryad files present and verified | **no** | Part A headline stays BBBC054; the gap is stated in one sentence |
| G-Forecast | beats copy-last and slot mean; swap/static remove most of the advantage | **fail** | fail: forecasting is reported as explained by temporal-position structure and the forecasting claim is dropped |
| G-Recipe | validation score +0.02 and 2x effective rank vs S0 at 20k | **pending** | sweep not finished when this report was generated |
| G-BetaGlucan | identity QC passes and replicated in an independent dataset or the source RNA | **pass** | pass: report as a replicated transcript-level observation (GSE141656, Mtb restimulation) |

## Section 14: findings → status

| ID | status | evidence | note |
|---|---|---|---|
| R1 | done ('no' branch) | `results/v3/manual_intake.json` | Dryad still absent; G-Dryad 'no' branch taken |
| R2 | done (persistent filesystem); Drive pending | `reports/BACKUP.md` | Lambda persistent filesystem verified by SHA-256; rclone has no OAuth token |
| R3 | done | `results/v2/pretrain_monitor.csv` | monitor and backup loops stopped; duplicates removed |
| R4 | done | `results/v3/partA/validation_reid.json (diagnosis); results/v3/partA/recipe_sweep.json` | collapse diagnosis; G-Recipe in the gate table |
| R5 | done (7 snapshots) | `checkpoints/sweep/sweep_*/step_*.pt` | every 5,000 steps, immutable |
| R6 | done | `results/v3/partA/forecast_controls.json` | G-Forecast in the gate table |
| R7 | done (5 benchmarks) | `results/v3/partA/bbbc054.json; results/v3/coverage/autofluorescence.json; results/v3/coverage/flim.json; results/v3/partA/validation_reid.json; results/v3/partA/association.json; manifests/models.yaml` | MIT licence recorded; forecasting is N/A for E7 |
| R8 | reported | `results/v3/partA/bbbc054.json` | TiJepa-1 vs DINOv2, V-JEPA 2, TiJepa-0 stated plainly in the paper |
| R9 | done | `results/v3/partA/association.json` | point tracking demoted to the supplement |
| R10 | done | `results/v3/validation_set.json` | 64 non-corpus ComplexEye movies, never a test set |
| R11 | done | `results/v3/partA/forecast_controls.json (strides 1, 3, 9, 27)` | S4 (long-horizon training) in the sweep |
| R12 | done (classical segmentation recorded; scale result stated) | `results/v3/partA/bbbc054.json (e1_segmentation_methods, scale_sensitivity)` | Cellpose not installed; stated in Methods |
| R13 | done | `results/v3/partC/sample_qc.json; source_rna.json; beta_glucan_replication.json` | G-BetaGlucan in the gate table |
| R14 | done | `results/v3/partB/partB_v3.json` | trace JEPA vs hand-crafted vs raw, well-bootstrap CIs |
| R15 | done | `results/v3/partB/partB_v3.json (b31_power_analysis)` |  |
| R16 | done | `results/v3/partB/b4_symmetric.json` | exploratory |
| R17 | done | `paper/v3/tmlr/main.tex; paper/v3/biorxiv/main.tex` | coverage map as the organising frame; C3 stated once as not performed |
| R18 | done (reduced to a Future-work subsection, author scope decision) | `paper/v3/sections/future.tex` |  |
| R19 | done | `scripts/run_all_v3.sh; legacy/README.md` |  |
| R20 | done | `grep -rl /home/ubuntu results/` |  |
| R21 | open | `reports/PR_BODY_v3.md` | author squash-merges PR #1 |
| R22 | done | `scripts/74_recipe_sweep_v3.py; legacy/scripts/53_pretrain_monitor_v2.py` | new runs scored on the validation set only |
| R23 | stated | `paper/v3/sections/limitations.tex` |  |
| R24 | stated (one seed per recipe; one GPU) | `paper/v3/sections/limitations.tex` |  |

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

## GPU status

The 5 x 20,000-step recipe sweep is still training in the background (expected to finish around 17:30 UTC on 2026-10-19); this PR will be updated with the G-Recipe outcome. **After the sweep finishes, the GPU is idle**: nothing else is scheduled, so the instance can be stopped once the final commit lands.

## Checks

- `pytest`: all tests pass (CPU); `ruff check`: clean.
- `scripts/98_check_claims_v3.py`: both manuscripts pass (numbers generated from results, directional claims, banned strings, anonymity of the TMLR version, provenance labels).
- Nothing was submitted, posted or released; no weights are published.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
