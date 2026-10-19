# TiJepa v1 — run report

Generated 2026-10-16T03:15:20Z from `results/v1/`.
Branch `tijepa-v0`, commit `396dfe6`.

This run executes `PLAN_v1.md`, which is a fix plan for the v0 run. Corrections to the
v0 record are in [`reports/V0_ERRATA.md`](V0_ERRATA.md); the backup record is in
[`reports/BACKUP.md`](BACKUP.md).

## 1. Compute

NVIDIA A100-SXM4-40GB x1, 30 logical cores, 226.8 GB RAM,
tier **T1**, torch
`2.11.0+cu128`, all data on local disk.

_No pretraining run completed in this report's snapshot._

One GPU means pretraining has a **single seed**. Seed-to-seed variation in Part A comes
from independent random-init draws, stratified cell subsamples and fold assignments, and
this is stated in the paper's Methods.

## 2. Data

| dataset | role | outcome |
|:---|:---|:---|
| ComplexEye (Zenodo 10082674, CC BY 4.0) | pretraining | 10/21 archives; 407 movies, 279,647 clips, 41.24 GB |
| S-BIAD2196 (CC BY 4.0) | tracking + forecasting | all 6 movies; 1 twin(s) excluded from pretraining |
| BBBC054 (CC BY 3.0) | morphology-class probe | 3 classes, 60 fields of view |
| GSE260996 / GSE261647 | Part B expression / accessibility | both pass their positive controls |
| Tay NF-kB traces | Part B dynamics | ligand codes sourced; 4 matched wells |
| GSE46903 / GSE58310 | Part C anchors / resting scores | 1 of 3 anchors usable |
| GSE212282 / GSE185433 | Part C restimulation | both qualify, 3 matched donors each |
| Dryad m0cfxpp8v | planned Part A headline | **not_provided** |
| LTDB | tracking (optional) | **not_provided** |

The published ComplexEye `R06.zip` has no end-of-central-directory record and is 57.5% zero padding, while its md5 matches the Zenodo record exactly - so the defect is in the deposit, not in our transfer (`complexeye_archive_defect.json`). S-BIAD2196 holds 6 movies, not 12: the BioStudies API serves only 5 rows however it is paged, and the sixth - the one duplicated in the pretraining corpus - is visible only in the FIRE file tree.

## 3. Section 12 traceability

**30 fixed, 3 partially fixed, 3 not fixed**, of 36 findings.

| ID | Finding | Status | Evidence |
|:---|:---|:---|:---|
| D1 | Part B used ATAC as RNA; B4 artifact | fixed | `partB/rna_guard.json`, `partB/rna_positive_control.json`, `partB/atac_positive_control.json`, `partB/b33_b4.json` |
| D2 | Dryad headline missing | not fixable — the dataset is an author prerequisite and was not provided; an explicit status is recorded and the code path is in place | `partA/dryad_benchmark.json`, `manual_intake.json` |
| D3 | Possible tracking leakage | fixed | `leakage_check.json` — real overlap found and the matching pretraining movie excluded |
| D4a | 7/21 archives, 14 corrupt | partially fixed — 10 of 21 archives processed; the rest were cut by the wall-clock budget at roughly 56 min per archive over an intermittent Zenodo connection. Every processed archive was MD5-verified, R08 was re-fetched after failing, and R06's defect is in the published file | `pretrain_corpus.json`, `complexeye_archive_defect.json` |
| D4b | Single source | partially fixed — the Leica reference movies are a second instrument with a separately verified pixel size, but PhagoStat microglia video was not added: the 94 GB archive could not be fetched within the run's remaining budget | `pretrain_corpus.json` (`clips_by_source`) |
| D4c | Leica pixel size unverified | fixed | `pretrain_corpus.json` (`pixel_size_provenance`) — 0.4762 µm/px derived from the published field of view, consistent on both axes |
| D4d | No QC | fixed | `pretrain_corpus.json` (`qc_thresholds`, `n_movies_excluded_qc`) — heuristic and labelled as such, since S-BIAD2133 could not be retrieved in full |
| D4e | PhagoStat video unused | not fixed — see D4b | `pretrain_corpus.json` |
| D5a | BBBC054 static | fixed | `pretrain_corpus.json` (`static_fraction`) |
| D5b | 2 of 3 classes | fixed | `partA/bbbc054_v1.json` — the third class is `ramified`, not `stratified`; v0's label tuple silently dropped 11,143 of 58,186 annotations (`reports/V0_ERRATA.md` E2) |
| D5c | FOV–time confound | fixed | `partA/bbbc054_v1.json` (`time_only_baseline`, `within_time_bin`, `fov_identity_mean`) — the time-only baseline scores at chance |
| D5d | BBBC054 pixel size | fixed | `partA/bbbc054_v1.json` (`um_per_px_provenance`) — 0.65 µm/px from the camera array and 2×2 binning, plus a scale-sensitivity analysis |
| D6 | Small tracking eval | fixed | `partA/tracking_forecast_v1.json` |
| D7 | Tay layout inferred | fixed | `partB/ligand_codes.json` — sourced from the deposit's plotting code and independently confirmed by the GEO treatment protocol |
| C1 | Cosmetic seeds | fixed | `partA/bbbc054_v1.json` (`seed_varies`), `partB/b3_dynamics.json` — every sd now comes from a changed subsample, fold assignment or init draw |
| C2 | 4-well bootstrap CI | fixed | `partB/b3_dynamics.json` — no CI, exact permutation and its floor reported |
| C3 | Undertrained | partially fixed — pretraining ran to its GPU-hour budget; see the compute table | `partA/pretrain_tijepa1_seed0.json`, `checkpoints/pretrain_log_tijepa1_seed0.csv` |
| C4 | No future prediction | not fixed | `partA/pretrain_tijepa1_seed0.json` (`objective`, per-mask losses), `partA/tracking_forecast_v1.json` (`forecasting`), `tests/test_masks.py` |
| C5 | Static frames OOD | fixed | `pretrain_corpus.json` (`static_fraction`) |
| C6 | Weak hand-crafted baseline | fixed | `src/tijepa/preprocess/morphology_v1.py`, `partA/bbbc054_v1.json` (`e1_segmentation_methods`) — Cellpose was not installed; the documented classical pipeline was used instead |
| C7 | Statistical floors / paired t / 4 conditions | fixed | `partB/b33_b4.json` (exact condition permutations), `partC/c1_anchors.json` (moderated paired tests), `tests/test_moderated.py` |
| C8 | Circular one-axis analysis | fixed | `partC/c3_projection.json`; `scripts/41_partC_projection.py` deleted |
| B1 | LPS anchor donor-confounded | fixed | `partC/c1_anchors.json` — under donor matching the LPS anchor cannot be built at all (2 pairs, minimum 3), which explains the v0 artefact |
| B2 | Rest vs restimulation | fixed | `partC/c3_restimulation.json` — GSE212282 and GSE185433, both with matched donors and an LPS restimulation |
| B3 | Macrophage type mismatch | fixed | `partC/c1_anchors.json` (`differentiation_choice`) — GM-CSF used and the mismatch stated, since the M-CSF arm has no differentiated control |
| B4 | "Acute endotoxin tolerance" | fixed | `paper/tijepa.md` — abstract, results and methods |
| B5 | Cell-type mismatch; shape ≠ activation | fixed | `paper/tijepa.md` §3 |
| B6 | Discussion overclaim | fixed | `claims_check.json` — the checker passes, and it caught the v0 abstract's three-way ranking of two encoders that are not distinguishable |
| E1 | Table header mismatch | fixed | `src/tijepa/viz/tables.py`, `tests/test_paper_tools.py` |
| E2 | §2.6 strings, repeat, missing Methods | fixed | `paper/tijepa.md` §2.6 and §4.1–4.11 |
| E3 | References | fixed | `references.json` — all resolved against Crossref or DataCite, each title keyword-checked because a DOI can resolve and still be the wrong paper |
| E4 | Figures | fixed | `figures.json`, `figures/fig*_v1.png` — each inspected, not just rendered |
| E5 | Hard-coded conclusions | fixed | `claims_check.json`, `src/tijepa/viz/claims.py` |
| E6 | Reproducibility gaps | fixed | `requirements-lock.txt`, `scripts/run_all_v1.sh`, `reports/V0_ERRATA.md` |
| E7 | Repo hygiene | fixed | no tracked file over 5 MB; `*.npz`/`*.npy` ignored; the stale axis comment in `scripts/60_partA_eval.py` fixed |
| O1 | Weights only on instance disk | fixed | `reports/BACKUP.md` — v0 checkpoint md5-verified on the author's Drive |

## 4. Results, with their source files

| result | value | source |
|:---|:---|:---|
| RNA positive control | 7/7 targets, PASS | `partB/rna_positive_control.json` |
| ATAC positive control | 7/7 genes, PASS | `partB/atac_positive_control.json` |
| B3.1 acute endotoxin tolerance | BA 0.547, exact p floor 0.167 | `partB/b3_dynamics.json` |
| B3.3 dynamics -> omics | null for both readouts | `partB/b33_b4.json` |
| B4 cross-species overlap | 0 genes | `partB/b33_b4.json` |
| C1 usable anchors | 1 of 3 | `partC/c1_anchors.json` |
| C3 projection | not performed; two independent reasons | `partC/c3_projection.json` |
| Part A BBBC054 | see the table in the paper | `partA/bbbc054_v1.json` |
| Tracking / forecasting | see the tables in the paper | `partA/tracking_forecast_v1.json` |

## 5. Deviations from PLAN_v1.md

1. **The corpus stopped at 10 of 21 archives.** zenodo.org answered
   about one connection in three from this instance and each 10 GB archive took roughly
   56 minutes, so continuing would have consumed the pretraining budget. The corpus is
   279,647 clips, comparable to v0's 288,081 but from more
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
