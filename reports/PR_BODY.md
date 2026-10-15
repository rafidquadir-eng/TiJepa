# TiJepa-0: computational preprint v0

Executes `PLAN.md` end to end (P0 preflight → M0–M7) on a single A100-40GB. Full detail
in [`reports/RUN_REPORT.md`](reports/RUN_REPORT.md); the manuscript draft is
[`paper/tijepa.md`](paper/tijepa.md).

**Every number in the paper is written by `scripts/90_make_paper_numbers.py` from
`results/*.json`. Nothing is typed by hand, and a missing result renders as an explicit
"not available in this run" rather than a placeholder number. No synthetic data outside
`tests/`, and no image is paired with any molecular sample anywhere.**

## What ran

| Milestone | Outcome |
|---|---|
| P0 | 1×A100-SXM4-40GB, 30 cores, 216 GB RAM → tier **T1**, bf16, pretraining seed 0 only |
| M1 | 13 datasets, licence read from each record, sizes and checksums verified |
| M2 | Part B complete: B3.1, B3.2, B3.3, B4 |
| M3 | Part C omics: C1 anchors, C2 trained/tolerant scores |
| M4 | Pretraining corpus: **288,081 clips / 42.5 GB** from 288 of 302 movies |
| M5 | TiJepa-0 pretrained (see RUN_REPORT §1 for steps and GPU-hours) |
| M6 | BBBC054 probes, S-BIAD2196 tracking, Part C3 |
| M7 | 7 figures, paper draft, RUN_REPORT, MANUAL_DOWNLOADS, WEIGHTS |


## Results

Pretraining: **55,000 steps, 4.43 GPU-hours**, 24.44 effective
epochs over 288,081 clips; loss 0.400 → 0.312;
no collapse (effective rank rose from ~3.5 to ~29).

**BBBC054 morphology probe** (FOV-grouped CV, 3 seeds, 2,000-resample cluster bootstrap over FOVs):

| encoder | balanced accuracy [95% CI] | vs E6 random init (paired) | ρ exposure time | FOV-identity (lower better) |
|---|---|---|---|---|
| E1_handcrafted_morphology | **0.697** [0.678, 0.715] | -0.076 [-0.094, -0.058] | -0.000 | 0.095 |
| E2_imagenet_vit_s | **0.755** [0.738, 0.772] | -0.017 [-0.036, +0.002] | +0.197 | 0.234 |
| E3_dinov2_vit_s | **0.769** [0.755, 0.786] | -0.003 [-0.019, +0.014] | +0.277 | 0.230 |
| E4_tijepa0 | **0.828** [0.814, 0.843] | +0.056 [+0.045, +0.067] | +0.083 | 0.110 |
| E5_tijepa0_frame_shuffled | N/A (undefined on static frames) | | | |
| E6_tijepa0_random_init | **0.772** [0.756, 0.790] | - (reference) | +0.100 | 0.065 |

TiJepa-0 is the best encoder, and the comparison that isolates **pretraining** from
architecture — E4 against the identical network with random weights — is
**+0.056 [0.045, 0.067], p < 0.0001**. It also carries *less* FOV identity than the
ImageNet and DINOv2 baselines, so the gain is not field memorisation. Note that E6 alone
already matches DINOv2 (d = −0.003, p = 0.73): much of the performance on this task comes
from the architecture and probe rather than from either pretraining corpus.

**Two negative results, reported rather than buried:**

1. LPS exposure-time regression is weak for every encoder and **worse** for TiJepa-0
   (ρ 0.083) than for DINOv2 (ρ 0.277); all MAEs sit at or near the mean-predictor's.
2. Point tracking on held-out S-BIAD2196 — TiJepa-0 **does not beat** the baseline and is
   indistinguishable from its own frame-shuffled and random-init controls, so the learned
   temporal structure does not transfer:

| method | ⟨δ⟩ (1–16 µm) | median survival (frames) |
|---|---|---|
| baseline_nearest_detection | 0.527 | 17 |
| E4_tijepa0 | 0.463 | 4 |
| E5_tijepa0_frame_shuffled | 0.474 | 3 |
| E6_tijepa0_random_init | 0.436 | 4 |

## Three things a reviewer should look at first

**1. The planned headline benchmark could not be obtained.** Dryad
`doi:10.5061/dryad.m0cfxpp8v` (M0/M1/M2 macrophage movies) returns 401/403 on every
official endpoint, and the file-stream endpoint serves an Anubis proof-of-work
interstitial. Solving it would be bypassing an access control, which `PLAN.md` §2 and §9
forbid, so the run stopped and documented it. Consequences, none of which were papered
over:

- Part A evaluation rests on **BBBC054**, which is *static*, so the frame-shuffle
  control (E5) is undefined there and is reported **N/A**, not as a number. It runs only
  on the tracking task.
- **No dataset in this run provides video of innate immune cells with activation-state
  labels** — the paper's own thesis, restated as an experimental limitation.
- The Part C3 projection is **not performed**: it needs image-space M0/M1/M2 centroids
  that only that dataset supplies. BBBC054's microglial classes are not the same labels,
  so substituting them was refused; a clearly-labelled reduced one-axis analysis is
  offered instead.

**2. A bug was found and fixed that would have invalidated all of Part A.** The BBBC054
page states "axis-1 indicates the x-coordinate, and axis-2 indicates the y-coordinate".
The data contradict it: the images are 1080 rows × 1280 columns, and across all 47,043
annotations axis-1 spans −4..1080 while axis-2 spans −6..1284, so axis-1 indexes rows
and axis-2 columns (the names are numpy-style). Following the page transposed every crop:

| | hand-crafted morphology probe |
|---|---|
| following the page | BA 0.482 (chance), permutation p = 0.65 |
| following the data | **BA 0.713**, 95% CI [0.684, 0.741], p = 0.005 |

The data are followed and the discrepancy is documented in the loader.

**3. Part B is a reported null, not a result.** NF-κB dynamics separate naive→LPS from
LPS→LPS at BA 0.701 (95% CI [0.643, 0.752]) from the raw trace, but the contrast rests
on **4 wells**, so a well-level permutation null admits only a handful of distinct
assignments and p cannot fall below ~1/6 — it is **not significant** (p = 0.30). B3.3
lands *below* its permutation null (p = 0.49) and B4's cross-species overlap is 0 genes.
All are reported rather than dropped, per `PLAN.md` §1.1.4.

## Data integrity notes

- **Tay deposit reverse-engineering.** The deposit ships no column dictionary and omits
  the folder holding the human-readable condition labels. The matrix layout was
  reconstructed from the deposit's own MATLAB preprocessing script, and the LPS ligand
  code was identified by intersecting the second-ligand code sets of two deposit files
  with their file-name ligand sets. This is recorded as an **inference**, and all
  analyses are restricted to the five files whose layout could be confirmed; the
  cytokine-block and sepsis files are flagged layout-unverified and excluded.
- **GSE58310 is donor-paired**, so a paired test is used; an unpaired test discards the
  pairing and is badly underpowered at n = 4.
- **14 movies were excluded** on corrupt zip members (bad CRC / magic number) in
  archives whose transfer had been interrupted, rather than partially decoded.
- Pretraining used **only CC BY 4.0** data. The datasets the plan excluded on licence
  grounds were never downloaded.

## Deviations

17 deviations are tabulated in `reports/RUN_REPORT.md` §4, each with what the plan asked,
what was done, and why. The load-bearing ones are above; the rest cover the classical
detector in place of Cellpose, the untrained frame-QC classifier (only 5 of 8,441
S-BIAD2133 files retrievable), 200 rather than 1,000 permutations, a PCA step added
identically to every encoder's probe, single-seed pretraining, and a mid-run torch
upgrade.

## TODO(author)

- [ ] Confirm the affiliation in `paper/tijepa.md` (placeholder: *Immunome AI
      Biotechnologies, New York, NY*).
- [ ] Review the AI-assistance disclosure in `paper/tijepa.md` §4.9.
- [ ] Declare competing interests (`paper/tijepa.md` §4.10).
- [ ] Confirm the licence choice: Apache-2.0 for code, CC BY 4.0 for text and figures —
      say so if a non-commercial licence is preferred.
- [ ] Resolve the DOIs and arXiv ids marked `TODO(verify)` in the references before
      posting. Only identifiers actually resolved during the run are stated; nothing was
      guessed.
- [ ] Fetch Dryad `m0cfxpp8v` by hand (`reports/MANUAL_DOWNLOADS.md`) and re-run
      `scripts/60_partA_eval.py` and `scripts/41_partC_projection.py` to obtain the
      planned headline benchmark and the C3 projection.
- [ ] Optionally fetch LTDB and complete S-BIAD2133 to train the frame-QC classifier.
- [ ] Publish model weights if desired — see `reports/WEIGHTS.md`. They are **not** in
      git and were **not** published.

### One self-inflicted issue, disclosed

`scripts/60_partA_eval.py` began caching Part A embeddings to
`results/partA/bbbc054_embeddings.npz` (39 MB), which breaches `PLAN.md` §1.3.2
("committed result files must be < 5 MB each"). It is now untracked and ignored, but the
blob remains in this branch's history — removing it there would require rewriting history
and force-pushing, which §1.3.1 forbids. **Squash-merging this PR drops it.**

**Nothing has been posted to bioRxiv, Zenodo, Hugging Face or anywhere else**, per
`PLAN.md` §1.4.3. No GitHub Release was created. `main` was never pushed to and nothing
was force-pushed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
