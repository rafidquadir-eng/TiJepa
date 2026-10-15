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

**Nothing has been posted to bioRxiv, Zenodo, Hugging Face or anywhere else**, per
`PLAN.md` §1.4.3. No GitHub Release was created. `main` was never pushed to and nothing
was force-pushed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
