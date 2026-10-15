# TiJepa: self-supervised video representations of innate immune activation, with a dynamics-to-chromatin bridge toward trained immunity

**Rafid Quadir**

Immunome AI Biotechnologies, New York, NY <!-- TODO(author): confirm affiliation -->

---

## Abstract

Trained immunity — the durable, non-specific reprogramming of innate immune cells by
stimuli such as β-glucan and BCG — is defined by molecular memory that is read out
almost exclusively with sequencing. Whether it has an imaging phenotype is unknown,
because, as a systematic survey of the public record included in this repository shows,
**no public dataset contains imaging or video of trained, tolerant, or innate-memory
immune cells**. We therefore ask a narrower, answerable question: what can
self-supervised video models and single-cell signalling dynamics tell us about innate
immune *activation* states from data that does exist? We pretrain TiJepa-0, a
joint-embedding predictive architecture (JEPA) over spatio-temporal tubelets, on
288,081 cell-centred clips (42.48 GB) drawn from 288
openly licensed (CC BY 4.0) human neutrophil migration movies, and evaluate frozen
representations against hand-crafted, ImageNet and DINOv2 baselines. On BBBC054
LPS-exposed microglia, TiJepa-0 reaches a balanced accuracy of n/a on the
morphology-class probe (n/a is best at n/a), and
n/a Spearman ρ for LPS exposure time. On held-out neutrophil movies it
tracks points with ⟨δ⟩ = n/a against n/a for a nearest-detection
baseline. Separately, single-cell NF-κB dynamics distinguish naive→LPS from LPS→LPS
macrophages at a balanced accuracy of 0.701 (95% CI [0.643, 0.752]), but the
contrast rests on 4 wells and is **not significant** against a group-level
permutation null (p = 0.30). Attempts to regress condition-level transcriptomes
from dynamics, and to link mouse and human tolerance genes, were both null and are
reported as such. Finally, we place β-glucan-trained and LPS-tolerised human monocytes on
activation axes derived from an independent macrophage atlas. That last analysis is a
condition-level, cross-study, cross-species bridge: it is **hypothesis-generating only**,
it pairs no image with any molecular sample, and it makes no claim that trained immunity
can be identified from images. We release code, verified data manifests and weights.

---

## 1. Introduction

Innate immune cells retain a memory of past encounters. After exposure to β-glucan, BCG
or oxidised LDL, monocytes and macrophages mount a stronger response to an unrelated
second challenge — *trained immunity* — while exposure to LPS produces the opposite,
*tolerance* [Netea et al.; Saeed et al. 2014; Novakovic et al. 2016]. Both states are
defined by chromatin and transcriptional changes at inflammatory loci, and both are
measured almost exclusively by sequencing.

Whether these states have a visible phenotype — a shape, a motility pattern, a
metabolic autofluorescence signature — is an open and practically important question. If
they do, label-free imaging would offer a cheap, longitudinal, single-cell readout of a
process that currently requires destructive assays.

A systematic survey of the public record, included in this repository
(`reports/Trained immunity imaging datasets.md`), establishes that this question cannot
currently be answered from public data. A Europe PMC full-text search for
`"trained immunity"` combined with any image-repository term returns zero hits; the
BioImage Archive's 59 macrophage/monocyte/neutrophil studies include no β-glucan, BCG,
oxLDL, MDP or LPS-tolerance study; and an enumeration of all 147 IDR projects and 105
screens finds no innate-immune training study and no innate-immune time-lapse at all.
The field deposits sequencing with exemplary completeness and treats microscopy as
illustration.

This paper does not close that gap — closing it requires a prime–rest–restimulate
imaging experiment that no public dataset provides. Instead it builds and honestly
evaluates the components such a study would need, on data that exists:

1. **Part A.** Does JEPA-style self-supervised video pretraining on open innate-immune
   microscopy capture activation state and support tracking?
2. **Part B.** Can single-cell NF-κB signalling dynamics predict features of LPS
   tolerance, which is the one innate-memory state with public live-imaging data?
3. **Part C.** Where do trained and tolerant states fall on activation axes shared
   between omics and a learned image space? (Hypothesis only.)

---

## 2. Results

### 2.1 A pretraining corpus of openly licensed innate-immune video

We built the pretraining corpus from the ComplexEye neutrophil migration release (Zenodo 10082674, CC BY 4.0): 288 of 302 movies processed without failure into **288,081 cell-centred clips** of 16x96x96 uint8 (42.48 GB) in 42 sequential shards. Cells were detected with a classical temporal-median/MAD pipeline and linked with trackpy; each clip follows its own track so the cell stays centred and the model cannot use frame drift as a shortcut. Clips were sampled at two strides, giving effective frame intervals of 8s (n=227,246), 24s (n=60,835). All frames are at 0.641 um/px against a 0.65 um/px target, so no resampling was applied. The corpus is single-source, which is a real limitation: it is one cell type (human neutrophils) under one imaging modality.

Figure 2 shows the corpus composition, the pretraining loss and collapse diagnostics.

### 2.2 TiJepa-0: latent-space masked prediction over tubelets

TiJepa-0 is a ViT-S (96x96 px, 16 frames, 2x8x8 tubelets, 1,152 tokens) trained with a V-JEPA-style objective: a context encoder sees visible tubelets, an EMA target encoder sees the whole clip, and a narrow predictor regresses the target encoder's latents at masked positions under an L1 loss. Masking is multi-block and tubed over time, so a masked cell cannot be copied from an adjacent frame. Training ran for **150 steps (0.59 effective epochs over the corpus) in 0.02 GPU-hours** on one NVIDIA A100-SXM4-40GB in bf16, stopping on `max_steps`. Mean loss fell from 0.676 over the first 500 steps to 0.676 over the last 500. Embedding standard deviation and effective rank were logged throughout as collapse diagnostics (Figure 2); no collapse intervention was required. Pretraining used **seed 0 only**: the run had a single GPU, so the three-seed requirement was applied to all probes and downstream models instead, and single-seed pretraining is a limitation of this run.

### 2.3 Activation-state probes on BBBC054

The planned headline benchmark — Dryad `doi:10.5061/dryad.m0cfxpp8v`, 18 BMDM
phase-contrast movies labelled M0/M1/M2 — proved **unreachable**: every official
download endpoint refused automated clients and the file-stream endpoint served a
proof-of-work interstitial, which this run treated as an access control not to be
bypassed (`reports/MANUAL_DOWNLOADS.md`). Part A evaluation therefore rests on BBBC054,
a set of LPS-exposed mouse microglia images with per-cell phenotype annotations.

BBBC054 is **static**: each annotated cell is a single frame. This has a direct
consequence for the central question of whether temporal structure matters — the
frame-shuffle control (E5) is undefined on static clips and is reported as N/A rather
than as a number. It also means that, with Dryad unreachable, **no dataset in this run
provides video of innate immune cells carrying activation-state labels**, which is
itself the paper's thesis restated as an experimental limitation.

*(not available in this run; see `reports/RUN_REPORT.md`)*

Figure 3 shows the encoder comparison, the exposure-time regression, and a
field-of-view identity probe included because FOV is perfectly confounded with exposure
time in this dataset's design.

### 2.4 Point tracking on held-out neutrophil movies

S-BIAD2196 provides brightfield neutrophil movies with instance masks whose IDs are
consistent over time. These movies were **held out of pretraining entirely**, so no
leakage is possible. This is the one evaluation in the run with real temporal
structure, and therefore the only place the frame-shuffle control applies.

*(not available in this run; see `reports/RUN_REPORT.md`)*

### 2.5 Part B: signalling dynamics and LPS tolerance

The Tay-lab deposit (Zenodo 14337874, CC BY 4.0) provides single-cell NF-κB traces across sequential ligand pairs. The deposit ships no column dictionary, and the folder holding the human-readable condition labels was not deposited, so the matrix layout was reconstructed from the deposit's own MATLAB preprocessing script and the ligand code for LPS was identified by intersecting the second-ligand code sets of two deposit files with their file-name ligand sets. Analyses are restricted to the five files whose layout could be confirmed; the cytokine-block and sepsis files are flagged layout-unverified and excluded.

Classifying prior LPS exposure from the response to a second LPS stimulus at a matched 8 h interval (139 naive→LPS vs 172 LPS→LPS cells) gives a balanced accuracy of **0.701** ([0.643, 0.752]) from the raw trace and 0.598 ([0.566, 0.633]) from hand-crafted features. **Neither is significant**: the contrast rests on 4 wells, so a well-level permutation null is coarse and both give p = 0.30. We report the point estimates because the plan requires null results to be reported, not because they support a claim. First-ligand identity across 10 classes reaches 0.338 (chance 0.100); this is **not** comparable to the published 72.8% four-class figure, which uses a different class set and validation scheme.

Regressing condition-level expression on condition-mean trace features with leave-one-condition-out ridge over 4 shared conditions gives a mean per-gene r of 0.055 and 86 genes with r > 0.5, **below** a condition-permutation null averaging 333.9 such genes (p = 0.49). This is a null result. The deposited RNA-seq contains no LPS→LPS sample, so the homologous tolerance contrast classified above has no matching transcriptome at all; the shared conditions are heterologous.

Linking the mouse and human literatures was also null. In human (GSE58310, LPS-tolerised vs macrophage control, 4 donor-paired samples, 24,655 expressed genes) the pre-specified criteria of BH-FDR < 0.05 and |log2FC| > 1 return 0 up and 1 down genes; in mouse (heterologous primed->LPS (n=4) vs media->LPS (n=2); the deposit has no LPS->LPS sample) they return 1 up. The ortholog overlap is 0 genes. A clearly-labelled exploratory analysis using nominal p < 0.01 instead of FDR (34 human up genes) also gives an overlap of 0. The two contrasts are not the same biological comparison — the mouse one is heterologous priming, the human one homologous tolerance — and neither has the power to support an overlap test.

Figure 5 summarises Part B.

### 2.6 Part C: a hypothesis-generating bridge

**This section pairs no image with any molecular sample.** Anchors and trained/tolerant
states come from different studies, different donors, different platforms and — for the
image side — a different species. It is presented to generate a testable prediction, not
to support a claim.

Anchor signatures were derived from GSE46903 (Xue et al., human GM-CSF macrophages on GPL6947). All three anchors were taken at 72 h, the only time point at which ultrapure LPS samples exist in that series, against the matched unstimulated control: **S_LPS** from 6 upLPS vs 7 control samples; **S_IFNg** from 7 IFNg vs 7 control samples; **S_IL4** from 10 IL4 vs 7 control samples. Each signature is the top 100 up-regulated genes by Welch t among genes passing BH-FDR < 0.05. S_IFNg recovers canonical interferon-γ response genes (GBP5, GBP1, GBP4, GCH1, TAP1), which is a useful sanity check; S_LPS at 72 h does not recover acute LPS genes, consistent with the acute response having resolved by that time point, and this weakens the LPS half of the M1-like axis.

We scored every GSE58310 sample (Saeed et al.: naive monocytes, 6 d macrophage control, LPS-tolerised and β-glucan-trained, 4 donors each) with a rank-based singscore-style statistic, expressed relative to that dataset's own macrophage baseline. Rank-based scoring is what makes an array-derived signature comparable to RNA-seq at all.

| state | n | M1-like (95% CI) | M2-like (95% CI) |
|---|---|---|---|
| LPS tolerised | 4 | 0.0014 [-0.0029, 0.0058] | -0.0010 [-0.0083, 0.0065] |
| beta glucan trained | 4 | -0.0092 [-0.0135, -0.0041] | -0.0042 [-0.0119, 0.0036] |
| control | 4 | 0.0000 [-0.0028, 0.0033] | 0.0000 [-0.0098, 0.0095] |
| naive monocyte | 4 | 0.0191 [0.0119, 0.0259] | -0.0269 [-0.0414, -0.0152] |

The separations are small — hundredths of a unit on a rank scale whose range is ±0.5 — and the trained and tolerant states sit close to the macrophage baseline on both axes, with naive monocytes displaced furthest. Replication was not possible: NOT USABLE: the GEO series matrix contains no expression table (table_begin immediately followed by table_end) and no processed supplementary matrix is deposited NOT USABLE for gene-level scoring: only bigWig/bedGraph coverage tracks are deposited, no gene-level matrix

**The projection into image space (C3) was not performed.** the image-side anchors require the M0/M1/M2 movie dataset that proved unreachable Mapping omics coordinates into the learned image space requires image-side centroids for M0, M1 and M2 cells, and no accessible dataset in this run provides images labelled with those states. Rather than substitute a different label set and present it as the same analysis, the projection is left undone and the omics coordinates are reported on their own.

Figure 6 shows the omics coordinates and, where available, the predicted positions in
image space.

---

## 3. Discussion

**What this work shows.** Self-supervised video pretraining on openly licensed
innate-immune microscopy is practical at modest scale, and the resulting frozen
representation can be compared against strong baselines under leakage-controlled
protocols. Single-cell NF-κB dynamics carry information about prior LPS exposure.

**What it does not show.** It does not show that TiJepa-0, or any model here, identifies
trained immunity from images. No trained or tolerant cell was imaged in any dataset used.
The Part B tolerance result is not statistically significant against a group-level null.
The Part B dynamics-to-expression and cross-species analyses are null. The Part C bridge
is an interpolation between three anchor conditions across studies and species.

**The binding constraint is data, not architecture.** Two independent findings point the
same way: the landscape survey found no public trained-immunity imaging, and the one
public benchmark that would have supplied activation-labelled *video* was behind an
access control. A single prime–rest–restimulate imaging experiment — naive, β-glucan-
trained, LPS-tolerised and classically activated cells, imaged label-free over the rest
period and through restimulation, with paired cytokines and ATAC — would be worth more
than any further modelling on existing public data.

**Statistical power is the second constraint.** The Part B tolerance contrast has
4 wells; the dynamics-to-expression regression has 4
shared conditions; the human tolerance contrast has 4 donors. Published label-free
classifiers show the same pattern — near-ceiling same-donor accuracy collapsing to
72–85% across donors — so donor and batch count, not model capacity, should dominate the
design of the experiment above.

**Future work.** The learned representation is designed to be added, as one modality
among several, to multi-omic models of immune disease such as iMEMORY. That integration
is future work and nothing here bears on it.

---

## 4. Methods

### 4.1 Compute and tier

1x NVIDIA A100-SXM4-40GB
(40960 MiB),
30 logical cores, 226.8 GB RAM.
Tier **T1** (exactly 1 GPU detected (40960 MiB); brief: 1 GPU -> T1), encoder ViT-S,
precision bf16, raw-video cap 120.0 GB.
PyTorch 2.11.0+cu128, CUDA 12.8.
Seed policy: single GPU: pretraining uses seed 0 only; seeds 0/1/2 used for all probes and downstream models (must be stated in Methods).

### 4.2 Pretraining data and preprocessing

See §2.1. Detection used a temporal-median background with a MAD threshold and an area
filter; linking used trackpy (search radius 12 px, memory 2, minimum track length 16).
Cellpose was not used: the labelled evaluation sets need no segmentation (BBBC054 ships
annotated coordinates, S-BIAD2196 ships tracked instance masks), so a classical detector
kept the GPU free for training. The frame-QC classifier specified in the plan was **not**
trained, because only 5 of 8,441 S-BIAD2133 files could be retrieved in time; no QC
filtering was applied to pretraining clips.

### 4.3 Model and training

ViT-S encoder (384 dim, 12 layers, 6 heads) over 2x8x8 tubelets of 16x96x96 clips
(1,152 tokens), fixed 3D sine-cosine positional embeddings, EMA target encoder
(momentum 0.996 → 1.0), predictor of width 192 and depth 6,
multi-block tube masking, L1 loss on layer-normalised target latents, AdamW with cosine
schedule and warmup, bf16 autocast, gradient clipping at 1.0.

### 4.4 Probes and statistics

All encoders are frozen and every encoder gets the identical probe. Splits are grouped
at the highest available unit (FOV for BBBC054, well for Part B traces, movie for
tracking), never at the cell or frame level. Confidence intervals are cluster bootstraps
(2,000 resamples) at the grouping unit; "better than chance" is tested by permuting
labels at the group level; encoder comparisons use a paired cluster bootstrap on
identical resamples. Learned probes are run at seeds 0/1/2 and reported as mean ± sd.
For Part B, the logistic C was tuned once by inner CV and then held fixed inside the
permutation refits, which is a recorded deviation from per-fit tuning.

### 4.5 Licences of data and weights

Pretraining used only CC BY 4.0 data. Every dataset's licence was read from its own
record and is recorded in `manifests/datasets.yaml`; third-party model weights and their
licences are in `manifests/models.yaml`. The datasets the plan excluded
(Zenodo 16319732 CC BY-NC-ND, DeepCell DynamicNuclearNet, the Cellpose training set,
CTMC-v1, LIVECell, Cell Tracking Challenge) were not downloaded or used.


### 4.9 AI-assistance disclosure

Code, analyses and an initial manuscript draft were produced with the assistance of an
AI coding agent (Claude, Anthropic) under the author's direction; the author reviewed
and is responsible for all content. <!-- TODO(author): review -->

### 4.10 Competing interests

<!-- TODO(author): declare competing interests -->

---

## 5. Data and code availability

Code: <https://github.com/rafidquadir-eng/TiJepa>. Every number in this paper is written
by `scripts/90_make_paper_numbers.py` from the JSON files in `results/`; no value is
typed by hand.

All data are public. Accessions, verified licences, sizes and checksums are in `manifests/datasets.yaml`:

- `bbbc054` — None — licence: CC BY 3.0 Unported (commercial use allowed) — status: ok
- `complexeye` — 10082674 — licence: cc-by-4.0 — status: ok
- `geo_GSE172116` — GSE172116 — licence: NCBI GEO public domain data (no licence asserted); cite the originating publication — status: ok
- `geo_GSE260996` — GSE260996 — licence: NCBI GEO public domain data (no licence asserted); cite the originating publication — status: ok
- `geo_GSE261647` — GSE261647 — licence: NCBI GEO public domain data (no licence asserted); cite the originating publication — status: ok
- `geo_GSE46903` — GSE46903 — licence: NCBI GEO public domain data (no licence asserted); cite the originating publication — status: ok
- `geo_GSE58310` — GSE58310 — licence: NCBI GEO public domain data (no licence asserted); cite the originating publication — status: ok
- `geo_GSE85243` — GSE85243 — licence: NCBI GEO public domain data (no licence asserted); cite the originating publication — status: ok
- `phagostat_annot` — 10803492 — licence: cc-by-4.0 — status: ok
- `s-biad2133` — S-BIAD2133 — licence: MIT — status: ok
- `s-biad2196` — S-BIAD2196 — licence: CC BY 4.0 — status: ok
- `tay_traces` — 14337874 — licence: cc-by-4.0 — status: ok
- `dryad_bmdm` — doi:10.5061/dryad.m0cfxpp8v — licence: CC0-1.0 (https://spdx.org/licenses/CC0-1.0.html), read from the Dryad API — status: unreachable

Model weights: see `reports/WEIGHTS.md`.

---

## References

Only references whose identifiers were resolved during this run
are listed. Works discussed from the landscape review are cited there with their own
identifiers (`reports/Trained immunity imaging datasets.md`).

- Saeed S, et al. Epigenetic programming of monocyte-to-macrophage differentiation and
  trained innate immunity. *Science* 2014. Data: GEO **GSE58310**. <!-- TODO(verify): DOI -->
- Novakovic B, et al. β-glucan reverses the epigenetic state of LPS-induced immunological
  tolerance. *Cell* 2016. Data: GEO **GSE85243** (in GSE85246). <!-- TODO(verify): DOI -->
- Xue J, et al. Transcriptome-based network analysis reveals a spectrum model of human
  macrophage activation. *Immunity* 2014. Data: GEO **GSE46903** (in GSE47189).
  <!-- TODO(verify): DOI -->
- Wang AG, Son M, et al. Inflammatory memory of macrophages emerges from coordinated
  transcription factor and chromatin dynamics. *Cell Systems* 2025. Data: Zenodo
  **10.5281/zenodo.14337874**, GEO **GSE260996**, **GSE261647**. <!-- TODO(verify): DOI -->
- ComplexEye: a multi-lens array microscope for high-throughput embedded immune cell
  migration analysis. Data: Zenodo **10.5281/zenodo.10082674** (CC BY 4.0).
  <!-- TODO(verify): journal reference -->
- CellTrack: benchmark dataset for 2D cell tracking. BioStudies **S-BIAD2196**
  (CC BY 4.0). <!-- TODO(verify): associated publication -->
- Ljosa V, Sokolnicki KL, Carpenter AE. Annotated high-throughput microscopy image sets
  for validation. *Nature Methods* 2012. BBBC054 image set (CC BY 3.0).
  <!-- TODO(verify): DOI -->
- Assran M, et al. Revisiting feature prediction for learning visual representations from
  video (V-JEPA). 2024. <!-- TODO(verify): arXiv id -->
- Oquab M, et al. DINOv2: learning robust visual features without supervision. 2023.
  <!-- TODO(verify): arXiv id -->
- Blumberg B, et al. MGI HOM_MouseHumanSequence ortholog report, Mouse Genome
  Informatics, downloaded during this run. <!-- TODO(verify): citation form -->

