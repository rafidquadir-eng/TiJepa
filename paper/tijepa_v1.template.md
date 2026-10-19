# TiJepa-1: self-supervised video representations of innate immune cell behaviour, and what public data can and cannot say about trained immunity

**Rafid Quadir**

Immunome AI Biotechnologies, New York, NY <!-- TODO(author): confirm affiliation -->

---

## Abstract

Trained immunity — the durable, non-specific reprogramming of innate immune cells by
stimuli such as β-glucan and BCG — is defined by molecular memory and is read out almost
exclusively by sequencing. Whether it has an imaging phenotype is unknown, because, as a
systematic survey of the public record included in this repository shows, **no public
dataset contains imaging or video of trained, tolerant or innate-memory immune cells**.
We therefore ask narrower, answerable questions on data that exists. We pretrain
TiJepa-1, a joint-embedding predictive architecture over spatio-temporal tubelets, on
{{CORPUS_CLIPS}} cell-centred clips ({{CORPUS_GB}} GB) from {{CORPUS_MOVIES}} openly
licensed (CC BY 4.0) human neutrophil migration movies, with an objective that mixes
spatial masking with **future masking**, so the model is trained to forecast its own
latents. {{ABSTRACT_PARTA}} {{ABSTRACT_FORECAST}} Separately, single-cell NF-κB dynamics
are used to ask whether **acute endotoxin tolerance** (LPS then LPS, 8 h apart, mouse
BMDM) is visible in signalling behaviour: {{ABSTRACT_B31}} Analyses relating dynamics to
transcription and to chromatin accessibility, and a cross-species comparison of mouse
heterologous priming with human LPS tolerance, are null and reported as such. Finally we
place β-glucan-trained and LPS-tolerised human monocytes on donor-matched activation
axes, at rest and — the test that matters for trained immunity — **on restimulation**.
That last analysis is a condition-level, cross-study bridge: it is
**hypothesis-generating only**, it pairs no image with any molecular sample, and it makes
no claim that trained immunity can be identified from images. We release code, verified
data manifests and the errata for our own previous version.

---

## 1. Introduction

Innate immune cells retain a memory of past encounters. After exposure to β-glucan, BCG
or oxidised LDL, monocytes and macrophages respond more strongly to an unrelated second
challenge — *trained immunity* — while prior LPS produces the opposite, *tolerance*
[Netea 2016; Saeed 2014; Novakovic 2016]. Both states are defined by chromatin and
transcriptional changes at inflammatory loci, and both are measured almost exclusively by
sequencing.

Whether these states have a visible phenotype — a shape, a motility pattern, a metabolic
autofluorescence signature — is an open and practically important question. If they do,
label-free imaging would give a cheap, longitudinal, single-cell readout of a process
that currently requires destructive assays.

A systematic survey of the public record, included in this repository
(`reports/Trained immunity imaging datasets.md`), establishes that this question cannot
currently be answered from public data. A Europe PMC full-text search for
`"trained immunity"` combined with any image-repository term returns zero hits; the
BioImage Archive's 59 macrophage/monocyte/neutrophil studies include no β-glucan, BCG,
oxLDL, MDP or LPS-tolerance study; and an enumeration of all 147 IDR projects and 105
screens finds no innate-immune training study.

This paper does not close that gap. It builds and honestly evaluates components such a
study would need, on data that exists:

1. **Part A.** Does JEPA-style video pretraining on open innate-immune microscopy capture
   cell state, support tracking, and — new in this version — **forecast** its own latents?
2. **Part B.** Can single-cell NF-κB dynamics distinguish acutely endotoxin-tolerised
   macrophages from naive ones, and do those dynamics predict transcription or chromatin
   accessibility?
3. **Part C.** Where do trained and tolerant cells sit on donor-matched activation axes,
   at rest and on restimulation? (Hypothesis only.)

**This is version 1.** It is a correction of a previous run whose errors we document
rather than quietly fix: `reports/V0_ERRATA.md` lists six, including an accessibility
matrix analysed as if it were expression, a dataset label name that silently discarded a
fifth of the annotations, and a confidence interval reported over four groups.

---

## 2. Results

### 2.1 A pretraining corpus of openly licensed innate-immune video

{{CORPUS_PARA}}

{{LEAKAGE_PARA}}

### 2.2 TiJepa-1: latent masked prediction with a future-prediction objective

{{PRETRAIN_PARA}}

### 2.3 The planned headline benchmark could not be run

{{DRYAD_PARA}}

### 2.4 Morphology-class probes on BBBC054

{{BBBC_PARA}}

{{BBBC_CONFOUND_PARA}}

{{BBBC_TABLE}}

### 2.5 Point tracking and latent forecasting on held-out movies

{{TRACK_PARA}}

{{FORECAST_PARA}}

### 2.6 Part B: signalling dynamics, tolerance, transcription and accessibility

{{PC_PARA}}

{{B31_PARA}}

{{B32_PARA}}

{{B33_PARA}}

{{B4_PARA}}

### 2.7 Part C: a hypothesis-generating bridge, at rest and on restimulation

**This section pairs no image with any molecular sample.** Anchors and trained/tolerant
cells come from different studies, different donors and different platforms. It is
presented to generate a testable prediction, not to support a claim.

{{C1_PARA}}

{{C2_PARA}}

{{C3_RESTIM_PARA}}

{{C3_PROJECTION_PARA}}

---

## 3. Discussion

**What this work shows.** Self-supervised video pretraining on openly licensed
innate-immune microscopy is practical at modest scale, and the resulting frozen
representation can be compared against strong baselines under leakage-controlled
protocols. {{DISCUSSION_PARTA}}

**What it does not show.** It does not show that TiJepa-1, or any model here, identifies
trained immunity from images. **No trained or tolerant cell was imaged in any dataset
used in this work.** {{DISCUSSION_LIMITS}}

**The cell-type chain is a limitation in its own right.** Pretraining used human
neutrophil migration video; evaluation used mouse microglia (BBBC054) and human
neutrophils (S-BIAD2196); Part B used mouse bone-marrow-derived macrophages; and the
target biology is human monocytes and monocyte-derived macrophages. Each link in that
chain is a change of cell type, species or both, and nothing here establishes that a
representation learned on migrating neutrophils transfers to macrophage activation
state.

**The binding constraint is data, not architecture.** Two independent findings point the
same way: the landscape survey found no public trained-immunity imaging, and the one
public benchmark that would have supplied activation-labelled *video* was not available
for this run. A single prime–rest–restimulate imaging experiment — naive, β-glucan-
trained, LPS-tolerised and classically activated cells, imaged label-free through the
rest period and restimulation, with paired cytokines and ATAC — would be worth more than
further modelling on existing public data.

**Statistical power is the second constraint.** {{POWER_PARA}}

**Future work.** The learned representation is intended as one modality among several in
multi-omic models of immune disease. That integration is future work and nothing here
bears on it.

---

## 4. Methods

{{METHODS}}

---

## 5. Data and code availability

Code: <https://github.com/rafidquadir-eng/TiJepa>. Every number in this paper is written
by `scripts/91_make_paper_v1.py` from the JSON files in `results/v1/`; no value is typed
by hand, and every directional phrase is generated from an interval or an exact
permutation p and then re-verified against the results by `scripts/92_check_claims.py`.

{{DATA_AVAILABILITY}}

Model weights: see `reports/WEIGHTS.md`. They are not published.

---

## References

{{REFERENCES}}
