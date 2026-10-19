# {{TITLE}}

**Rafid Quadir**

Immunome AI Biotechnologies, New York, NY <!-- TODO(author): confirm affiliation -->

---

## Abstract

Trained immunity, the durable non-specific reprogramming of innate immune cells by
stimuli such as β-glucan, is defined by molecular memory and is read out almost
exclusively by sequencing. A systematic survey of the public record, included with this
work, finds **no public dataset with imaging or video of trained, tolerant or
innate-memory immune cells**, so whether these states have an imaging phenotype cannot
yet be tested. We therefore ask narrower questions on data that exist. We pretrain
TiJepa-1, a joint-embedding predictive architecture over spatio-temporal tubelets, on
{{CORPUS_CLIPS}} cell-centred clips from {{CORPUS_MOVIES}} openly licensed human
neutrophil migration movies, with an objective that mixes spatial masking with future
masking so that the model learns to forecast its own latents; a matched-step ablation
uses spatial masking only. {{ABSTRACT_PARTA}} {{ABSTRACT_FORECAST}} Single-cell NF-κB
dynamics are then used to ask whether **acute endotoxin tolerance** (LPS then LPS, mouse
bone-marrow-derived macrophages) is visible in signalling behaviour: {{ABSTRACT_B31}}
Analyses relating dynamics to transcription and to chromatin accessibility are null, and
a cross-species comparison of mouse priming with human LPS tolerance is exploratory.
Finally, in two public human restimulation datasets, we compare TNF and IL6 expression
after LPS restimulation of trained, tolerised and untrained monocyte-derived
macrophages with the directions the datasets' own publications report.
{{ABSTRACT_PARTC}} This last part is a condition-level, cross-study bridge: it is
**hypothesis-generating only**, it pairs no image with any molecular sample, and it makes
no claim that trained immunity can be identified from images.

---

## 1. Introduction

Innate immune cells retain a memory of past encounters. After exposure to β-glucan, BCG
or oxidised LDL, monocytes and macrophages respond more strongly to an unrelated second
challenge (*trained immunity*), while prior LPS can produce the opposite, *tolerance*
[netea2016; saeed2014; novakovic2016]. Both states are defined by chromatin and
transcriptional changes at inflammatory loci, and both are measured almost exclusively by
sequencing.

Whether these states have a visible phenotype (a shape, a motility pattern, a metabolic
autofluorescence signature) is an open and practically important question. If they do,
label-free imaging would give a cheap, longitudinal, single-cell readout of a process
that currently requires destructive assays.

A systematic survey of the public record, included in the repository, establishes that
this question cannot currently be answered from public data. A Europe PMC full-text
search for "trained immunity" combined with any image-repository term returns no hits;
the BioImage Archive's 59 macrophage, monocyte and neutrophil studies include no
β-glucan, BCG, oxLDL, MDP or LPS-tolerance study; and an enumeration of all 147 IDR
projects and 105 screens finds no innate-immune training study.

This paper does not close that gap. It builds and evaluates components such a study would
need, on data that exist:

1. **Part A.** Does JEPA-style video pretraining on open innate-immune microscopy capture
   cell morphology, support tracking, and forecast its own latents better than the same
   model trained without the forecasting objective?
2. **Part B.** Can single-cell NF-κB dynamics distinguish acutely endotoxin-tolerised
   macrophages from naive ones, and do those dynamics predict transcription or chromatin
   accessibility?
3. **Part C.** On restimulation, do public trained and tolerised human macrophage samples
   move TNF and IL6 in the directions their source publications report? (Hypothesis
   only.)

---

## 2. Results

### 2.1 A pretraining corpus of openly licensed innate-immune video

{{CORPUS_PARA}}

{{LEAKAGE_PARA}}

### 2.2 TiJepa-1: latent masked prediction with a future-prediction objective

{{PRETRAIN_PARA}}

{{FIG2}}

### 2.3 No activation-labelled video benchmark was available

{{DRYAD_PARA}}

### 2.4 Morphology-class probes on BBBC054

{{BBBC_PARA}}

{{BBBC_TABLE}}

{{BBBC_CONFOUND_PARA}}

{{BBBC_REGRESSION_PARA}}

{{FIG4}}

### 2.5 Point tracking and latent forecasting on held-out movies

{{TRACK_PARA}}

{{FORECAST_PARA}}

{{FIG5}}

### 2.6 Part B: signalling dynamics, tolerance, transcription and accessibility

{{PC_PARA}}

{{B31_PARA}}

{{B32_PARA}}

{{B33_PARA}}

{{B4_PARA}}

{{FIG6}}

### 2.7 Part C: a hypothesis-generating bridge on restimulation

**This section pairs no image with any molecular sample.** The trained, tolerised and
untrained cells come from different studies, donors and platforms from everything in
Parts A and B.

{{C1_PARA}}

{{C3_RESTIM_PARA}}

{{C3_PROJECTION_PARA}}

{{FIG7}}

---

## 3. Discussion

**What this work shows.** {{DISCUSSION_PARTA}}

**What it does not show.** It does not show that TiJepa-1, or any model here, identifies
trained immunity from images. **No trained or tolerant cell was imaged in any dataset
used in this work.** {{DISCUSSION_LIMITS}}

**The cell-type chain is a limitation in its own right.** Pretraining used human
neutrophil migration video; evaluation used mouse microglia (BBBC054) and human
neutrophils (S-BIAD2196); Part B used mouse bone-marrow-derived macrophages; and the
target biology is human monocytes and monocyte-derived macrophages. Each link is a change
of cell type, species or both, and nothing here establishes that a representation learned
on migrating neutrophils transfers to macrophage activation state. The pretraining corpus
also comes from one cell type and one instrument family, and it is kept fixed so that
every model trained on it is evaluated against the same data.

**The binding constraint is data, not architecture.** The landscape survey found no
public trained-immunity imaging, and the one public benchmark that would have supplied
activation-labelled *video* was not available. A single prime, rest and restimulate
imaging experiment (naive, β-glucan-trained, LPS-tolerised and classically activated
cells, imaged label-free through the rest period and restimulation, with paired cytokines
and ATAC) would be worth more than further modelling on existing public data.

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
by code from machine-readable result files in the repository; no value is typed by hand.
Every directional phrase is generated from an interval or an exact permutation p and
re-verified against the results, and every model label is derived from the checkpoint
record stored with the result it labels.

{{DATA_AVAILABILITY}}

Model weights are not published with this preprint.

---

## References

{{REFERENCES}}
