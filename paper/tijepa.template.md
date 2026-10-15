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
{{CORPUS_CLIPS}} cell-centred clips ({{CORPUS_GB}} GB) drawn from {{CORPUS_MOVIES}}
openly licensed (CC BY 4.0) human neutrophil migration movies, and evaluate frozen
representations against hand-crafted, ImageNet and DINOv2 baselines. On BBBC054
LPS-exposed microglia, TiJepa-0 reaches a balanced accuracy of {{BBBC_E4_BA}} on the
morphology-class probe ({{BBBC_BEST_NAME}} is best at {{BBBC_BEST_BA}}), and
{{BBBC_E4_RHO}} Spearman ρ for LPS exposure time. On held-out neutrophil movies it
tracks points with ⟨δ⟩ = {{TRACK_E4}} against {{TRACK_BASE}} for a nearest-detection
baseline. Separately, single-cell NF-κB dynamics distinguish naive→LPS from LPS→LPS
macrophages at a balanced accuracy of {{B31_RAW_BA}} (95% CI {{B31_RAW_CI}}), but the
contrast rests on {{B31_GROUPS}} wells and is **not significant** against a group-level
permutation null (p = {{B31_RAW_P}}). Attempts to regress condition-level transcriptomes
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

{{CORPUS_PARA}}

Figure 2 shows the corpus composition, the pretraining loss and collapse diagnostics.

### 2.2 TiJepa-0: latent-space masked prediction over tubelets

{{PRETRAIN_PARA}}

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

{{BBBC_PARA}}

Figure 3 shows the encoder comparison, the exposure-time regression, and a
field-of-view identity probe included because FOV is perfectly confounded with exposure
time in this dataset's design.

### 2.4 Point tracking on held-out neutrophil movies

S-BIAD2196 provides brightfield neutrophil movies with instance masks whose IDs are
consistent over time. These movies were **held out of pretraining entirely**, so no
leakage is possible. This is the one evaluation in the run with real temporal
structure, and therefore the only place the frame-shuffle control applies.

{{TRACK_PARA}}

### 2.5 Part B: signalling dynamics and LPS tolerance

{{B31_PARA}}

{{B33_PARA}}

{{B4_PARA}}

Figure 5 summarises Part B.

### 2.6 Part C: a hypothesis-generating bridge

**This section pairs no image with any molecular sample.** Anchors and trained/tolerant
states come from different studies, different donors, different platforms and — for the
image side — a different species. It is presented to generate a testable prediction, not
to support a claim.

{{C1_PARA}}

{{C2_PARA}}

{{C3_PARA}}

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
{{B31_GROUPS}} wells; the dynamics-to-expression regression has {{B33_CONDITIONS}}
shared conditions; the human tolerance contrast has 4 donors. Published label-free
classifiers show the same pattern — near-ceiling same-donor accuracy collapsing to
72–85% across donors — so donor and batch count, not model capacity, should dominate the
design of the experiment above.

**Future work.** The learned representation is designed to be added, as one modality
among several, to multi-omic models of immune disease such as iMEMORY. That integration
is future work and nothing here bears on it.

---

## 4. Methods

{{METHODS}}

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

{{DATA_AVAILABILITY}}

Model weights: see `reports/WEIGHTS.md`.

---

## References

{{REFERENCES}}
