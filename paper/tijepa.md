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
279,647 cell-centred clips (41.24 GB) from 407 openly
licensed (CC BY 4.0) human neutrophil migration movies, with an objective that mixes
spatial masking with **future masking**, so the model is trained to forecast its own
latents. On BBBC054 microglial morphology classes the strongest encoder is E4 TiJepa-1 at 0.623 balanced accuracy on the three-class task; a time-only baseline scores at chance and within-time-bin scores match the overall figures, so the field-of-view/time confound does not explain the result. In latent forecasting, E0 TiJepa-0 (v0 reference, different corpus) reaches a ratio of 0.869 against copying the last observed latent, where below 1 means it beats assuming nothing changes. Separately, single-cell NF-κB dynamics
are used to ask whether **acute endotoxin tolerance** (LPS then LPS, 8 h apart, mouse
BMDM) is visible in signalling behaviour: leave-one-well-out balanced accuracy is 0.547 over 4 wells, and the exact permutation null admits no p below 0.167, so the design cannot reach significance and we draw no conclusion from it. Analyses relating dynamics to
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

The corpus is **279,647 cell-centred clips** of 16 frames at 96x96 px (41.24 GB in 43 sequential shards) from 407 movies across 10 of 21 ComplexEye archives (Zenodo 10.5281/zenodo.10082674, CC BY 4.0). Pixel sizes are verified rather than assumed: 0.641 um/px for the ComplexEye movies and 0.4762 um/px for the Leica reference movies, both derived in Methods 4.2 from the instrument specifications, and everything is resampled to a common 0.65 um/px. Clips are sampled at temporal strides x1, x3, x9, x27 (1: 211,661; 3: 61,078; 9: 6,173; 27: 735), and 10.2% of clips are made static by repeating a single frame, so that still-image evaluation is inside the training distribution rather than out of it. Frame-level quality control removed 18 movies; it is heuristic, with thresholds set from the empirical distribution of this corpus, because the deposit intended for a learned quality classifier could not be retrieved in full. Clips by source: complexeye_leica 2,511, complexeye 277,136.

The tracking evaluation uses S-BIAD2196, which like the pretraining corpus is human neutrophil brightfield video, so overlap between the two was checked rather than assumed. Every one of the 6 evaluation movies was fingerprinted against all 288 candidate pretraining movies then on disk (three frames, block-averaged, z-scored, compared by Pearson correlation, with the SHA-1 of the first frame recorded as well). **The check found real overlap**: 1 pair(s) matched, the strongest being 08_H4.ome.zarr.zip against R08.zip:H4.tiff at r = 0.99749 with identical frame dimensions. The evaluation movies are evidently drawn from the same collection. The matching pretraining movies are excluded from the corpus and the evaluation movies are kept, which is the right way round: excluding the evaluation movie would discard evaluation data to protect training data. This is also why the deposit had to be enumerated from its file tree rather than its API — the API exposes only five of its six movies, and the hidden sixth is the one that is duplicated in the corpus.

### 2.2 TiJepa-1: latent masked prediction with a future-prediction objective

Pretraining results are not available in this run, so the model's training is not described here.

### 2.3 The planned headline benchmark could not be run

The planned headline for this version was a video benchmark with activation-state labels: Dryad `doi:10.5061/dryad.m0cfxpp8v`, bone-marrow-derived macrophage movies labelled M0, M1 and M2, evaluated leave-one-batch-out. Every automated download endpoint serves a proof-of-work interstitial, which we treat as an access control not to be bypassed, so the dataset had to be fetched by hand; it was not available for this run. We report that plainly because its absence shapes what follows. **No dataset in this run provides video of innate immune cells carrying activation-state labels**, which is the paper's own thesis restated as an experimental limitation. In consequence: part a has no video benchmark with activation-state labels, so the frame-shuffle control e5 cannot be evaluated on a labelled video task and part a's classification evidence rests on static bbbc054 morphology classes; the f3.4 forecasting evaluation runs on s-biad2196 only, not on dryad; part c 8.4 (the c3 projection) cannot be performed: it needs image-space m0/m1/m2 centroids with stimuli matched to the omics anchors. see results/v1/partc/c3_projection.json.

### 2.4 Morphology-class probes on BBBC054

BBBC054 is a set of LPS-exposed mouse microglia images with per-cell **morphology-class** annotations. We call it that throughout: the labels are cell shapes, and shape is not activation state. Replicate 1 annotates all THREE classes - round, ramified and amoeboid - contrary to the premise of review finding D5b and PLAN_v1.md F3.2.1 that only two of three are present. The three-class task is primary; the two-class round-vs-amoeboid contrast is also reported for comparability with v0. The images are static, so the frame-shuffle control is undefined here and is reported as such; it applies only to the tracking and forecasting evaluations below.

Probes are identical for every encoder, cross-validated with `StratifiedGroupKFold` grouped by field of view, and repeated over three seeds that each change a stratified cell subsample, the fold assignment and, for the random-init control, the network draw — so every standard deviation reported is real variation rather than a decoration. Encoder pairs are compared with a paired cluster bootstrap over the 60 fields (2,000 resamples, both encoders scored on the same resample) with Benjamini-Hochberg correction across all comparisons within a task. The comparison that isolates pretraining from architecture is the pretrained model against the identical network with random weights: on the three-class task that difference is higher than (+0.082, 95% CI [+0.030, +0.145]).

All pairwise comparisons on the three-class task, with the direction generated from the interval rather than asserted:

| encoder A | encoder B | Δ BA (A − B) | 95% CI | BH p | direction |
|:---|---:|---:|---:|---:|---:|
| E1 hand-crafted (segmentation + Zernike + Haralick) | E2 ImageNet ViT-S | -0.126 | [-0.191, -0.061] | p < 0.0001 | lower |
| E1 hand-crafted (segmentation + Zernike + Haralick) | E3 DINOv2 ViT-S | -0.149 | [-0.206, -0.098] | p < 0.0001 | lower |
| E1 hand-crafted (segmentation + Zernike + Haralick) | E4 TiJepa-1 | -0.164 | [-0.220, -0.102] | p < 0.0001 | lower |
| E1 hand-crafted (segmentation + Zernike + Haralick) | E6 random init (3 draws) | -0.083 | [-0.155, -0.009] | p = 0.035 | lower |
| E2 ImageNet ViT-S | E3 DINOv2 ViT-S | -0.023 | [-0.075, +0.038] | p = 0.648 | not distinguishable |
| E2 ImageNet ViT-S | E4 TiJepa-1 | -0.038 | [-0.089, +0.022] | p = 0.648 | not distinguishable |
| E2 ImageNet ViT-S | E6 random init (3 draws) | 0.044 | [-0.030, +0.136] | p = 0.648 | not distinguishable |
| E3 DINOv2 ViT-S | E4 TiJepa-1 | -0.015 | [-0.058, +0.035] | p = 0.863 | not distinguishable |
| E3 DINOv2 ViT-S | E6 random init (3 draws) | 0.067 | [+0.007, +0.125] | p = 0.035 | higher |
| E4 TiJepa-1 | E6 random init (3 draws) | 0.082 | [+0.030, +0.145] | p < 0.0001 | higher |

In this dataset one field of view is one exposure time, so field identity and time are perfectly confounded and any field-grouped split leaks time. Three controls address it. First, a **time-only baseline** that classifies morphology from exposure time alone scores 0.332 on the three-class task and 0.500 on round-versus-amoeboid — chance in both cases, so exposure time by itself carries no morphology information. Second, **within-time-bin** balanced accuracy, computed inside six contiguous bins of ten fields and averaged, is within a thousandth of the overall figure for every encoder, so time is not what the encoders are reading. Third, a field-identity probe is reported per encoder. Exposure-time regression is demoted to exploratory and uses leave-six-contiguous-fields-out block cross-validation.

| encoder | 3-class BA (mean ± sd) | 3-class within-bin | round vs amoeboid BA | field identity (lower better) | ρ exposure time |
|:---|---:|---:|---:|---:|---:|
| E4 TiJepa-1 | 0.623 ± 0.019 | 0.623 | 0.828 ± 0.006 | 0.057 | -0.575 |
| E3 DINOv2 ViT-S | 0.608 ± 0.012 | 0.609 | 0.766 ± 0.015 | 0.111 | -0.108 |
| E2 ImageNet ViT-S | 0.585 ± 0.029 | 0.586 | 0.747 ± 0.014 | 0.116 | -0.145 |
| E6 random init (3 draws) | 0.541 ± 0.031 | 0.541 | 0.773 ± 0.006 | 0.036 | -0.528 |
| E1 hand-crafted (segmentation + Zernike + Haralick) | 0.459 ± 0.006 | 0.459 | 0.675 ± 0.022 | 0.034 | -0.806 |

### 2.5 Point tracking and latent forecasting on held-out movies

Point tracking uses all 6 movies the S-BIAD2196 deposit contains, with the leakage-matched pretraining movies excluded as described above. The same tracker is used for every encoder: the query cell's patch token is matched to candidate tokens inside a search window, with an identical motion prior. Six movies is below the ten-group minimum we set for interval estimates, so no bootstrap confidence interval is reported; per-movie values and an exact sign-flip permutation against the baseline are given instead.

| method | ⟨δ⟩ (1–16 µm) | median survival (frames) | vs baseline, exact permutation |
|:---|---:|---:|---:|
| E0 TiJepa-0 (v0 reference, different corpus) | 0.686 | 2 | p = 0.156 |
| E6_random_init_draw0 | 0.666 | 2 | p = 0.188 |
| E6_random_init_draw1 | 0.654 | 2 | p = 0.188 |
| E6_random_init_draw2 | 0.653 | 2 | p = 0.188 |
| baseline_nearest_detection | 0.488 | 6 | — (is the baseline) |

The nearest-detection baseline reaches ⟨δ⟩ 0.488. 

Because the v1 objective trains the predictor to produce future latents, we can ask directly whether it forecasts. Each model is given the first 12 frames of a held-out clip and predicts its own target-encoder latents for the last 4, scored against two baselines in its own latent space: copying the last observed latent, and the mean context latent. **Only within-model ratios are comparable** — each model has a different latent space, so raw L1 across models would be meaningless — and a ratio below 1 means the predictor beats assuming nothing changes.

| model | L1(predictor) / L1(copy-last) | / L1(mean latent) |
|:---|---:|---:|
| E0 TiJepa-0 (v0 reference, different corpus) | 0.869 | 0.796 |
| E6_random_init_draw1 | 2.891 | 1.633 |
| E6_random_init_draw2 | 2.899 | 1.663 |
| E6_random_init_draw0 | 2.957 | 1.697 |

### 2.6 Part B: signalling dynamics, tolerance, transcription and accessibility

Nothing in this section is reported unless its pipeline first passes a positive control. For expression, `media->LPS` versus `media->media` in GSE260996 recovers 7 of 7 canonical LPS/NF-κB targets with log2 fold change above 1 and FDR below 0.05 (2,757 differentially expressed genes in total): PASS. For accessibility, `ML` versus `MM` in GSE261647 raises accessibility at peaks within 10 kb of all 7 of those genes' transcription start sites, 1.096 logCPM against 0.072 genome-wide (p < 0.0001): PASS. The two matrices are also checked mechanically against their GEO records on row-identifier class, sample count and assay type, which is how the previous version's confusion of an accessibility matrix for an expression matrix is prevented from recurring.

The ligand codes in the trace deposit are now taken from the deposit's own plotting code, which assigns them with doses in trailing comments, rather than inferred: 1 = CpG 100 nM, 2 = LPS 1 ng/ml, 3 = polyI:C 1000 ng/ml, 6 = TNF 10 ng/ml. The GEO treatment protocol for the matching ATAC series states the same four doses independently. A biological sanity check runs before any tolerance result: the NF-κB second-response amplitude for LPS->LPS is 1.0585 against 1.3896 for naive->LPS at the matched 8.0 h interval, i.e. lower, as acute endotoxin tolerance predicts.

We then ask whether a classifier can tell an acutely endotoxin-tolerised macrophage from a naive one at the single-cell level. Every layout-verified file was searched: the design supplies 4 wells at the matched 8 h interval (1304 cells), the only interval at which homologous restimulation was performed. Leave-one-well-out balanced accuracy from the raw trace is 0.547 ± 0.005. With 4 wells the exact permutation null over all 6 distinct well-label assignments gives p = 0.333, and **no p below 0.167 is achievable whatever the classifier does**. We therefore report no confidence interval and draw no conclusion about tolerance from dynamics: the public deposit cannot support one. The previous version reported this contrast as 0.701 with a bootstrap interval over the same four wells. A secondary multi-class task — identifying which pair of ligands a cell saw — reaches 0.269 ± 0.005 across 11 classes against a chance level of 0.091, so the traces do carry stimulus information even where they cannot resolve priming history.

Condition-level dynamics were then related to the two molecular readouts **separately and under their own names** — expression from GSE260996 and chromatin accessibility from GSE261647 — matching trace conditions to omics conditions on stimulus identity and on the 4 h inter-stimulus interval the GEO protocol states. Both are null: for RNA expression, 358 of 2000 targets reach leave-one-condition-out r > 0.5 against a permutation null mean of 334.9 (p = 0.325, exact over all 120 condition orderings); for chromatin accessibility, 463 of 2000 targets reach leave-one-condition-out r > 0.5 against a permutation null mean of 352.9 (p = 0.317, exact over all 120 condition orderings). A pre-specified NF-κB target panel, analysed separately, is null as well. With five matched conditions this is descriptive and we label it so.

Finally we asked whether genes whose LPS induction is blunted by prior priming in mouse overlap the genes changed by LPS tolerance in human. On the mouse side 1,440 genes are LPS-induced in the naive arm and 1,142 of them show reduced induction after both CpG and polyI:C priming, giving 1,047 with human orthologs. On the human side a moderated paired test across 4 donors finds 37 genes down and 48 up at FDR 0.05. The overlap is **0 genes** (Fisher p = 1.000). We pre-registered that these two contrasts are not the same biology — mouse heterologous priming 4 h apart in bone-marrow-derived macrophages against human homologous tolerance 6 days apart in monocyte-derived macrophages — so a null is the expected outcome and is not evidence about either one.

### 2.7 Part C: a hypothesis-generating bridge, at rest and on restimulation

**This section pairs no image with any molecular sample.** Anchors and trained/tolerant
cells come from different studies, different donors and different platforms. It is
presented to generate a testable prediction, not to support a claim.

Activation axes are built from GSE46903 with two requirements the previous version did not impose: each anchor comes from **donor-paired** stimulus-versus-control samples analysed with a moderated paired test, with at least three donor-matched pairs, and each must pass a **sanity rule** requiring canonical marker genes among its top 100. The result is that the IFNg anchor is built from 7 donor-matched pairs and passes its sanity rule with 7 of the required 5 canonical markers in its top 100 (GBP1, GBP5, CXCL9, CXCL10, IRF1); the IL4 anchor has 7 donor-matched pairs but **fails its sanity rule**, with only 2 of 3 canonical markers in its top 100; **no LPS anchor can be built at all**: the atlas provides only 2 donor-matched pairs against a minimum of 3. Only one axis therefore survives, and the two-axis M1/M2 framing collapses to a single interferon-γ axis. This is also the mechanical explanation of the previous version's LPS signature, which consisted of zinc-finger genes with no canonical LPS target in it: it was built by comparing across different donors in a 73-donor study.

Scoring GSE58310 on the IFNg axis: beta glucan trained sits -0.0078 from the day-6 control (exact sign-flip p = 0.125, floor 0.125); LPS tolerised sits -0.0027 from the day-6 control (exact sign-flip p = 0.750, floor 0.125). These are **resting** day-6 cells, and trained immunity is defined by the response to restimulation, so proximity to baseline here is the expected result and is not evidence against training. Naive monocytes are excluded from the bridge and shown separately, because their displacement reflects monocyte-to-macrophage differentiation rather than activation.

The test that actually bears on trained immunity is restimulation, and two public datasets qualify: monocytes trained, differentiated for five days and re-exposed to LPS for four hours, with matched donors. We score a pre-specified panel of canonical human LPS/NF-κB genes, because the only anchor that survived the sanity rules is interferon-γ-derived while the restimulation here is LPS — a mismatch we state rather than paper over.

| dataset | arm (vs untrained → LPS) | Δ mean log expression | direction | as predicted | exact p |
|:---|---:|---:|---:|---:|---:|
| GSE212282 | beta glucan trained | -0.336 | lower | **no** | p = 0.250 |
| GSE212282 | itaconate trained | 0.164 | higher | yes | p = 0.250 |
| GSE212282 | 4OI trained | 0.275 | higher | yes | p = 0.250 |
| GSE212282 | beta glucan plus 4OI | -0.008 | lower | yes | p = 1.000 |
| GSE185433 | LPS tolerised | -0.169 | lower | yes | p = 0.750 |
| GSE185433 | IL4 trained | 0.276 | higher | yes | p = 0.250 |
| GSE185433 | IL4 plus LPS | -0.435 | lower | **no** | p = 0.500 |

With three donors the exact sign-flip null has eight assignments and no p below 0.25 exists, so none of these is significant and the direction is what the test establishes. Tolerance points the way tolerance should, and three of the four training arms point the way training should — but **β-glucan, the canonical training stimulus, points the other way**. We report that as observed.

The projection of omics coordinates into the learned image space is **not performed**, for two independent reasons either of which would be enough. It needs image-space M0/M1/M2 centroids, and the only dataset that provides them was not available (above); and it needs an omics anchor whose stimulus matches an image label, which none of the surviving anchors does. Substituting a different label set — BBBC054's microglial morphology classes are not macrophage polarisation states — and presenting the result as this analysis would misrepresent what was measured, so it is left undone. The previous version, unable to run it, substituted a one-axis analysis that expressed a human monocyte score as an equivalent LPS exposure time on a mouse microglia axis anchored at the extremes of the quantity being mapped; that analysis was circular and has been deleted from the code, the results and this paper rather than relabelled. (2 blocking requirements recorded.)

---

## 3. Discussion

**What this work shows.** Self-supervised video pretraining on openly licensed
innate-immune microscopy is practical at modest scale, and the resulting frozen
representation can be compared against strong baselines under leakage-controlled
protocols. The future-masking objective gives the model a forecasting task that can be measured directly, and the forecasting evaluation separates models that have learned temporal structure from those that have not.

**What it does not show.** It does not show that TiJepa-1, or any model here, identifies
trained immunity from images. **No trained or tolerant cell was imaged in any dataset
used in this work.** The Part B tolerance contrast cannot reach significance with the wells the public deposit provides. The dynamics-to-expression, dynamics-to-accessibility and cross-species analyses are null. The Part C bridge is an interpolation across studies, and its restimulation test is not significant with three donors and is inconsistent in direction across training stimuli.

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

**Statistical power is the second constraint.** The Part B tolerance contrast has 4 wells; the dynamics-to-omics regressions have 5 matched conditions; the human tolerance contrast has 4 donors; the restimulation test has 3. Published label-free classifiers show the same pattern — near-ceiling same-donor accuracy collapsing to 72-85% across donors — so donor and batch count, not model capacity, should dominate the design of the experiment above.

**Future work.** The learned representation is intended as one modality among several in
multi-omic models of immune disease. That integration is future work and nothing here
bears on it.

---

## 4. Methods

### 4.1 Compute

All work ran on a single node with one NVIDIA A100-SXM4-40GB, 30 logical CPU cores and
216 GB of RAM, with all data on local disk. Tier T1 in our own plan: one GPU, a ViT-S
encoder, bf16. Pretraining therefore used a **single seed**; the seed-to-seed variation
reported in Part A comes from independent random-init draws, stratified cell
subsamples and fold assignments, not from repeated pretraining. Exact versions are
pinned in `requirements-lock.txt`; every stage records its own torch version, which was
not recorded throughout.

### 4.2 Data and preprocessing

Pretraining video is the ComplexEye collection (Zenodo 10.5281/zenodo.10082674, CC BY
4.0), downloaded one archive at a time with MD5 verification against the deposit record,
preprocessed, and deleted before the next was fetched so that peak disk use is one
archive plus the output shards. Cells are detected with a temporal-median background
model and a MAD threshold, linked with trackpy, and cut into 16-frame 96x96 clips
centred on each tracked cell at temporal strides 1, 3, 9 and 27. One clip in ten is made
static by repeating its middle frame.

Pixel sizes are derived, not assumed. The ComplexEye imager is an OmniVision OV9712 with
a 1280x800 array over a 3888x2430 µm active area, i.e. 3.0375 µm per sensor pixel, at
4.69x magnification, giving 0.648 µm/px in the sample plane; the deposit's own metadata
states 0.641 µm/px for the same instrument and we use that, the two agreeing within 1%.
The reference microscope is a Leica DMI6000 B whose published field of view is
662.9 x 495.2 µm; those movies are 1392 x 1040 px, so the pixel size is 0.4762 µm/px on
both axes. Everything is resampled to a common 0.65 µm/px.

For BBBC054 the page states a Leica DMI 6000, 20x objective and an Andor
AND-DG-152V-C1E-FI camera. The Andor Neo 5.5 sCMOS specification gives a 2560 x 2160
array at 6.5 µm pitch; the BBBC054 images are 1080 x 1280, exactly half in each axis,
i.e. 2x2 binning, so the sample-plane pixel size is 2 x 6.5 / 20 = 0.65 µm/px — already
the common target. Because the camera model code could not be confirmed by name, a
scale-sensitivity analysis over resampling factors 0.5, 0.75, 1.0 and 1.5 is also
reported.

BBBC054 annotation coordinates follow the data rather than the dataset page: the page
states that axis-1 is the x-coordinate, but across all 47,043 annotations axis-1 spans
-4..1080 and axis-2 spans -6..1284 in images that are 1080 rows by 1280 columns, so
axis-1 indexes rows. Following the page transposes every crop.

### 4.3 Leakage check

Every S-BIAD2196 evaluation movie was fingerprinted against every candidate pretraining
movie: frames 0, n/2 and n-1, block-averaged to 40x25, z-scored and concatenated,
compared by Pearson correlation, with the SHA-1 of the raw first frame recorded
separately. A pair counts as a match when the frame dimensions are identical and
r > 0.98, or when the first frames are byte-identical; all pairs above r = 0.90 are
reported whatever their shapes, so the decision is auditable rather than hidden behind a
threshold. Matching **pretraining** movies are excluded.

### 4.4 Model and objective

TiJepa-1 is a V-JEPA-style joint-embedding predictive architecture: a ViT-S context
encoder sees unmasked tubelets, an EMA target encoder sees the whole clip, and a narrow
predictor maps context latents to target latents at masked positions, with an L1 loss in
latent space and no pixel reconstruction. Tubelets are 2x8x8 over 16x96x96 clips.

The v1 objective mixes two masks, sampled per batch with equal probability. The
**spatial tube mask** is the multi-block mask of the original architecture, applied at
every temporal position. The **future mask** makes every spatial position of the last two
temporal slots — the last four frames — a prediction target, leaving the first twelve
frames as context, which makes the predictor a latent forecaster. The two losses are
logged separately because they are not on the same scale. The ablation TiJepa-1-spatial
uses spatial masking only on the identical corpus for the same number of steps.

### 4.5 Probes and statistics

All probes are frozen-feature probes: encoders are never fine-tuned. BBBC054 uses
`StratifiedGroupKFold` grouped by field of view with a standardise-PCA-standardise
pipeline applied identically to every encoder, and three seeds that each vary a
stratified cell subsample, the fold assignment, and the random-init draw.

Interval estimates are reported only where there are at least ten grouping units.
BBBC054 has 60 fields, so encoder pairs there get a paired cluster bootstrap over fields
(2,000 resamples, both encoders scored on the same resample) with Benjamini-Hochberg
correction within each task. Part B's tolerance contrast has four wells and the tracking
evaluation has six movies, so both report per-group values and an **exact** permutation
distribution with its minimum achievable p, and no confidence interval. Where a
permutation can be enumerated exactly it is; otherwise at least 1,000 permutations are
drawn and the floor is reported.

### 4.6 Tracking and forecasting

Tracking: a query point at the first frame of a ground-truth track is propagated by
matching its patch token to candidate tokens within a search window, identically for
every encoder, against a nearest-detection linking baseline. Metrics are the TAP-Vid-style
⟨δ⟩ averaged over 1, 2, 4, 8 and 16 µm thresholds, and median track survival at 8 µm.

Forecasting: each model receives the first 12 frames and predicts its own target-encoder
latents for the last 4 through its own predictor under the future mask, scored as
L1(predictor)/L1(copy-last-observed-latent), with a mean-context-latent baseline.
Only within-model ratios are comparable.

### 4.7 Part B

Single-cell NF-κB traces come from the Tay-lab deposit. The matrix layout was
reconstructed from the deposit's own MATLAB preprocessing code; the numeric ligand codes
come from its plotting code, which states them with doses, and are independently
confirmed by the GEO treatment protocol of the matching ATAC series. Column 128 is the
inter-stimulus interval in hours: the deposit's code computes it from whether the ligand
list repeats a value, which reads like a homologous-versus-heterologous flag, but the
pair (naive, LPS) occurs at both 4 and 8 and every heterologous pair occurs at both
values, so it cannot be a function of ligand identity. Homologous restimulation exists
only at 8 h, which is why the matched comparison is at 8 h.

Expression is GSE260996, assembled from the per-sample Rsubread featureCounts objects
GEO deposits, keyed by Entrez gene ID; it is identical to the merged count matrix the
deposit itself ships. Accessibility is GSE261647. Both pass a data-type guard that checks
row-identifier class, sample count and assay type against the GEO record, and differential
expression uses DESeq2 on counts.

### 4.8 Part C

Anchors are built from GSE46903 with donor-paired moderated tests requiring at least
three donor-matched pairs and passing a canonical-marker sanity rule. GSE46903's M-CSF
arm has its only control at 0 h, i.e. undifferentiated monocytes, so a 72 h M-CSF
contrast would conflate differentiation with stimulus; GM-CSF is used and the mismatch
with the human-serum differentiation of the target datasets is a stated limitation.
Scores are rank-based single-sample scores, which are comparable across platforms.
`inmoose` could not be built in this environment, so the moderated tests use a
pure-Python implementation of the limma empirical-Bayes method, validated in
`tests/test_moderated.py` against hand-computable cases and against the limiting
behaviour that defines the method.

### 4.9 Licences

Pretraining used only CC BY 4.0 data. Datasets excluded on licence grounds were never
downloaded. BBBC054 is CC BY 3.0; S-BIAD2196 is CC BY 4.0; GEO series are US-government
public-domain records cited to their originating publications.

### 4.10 AI-assistance disclosure

Code, analyses and an initial manuscript draft were produced with the assistance of an
AI coding agent (Claude, Anthropic) under the author's direction; the author reviewed
and is responsible for all content. <!-- TODO(author): review -->

### 4.11 Competing interests

<!-- TODO(author): declare competing interests -->


---

## 5. Data and code availability

Code: <https://github.com/rafidquadir-eng/TiJepa>. Every number in this paper is written
by `scripts/91_make_paper_v1.py` from the JSON files in `results/v1/`; no value is typed
by hand, and every directional phrase is generated from an interval or an exact
permutation p and then re-verified against the results by `scripts/92_check_claims.py`.

All datasets used are public and are listed with their licences and verified sizes in `manifests/datasets.yaml` and `results/v1/downloads/`. The two datasets that required manual retrieval (Dryad doi:10.5061/dryad.m0cfxpp8v and the LTDB figshare collection) were not available for this run and the analyses that depend on them are reported as not performed, with their code in place. A defect in the published ComplexEye archive R06.zip is documented in `results/v1/complexeye_archive_defect.json`.

Model weights: see `reports/WEIGHTS.md`. They are not published.

---

## References

Every identifier below was resolved against Crossref or DataCite during this run, and each resolved title was checked for the keywords the work should contain, because a DOI can resolve cleanly and still be the wrong paper. Nothing is listed that did not resolve.

| key | first author | title | venue/year | identifier |
|:---|---:|---:|---:|---:|
| vjepa | — | — | — | arXiv:2404.08471 |
| dinov2 | — | — | — | arXiv:2304.07193 |
| complexeye | Zülal Cibir | ComplexEye: a multi-lens array microscope for high-throughput embedded i | Nature Communications | doi:10.1038/s41467-023-43765-3 |
| phagostat | Mehdi Ounissi | PhagoStat a scalable and interpretable end to end framework for efficien | Scientific Reports | doi:10.1038/s41598-024-56081-7 |
| bbbc | Vebjorn Ljosa | Annotated high-throughput microscopy image sets for validation | Nature Methods | doi:10.1038/nmeth.2083 |
| saeed2014 | Sadia Saeed | Epigenetic programming of monocyte-to-macrophage differentiation and tra | Science | doi:10.1126/science.1251086 |
| novakovic2016 | Boris Novakovic | β-Glucan Reverses the Epigenetic State of LPS-Induced Immunological Tole | Cell | doi:10.1016/j.cell.2016.09.034 |
| xue2014 | Jia Xue | Transcriptome-Based Network Analysis Reveals a Spectrum Model of Human M | Immunity | doi:10.1016/j.immuni.2014.01.006 |
| netea2016 | Mihai G. Netea | Trained immunity: A program of innate immune memory in health and diseas | Science | doi:10.1126/science.aaf1098 |
| wang2025 | Andrew G. Wang | Macrophage memory emerges from coordinated transcription factor and chro | Cell Systems | doi:10.1016/j.cels.2025.101171 |
| limma | Matthew E. Ritchie | limma powers differential expression analyses for RNA-sequencing and mic | Nucleic Acids Research | doi:10.1093/nar/gkv007 |
| deseq2 | Michael I Love | Moderated estimation of fold change and dispersion for RNA-seq data with | Genome Biology | doi:10.1186/s13059-014-0550-8 |
| cellpose | Carsen Stringer | Cellpose: a generalist algorithm for cellular segmentation | Nature Methods | doi:10.1038/s41592-020-01018-x |
| cotracker | — | — | — | arXiv:2307.07635 |
| smyth2004 | Gordon K Smyth | Linear Models and Empirical Bayes Methods for Assessing Differential Exp | Statistical Applications in Genetics and Molecular Biology | doi:10.2202/1544-6115.1027 |
| mgi | Richard M Baldarelli | Mouse Genome Informatics: an integrated knowledgebase system for the lab | GENETICS | doi:10.1093/genetics/iyae031 |
| trackpy | Allan, Daniel B. | trackpy: Trackpy v0.4.1 | Zenodo | doi:10.5281/zenodo.1226458 |
| scikit_image | Stéfan van der Walt | scikit-image: image processing in Python | PeerJ | doi:10.7717/peerj.453 |
