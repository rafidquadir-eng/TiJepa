# Next project: designs

Selection and evidence: `reports/NEXT_PROJECT_SELECTION.md`. Winner **A** (MitoCheck
action-conditioned latent world model); backup **F** (drug-conditioned live-cell world model).
Both use a **frozen V-JEPA 2** encoder and train only an action-conditioned predictor, as in
V-JEPA 2-AC. No backbone is retrained.

---

# A. GeneWorld: an action-conditioned latent world model of genome-scale live-cell RNAi time-lapse

## A.1 Question

Can a frozen, general-purpose video world-model encoder plus a small action-conditioned
predictor, trained on two-day live-cell movies of ~21,000 single-gene knockdowns, (1) predict how
a cell population's latent state evolves under a given knockdown, (2) do so for **genes it has
never seen**, using gene representations learned from independent modalities, and (3) **plan**:
given a starting state and a target future state, search over gene actions to find knockdowns
that reach it?

## A.2 Hypotheses (each with a pre-specified test and a meaning for either outcome)

- **H1 — prediction.** On held-out wells of seen genes (held out by plate), the
  action-conditioned predictor forecasts future V-JEPA 2 latents better than (a) the same
  predictor with the action removed, (b) copy-last, (c) the time-slot-conditional mean, and
  (d) the plate's negative-control trajectory. *If false:* knockdown identity adds no
  predictable information to the latent dynamics at these horizons — itself a measurement of
  how much of a two-day RNAi phenotype is forecastable from early frames plus the gene.
- **H2 — zero-shot to unseen genes.** On genes held out entirely, the predictor fed a gene's
  embedding beats the same predictor fed shuffled embeddings and a k-nearest-gene lookup
  baseline. The four encoding sources (Perturb-seq, JUMP/PERISCOPE morphology, ESM-2 protein,
  STRING/GO network) are compared. *If false:* the transferable part of a gene's mitotic
  phenotype is not carried by any of these encodings; the source ranking is still reported.
- **H3 — planning.** For a held-out target (the latent trajectory of a held-out gene's well),
  latent model-predictive search over the ~21,000 gene actions (exhaustive scoring, then CEM
  in embedding space) ranks the true gene, a gene from the same complex (CORUM) or the same
  MitoCheck phenotype class above chance, and **better than static endpoint retrieval**
  (matching only the final frame's embedding, the MolPhenix-style retrieval analogue).
  *If false:* dynamics add nothing to perturbation identification beyond the endpoint state.

## A.3 Data and subsets

| set | content | use | size to download |
|---|---|---|---|
| MitoCheck screenA (IDR idr0013, S-BIAD865, CC0) | 191,368 movies, 93 frames at 30 min, 1344 x 1024, H2B-GFP HeLa, ~21,000 genes | train / val / test by the splits below | stage 1: 40 plates (~1.1 TB, ~15,000 movies); stage 2 up to 120 plates (~3.4 TB) if throughput allows |
| MitoCheck screenB (validation, CC0) | 9,627 movies, 1,127 genes x 2 new siRNAs | independent-reagent test of H1–H3 | ~1 TB (or a subset) |
| idr0002 (CC BY 4.0) | 1,152 movies, 100 genes, 330 frames, 2 channels | transfer to another lab and protocol | 2.1 TB (or the H2B channel subset) |
| idr0061 (CC BY 4.0) | 5,952 movies, 1,200 genes x 4 siRNAs, 180 frames | transfer to another live readout (spindle positioning) | stretch goal |
| MitoCheck annotations | per-well phenotype scores; 3,277 annotated cells (16 classes); hit lists | probes, phenotype-class evaluation of H3 | small |
| Gene encodings | Replogle K562 GWPS and RPE1 pseudobulk (CC BY 4.0; 0.1–0.4 GB); JUMP CRISPR + ORF and PERISCOPE consensus profiles (CC0); ESM-2 (MIT) mean embeddings; STRING (CC BY 4.0) node embeddings; GO | action encoders for H2/H3 | < 20 GB |

Download plan: EBI HTTP with 64 parallel byte-range readers (measured 30.9 MB/s) reading
**only the needed wells** out of the per-plate OME-Zarr zips (remote zip index + range requests),
plus a one-day test of Aspera/Globus, which may be faster. 1.1 TB = ~10 h at the measured rate.
Downloaded raw data are reduced on arrival to 256 x 256 (or 384 x 384) tiles per well and
deleted after embedding.

## A.4 Splits

- **By plate:** within seen genes, whole plates are held out for H1 (controls plate and batch
  effects).
- **By action (gene):** 20% of genes held out entirely for H2/H3, stratified by MitoCheck
  phenotype class and by availability in each encoding source; a second split holds out whole
  **complexes** (CORUM) so that near-duplicate genes cannot leak.
- **By reagent:** screenB's independent siRNAs for genes seen with other siRNAs in screenA.
- **By lab/dataset:** idr0002 (and idr0061 if reached) as zero-shot transfer sets.
- Model and recipe selection only on a validation split of plates and genes; test splits are
  touched once, after freezing.

## A.5 Action encodings

Each gene a is encoded by a frozen vector e(a) from one source; a small learned adapter maps it
to the predictor's action space. Sources: (1) Replogle Perturb-seq pseudobulk response
(K562 GWPS; RPE1), PCA-reduced; (2) JUMP CRISPR/ORF and PERISCOPE morphological profiles;
(3) ESM-2 650M mean-pooled protein embedding; (4) STRING network embedding; (5) a learned
per-gene embedding (seen genes only; the upper bound for H1). siRNA identity is encoded as an
extra token so that reagent effects are separable. Genes missing from a source are evaluated
only with sources that cover them (coverage reported per source).

## A.6 Model

- **Encoder:** frozen V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`, MIT, pinned revision), on
  16-frame clips (8 h at 30 min) of 256 x 256 tiles, mean-pooled to tile tokens. A frozen DINOv2
  per-frame encoder and the TiJepa-1 encoder are alternative backbones (baselines).
- **State:** a well's state at time t is the set of tile-token latents of the clip ending at t.
- **Predictor:** a block-causal transformer (~50–100 M parameters, as in V-JEPA 2-AC) taking past
  latents, the action token e(a), and the elapsed time since transfection, and predicting the
  latents of the next clip; trained with teacher forcing plus a short rollout loss (L1), as in
  V-JEPA 2-AC.
- **Planner:** given a start state z0 and target z*, score candidate genes by rolling the predictor
  out to the target horizon and measuring latent distance to z*; exhaustive over the ~21,000
  genes (cheap: one batched rollout per gene), then cross-entropy-method refinement in the
  encoding space for continuous search (and to propose **combinations** as sums of encodings,
  reported as hypotheses only).

## A.7 Controls (built in from the start; the lessons of TiJepa)

1. **Time-slot / position control:** a predictor given only elapsed time and plate (no action,
   no past latents) and the time-slot-conditional mean latent fitted on other plates — the
   action-conditioned model must beat both, not only copy-last.
2. **Static control:** clips made of one repeated frame; the predicted change must vanish.
3. **Context swap:** past latents from another well with the same action; measures how much
   the prediction uses the well's own history versus the action.
4. **Action shuffle / random embedding:** the H2 null.
5. **Plate and position confounds:** plate-held-out splits, well-position covariates, and a
   probe that tries to decode plate identity from predicted latents.
6. **Leakage:** the duplicate check from TiJepa (fingerprints) between splits and transfer
   sets; genes with identical siRNA sequences grouped.

## A.8 Baselines

Copy-last; negative-control mean trajectory; time-slot mean; action-agnostic predictor;
k-nearest-gene trajectory lookup in each encoding space; a phenotype-score model (MitoCheck's
per-well scores regressed from gene encodings, for H2); a frozen DINOv2 per-frame encoder with
the same predictor; static endpoint retrieval for H3.

## A.9 Metrics

Latent forecast error relative to each baseline at horizons 4, 12, 24 and 40 h (per plate,
exact or bootstrap tests over plates/genes as group counts allow); phenotype-class accuracy of
a linear probe applied to predicted versus observed latents; for H2 the same metrics on unseen
genes with CIs over genes; for H3 hit@1/10/100, mean reciprocal rank, same-complex and
same-phenotype-class hit rates, against permutation nulls; calibration of the planner's scores.

## A.10 Decision gates

| gate | condition | if yes | if no |
|---|---|---|---|
| G-Data (day 2) | >= 10,000 movies decoded and embedded | continue | switch the stage-1 subset to fewer plates, start F in parallel |
| G-Signal (day 6) | H1: action-conditioned beats action-agnostic and time-slot mean on held-out plates (p < 0.05 over plates) | go to H2/H3 | report H1 as the result (with controls); run H2/H3 only as exploratory |
| G-ZeroShot (day 9) | H2: at least one encoding source beats shuffled embeddings with a CI over genes excluding 0 | planning with that source | planning with learned embeddings on seen genes only; H2 reported as negative with the source ranking |
| G-Plan (day 11) | H3: hit@100 above permutation and above endpoint retrieval | report planning | report the planning negative and the endpoint-vs-dynamics comparison |

## A.11 Compute estimate (one A100-40GB)

- V-JEPA 2 ViT-L embedding: measured on this instance at ~57 ms per 16-frame 256 x 256 clip with
  a shared GPU. 15,000 movies x ~11 clip positions x 4 tiles ≈ 660,000 clips ≈ 10 GPU-h;
  30,000 movies ≈ 20 GPU-h.
- Predictor training: ~50–100 M parameters on cached latents, ~1 GPU-day per configuration,
  3–5 configurations.
- Planning: exhaustive rollouts over 21,000 genes x a few thousand targets ≈ 1–2 GPU-h.
- Storage: cached latents ~1–2 TB at fp16 for 30,000 movies before pooling; ~100 GB pooled.

## A.12 Two-week timeline

| days | work |
|---|---|
| 1–2 | annotation tables, gene/siRNA mapping (HGNC), split definition, download stage 1, Aspera/Globus test, remote-zip reader; G-Data |
| 3–4 | embedding pipeline (V-JEPA 2, DINOv2, TiJepa-1), cached latents, controls data (static, swap) |
| 5–6 | predictor + H1 with all controls; G-Signal |
| 7–9 | action encoders (Perturb-seq, JUMP/PERISCOPE, ESM-2, STRING); H2; G-ZeroShot |
| 10–11 | planner, H3, endpoint-retrieval comparison; G-Plan |
| 12 | screenB and idr0002 transfer tests |
| 13–14 | figures, report, manuscript draft, backups |

## A.13 Venue

TMLR (claims-supported empirical study; partial negatives acceptable), or a world-model /
ML-for-biology workshop (ICLR/NeurIPS LMRL, GEM, or world-model workshops) for a short version.

---

# F (backup). DrugWorld: a drug-conditioned live-cell world model with dose and combination planning

## F.1 Question

Can the same frozen-encoder + action-conditioned predictor forecast live-cell dynamics under
**compounds** (encoded by structure and dose), transfer to **unseen compounds**, and plan a
**drug, dose or combination** that reaches a target cell-state trajectory — evaluated against
the measured dose grids and combinations?

## F.2 Hypotheses

- **H1:** on held-out wells/plates, compound- and dose-conditioned prediction of the latent
  trajectory (idr0119 phase + FUCCI movies, 96 h; idr0064 Erk-KTR movies, 5 h) beats
  action-agnostic, time-slot-mean and vehicle-trajectory baselines.
- **H2:** on held-out kinase inhibitors in idr0064 (429 compounds), structure encodings (ECFP,
  a pretrained molecular encoder) beat shuffled encodings; on idr0119, held-out **doses**
  (interpolation) and held-out **combinations** are predicted from single-drug data (the test
  Gross et al. 2023 ran with a mechanistic model — a direct comparison).
- **H3:** given a target cell-count/phase trajectory, latent search over drug x dose (and pairs)
  recovers the measured condition that produced it, and ranks conditions by their measured
  effect better than a mechanistic-model baseline.

## F.3 Data

idr0119 (Gross et al. 2023; CC BY 4.0): 3,026 movies, 193 frames, 4 breast-cancer lines,
single drugs over 8 doses and combinations in AU565 (3.16 TB; EBI). idr0064 (Goglia et al.
2020; CC BY 4.0): 450 movies, 121 frames, 429 kinase inhibitors, >80,000 cells (0.07 TB). Both
fit within two days of download at the measured EBI rate. Optionally MitoCheck clips as extra
unlabelled-action dynamics for predictor pre-training.

## F.4 Splits, model, controls, metrics

Splits by plate, by compound (idr0064), by dose and by combination (idr0119), by cell line
(idr0119: leave one line out). Same frozen V-JEPA 2 encoder and predictor as A, with the action
token = [structure encoding, log dose, time since addition] (pairs as two tokens). The same
controls as A.7 (time-slot mean, static, context swap, action shuffle). Baselines: copy-last,
vehicle trajectory, action-agnostic predictor, Gross et al.'s phase-specific ODE model
(reimplemented from the paper) for doses and combinations, nearest-compound lookup. Metrics as
A.9, plus error in predicted cell counts and phase fractions decoded from latents.

## F.5 Gates, compute, timeline, venue

Gates mirror A.10 (G-Data day 1, G-Signal day 4, G-ZeroShot day 7, G-Plan day 9). Compute:
~3,500 movies x ~12 clip positions x 4–9 tiles ≈ 250,000 clips ≈ 4 GPU-h of embedding; predictor
training < 1 GPU-day per configuration. Timeline: days 1–2 data, 3–5 H1, 6–8 H2, 9–10 H3, 11–14
writing. Venue: as A; F can also run as A's second action type ("genes and drugs") if A's gates
pass early.

---

## Shared rules carried over from TiJepa

Provenance-derived labels and a claims checker from day one; time-slot/position controls
before any forecasting claim; selection only on validation splits; no claims beyond what the
controls support; every licence recorded (`manifests/datasets.yaml` for the new project);
backups of every checkpoint to a second location with verified hashes; no background loops
left running; nothing submitted or posted by the agent.
