# TiJepa

Self-supervised imaging / video modelling of **trained immunity** and innate immune
activation states (monocytes, macrophages, neutrophils, NK cells, dendritic cells),
linked to **iMEMORY**, a multi-omic model of immune disease.

Target states: naive · trained (β-glucan, BCG, oxLDL) · tolerant (LPS pre-exposure) ·
classically activated / polarized (LPS, IFN-γ, IL-4, M1/M2).

## Current contents

This repo currently holds the landscape review that determines whether the imaging data
we need already exists or must be generated.

| Path | What it is |
|---|---|
| [`reports/Trained immunity imaging datasets.md`](reports/Trained%20immunity%20imaging%20datasets.md) | The synthesized report (~8.5k words): deposit tables, accession tables, licence constraints, gaps, recommendations |
| `research_notes/Trained immunity imaging datasets/` | Raw per-stream research notes behind the report |

### Research notes

| File | Scope |
|---|---|
| `imaging_of_trained_cells.md` | Papers/labs imaging trained or tolerant innate cells; deposit status per data-availability statement |
| `labelfree_qpi_imaging.md` | Brightfield / phase / QPI / holographic / Raman imaging of macrophage activation; repository sweep |
| `flim_autofluorescence.md` | NAD(P)H / FAD autofluorescence and FLIM metabolic imaging of innate immune activation |
| `trained_immunity_omics.md` | Public trained-immunity omics accessions (GEO/EGA/PRIDE/Metabolomics Workbench) for target signatures |
| `timelapse_video_datasets.md` | Downloadable immune time-lapse / video corpora for pretraining and point tracking |

## Headline finding

**No public imaging or video dataset of trained, tolerant, or innate-memory immune cells
exists.** The imaging + training/tolerance dataset has to be generated in-house.

Load-bearing evidence:

- Europe PMC full-text search for `"trained immunity"` × (BioImage Archive / IDR / `S-BIAD` / `idr00`) → **0 hits**
- BioImage Archive: 59 macrophage/monocyte/neutrophil studies, **none** a β-glucan, BCG, oxLDL, MDP or LPS-tolerance study
- All **147 IDR projects + 105 screens** enumerated → no innate-immune training study, and no innate-immune time-lapse at all

Two partial exceptions (detailed in the report, not treated as a flat zero):

1. **Carlile et al., *iScience* 2024** (McLoughlin + Monaghan, TCD) — NAD(P)H 2P-FLIM of human
   MDMs trained with heat-killed *S. aureus*, days 3 and 7. Real training biology, real
   label-free imaging; data "on request" only, with a self-contradictory availability statement.
2. **BioImage Archive `S-BIAD829`** — the one deposited image set labelled *trained immunity*
   (67 files, CC0), but fixed mouse brain immunofluorescence, not live cells under a
   prime–rest–restimulate design.

What *is* reusable: the trained-immunity omics canon is almost entirely open in GEO on matched
donors, and ~290 GB of permissively licensed (CC BY 4.0 / MIT) innate-immune video exists for
self-supervised pretraining.

## Two caveats that shape experimental design

- **Donor variance is the binding constraint, not class separability.** Across every published
  label-free classifier, ~98% same-donor accuracy collapses to 72–85% multi-donor / multi-class.
  Budget donors accordingly.
- **Licence traps.** The only video dataset carrying per-cell activation labels
  (Zenodo `10.5281/zenodo.16319732`) is CC BY-**ND** — no-derivatives blocks training on it
  without an author query. DeepCell DynamicNuclearNet and the Cellpose training set are
  non-commercial and access-gated.

## Caveat on completeness

Several items remain **unverified** behind publisher blocks and are flagged as such in the
report's gaps section. Highest priority: **Fanucchi et al., *Nature Genetics* 2019**
(Mhlanga lab, UMLILO immuno-RNA-FISH) — the best remaining candidate for deposited
trained-immunity microscopy.

Correction carried forward: there is no "Mulder et al. *Cell* 2021 (300BCG)". The 300BCG
flagship is **Moorlag et al., *Immunity* 2024** (PMID 38198850); Mulder K. et al.,
*Immunity* 2021 (PMID 34331874) is a separate cross-tissue monocyte/macrophage atlas.
Also: BioStudies `S-EPMC*` records are PubMed Central literature stubs, **not** image
deposits — only `S-BIAD*`, `S-BSST*`, `EMPIAR-*` and `idr####` are.
