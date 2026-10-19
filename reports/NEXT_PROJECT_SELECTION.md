# Next project: selection of an action-conditioned biological world model

Phase 0, 2026-10-19. Goal: a world model in the V-JEPA 2 / V-JEPA 2-AC sense — a model that
predicts how a biological system evolves **in latent space**, **conditioned on actions**
(perturbations), that transfers **zero-shot to unseen actions**, and that can **plan** (search
over actions to reach a target state) — on a verified white space, designed so every outcome
is informative.

Repository decision: this work lives on branch `next-project` of the TiJepa repository (no new
repository was created, so nothing new was published). The new project can be split into its
own repository later by the author.

Rules applied (from TiJepa): R1 labels exist; R2 scale and multi-dataset harvest (weighted x2);
R3 rich action encodings; R4 time/trajectory; R5 start from a pretrained world-model backbone;
R6 verified white space (weighted x2); R7 one A100, about two weeks.

---

## 1. Data inventory

Download speeds were **measured on this instance on 2026-10-19** with small tests (each
<= 50 GB, < 1 h, deleted afterwards). Speeds are per host; a resource on a host that was
measured with another resource says so.

| host | test | measured |
|---|---|---|
| EMBL-EBI HTTP/FTP (IDR mirror, BioImage Archive) | MitoCheck CellH5 (single stream, 30 min cap); `disk2.img`/`disk3.img` byte ranges | **1.1 MB/s** single stream; **20.1 MB/s** with 16 parallel ranges; **30.9 MB/s** with 64 |
| EMBL-EBI S3 (IDR OME-Zarr) | 3,000 chunks of `idr0013A/3451.zarr`, 64 parallel | **15.6 MB/s** |
| AWS S3, Cell Painting Gallery (anonymous) | 2,000 JUMP TIFFs (4.9 GB), 64 parallel | **410 MB/s** (338 MB/s on 300 files) |
| Hugging Face datasets | 20 Tahoe-100M parquet files (1.6 GB), 20 parallel | **410 MB/s** |
| figshare | Replogle RPE1 h5ad (8.7 GB), single stream | **23.4 MB/s** |
| NCBI GEO FTP | LINCS L1000 GSE92742 Level 3 (stopped at the 900 s cap) | **~45 MB/s** single stream (40.1 GB in 900 s) |
| Zenodo | 7 Open Worm Movement Database videos (1.7 GB), 7 parallel | **4.2 MB/s** |

### 1a. Imaging and video with per-sample actions (open)

| resource | modality | size | actions | outcomes / labels | time | licence | route (speed) |
|---|---|---|---|---|---|---|---|
| **MitoCheck** IDR idr0013 screenA (Neumann et al. 2010) | live HeLa H2B-GFP, widefield | **191,368 movies**, 195,840 wells, 510 plates, **18.98 TB** (IDR); raw also as 3 x 1 TB disk images; OME-Zarr 538 plate zips of 26–31 GB (S-BIAD865, ~15 TB) | siRNA per well, ~21,000 protein-coding genes (IDR lists 18,393 targets in screenA) | per-well phenotype scores (mitotic delay, polylobed, binuclear, death, ...), 1,249 mitotic hits; 16-class annotated cells (3,277 cells) | **93 frames, every 30 min over ~48 h**, 1344 x 1024 | **CC0** (IDR, S-BIAD865) | EBI HTTP/FTP/Globus/Aspera (30.9 MB/s at 64 streams; Aspera not tested) |
| MitoCheck screenB (validation) | same | 9,627 movies, 28 plates, 0.98 TB | 1,127 genes x 2 independent siRNAs | validated hits | 94 frames | CC0 | EBI (as above) |
| idr0002 Hériché (condensation) | live HeLa, 2 channels | 1,152 movies, 2.1 TB | 100 genes (siRNA) | prophase timing | **330 frames** | CC BY 4.0 | EBI |
| idr0061 Wolf (spindle positioning) | live, micropatterned cells | 5,952 movies, 9.1 TB | 1,200 kinases/phosphatases x 4 siRNAs | spindle orientation | **180 frames** | CC BY 4.0 | EBI |
| idr0064 Goglia (Erk dynamics) | live Erk-KTR keratinocytes | 450 movies, 0.07 TB, >80,000 cells | **429 kinase inhibitors** (SMILES available) | Erk activity traces | 121 frames over 5 h | CC BY 4.0 | EBI |
| idr0119 Gross (drug response) | live phase + FUCCI, 4 breast-cancer lines | 3,026 movies, 3.16 TB | ~5 drugs x 8 doses, plus combinations (AU565) | cell counts, cell-cycle phase | **193 frames** (30 min, ~96 h) | CC BY 4.0 | EBI |
| idr0088 Cox (phenomic profiling) | live reporter lines, snapshot | 739,991 images, 5.9 TB | 1,008 drugs + 169 natural products x 4 doses, 15 reporter lines | organelle/pathway phenotypes | single time point | CC BY-NC 4.0 | EBI |
| **JUMP-CP** cpg0016 | fixed Cell Painting (5 channels) | 116,753 compounds, 15,142 ORFs (12,609 genes), 7,977 CRISPR KOs; **115 TB images**, 358 TB total | compound (SMILES), ORF, CRISPR | profiles; MoA/target annotations | none (single time point) | **CC0** | AWS S3 (410 MB/s) |
| JUMP cpg0004 LINCS Cell Painting | fixed Cell Painting, A549 | ~1,570 compounds x 6 doses | compounds (shared with L1000) | profiles | none | CC0 | AWS S3 |
| **PERISCOPE** cpg0021 (Ramezani et al. 2025) | optical pooled Cell Painting | genome-wide CRISPR KO, A549 + HeLa (2 media), >30 million cells | ~20,000 genes | single-cell profiles | none | CC0 | AWS S3 |
| RxRx3-core (Recursion) | fixed 6-channel Cell Painting | 222,601 wells, 17.5 GB images + 0.5 GB embeddings | 735 CRISPR KOs, 1,674 compounds x 8 concentrations | embeddings, compound–gene links | none | Recursion end-user licence (attribution, share-alike-like; click-to-accept, not CC) | Hugging Face (host measured 410 MB/s) |
| Live Cell Painting cpg0039 | live acridine orange | small study | a few compounds | profiles | not established as time-lapse | CC0 | AWS S3 |
| Cell Tracking Challenge, LIVECell, ALFI | time-lapse / static | small | **no perturbation labels** (ALFI: few anti-mitotics) | tracks, cell cycle | yes | various | not used (fails R1/R3) |
| **Open Worm Movement Database** (Javer et al. 2018) | whole-organism video, single worm tracking | **14,874 experiments, 386 genotypes**, >2,700 ageing videos; ~0.3 GB/experiment (~4.4 TB) | genotype (mutant/strain) | Tierpsy features (8,289), behaviour | ~15 min videos | **CC BY 4.0** | Zenodo (4.2 MB/s) |
| O'Brien et al. 2024 (eLife) disease models | worm behaviour features + time series | 25 disease-model knockouts; 743 FDA drugs on unc-80 | genotype, drug (100 µM, 4 h) | behaviour features; 2 confirmed rescue drugs | feature time series (raw video not found) | Zenodo record (licence not checked) | Zenodo |
| Zebrahub | light-sheet development + scRNA | atlas | **no perturbations** | lineage, cell types | yes | not verified | CZ Biohub site |
| Drosophila blastoderm (used by Cornelis et al. 2026) | embryo velocity fields | 12 wild-type videos + perturbed | temperature, a few mutants | flow fields | yes | not verified | — |

### 1b. Perturbation omics that can supply action encodings or a paired readout (open)

| resource | size | actions | time / dose | licence | route (speed) |
|---|---|---|---|---|---|
| **Tahoe-100M** | ~100 M cells, ~60,000 drug-cell conditions, 50 cancer lines, ~1,100–1,200 drugs; 3,388 parquet files (~340 GB) | drugs with SMILES, PubChem CID, MoA | doses; single exposure time | **CC0** | Hugging Face (410 MB/s) |
| **Replogle 2022 Perturb-seq** | K562 genome-wide (66 GB single-cell), K562 essential (10.7 GB), RPE1 (8.7 GB); pseudobulk files 0.1–0.4 GB | CRISPRi, ~9,800 genes (GWPS) | single time point | **CC BY 4.0** | figshare (23.4 MB/s) |
| LINCS L1000 (GSE92742, GSE70138) | ~1.3 M profiles (Level 3: 1,319,138 x 12,328), ~20,000 compounds + genetic | compounds, shRNA, ORF | **doses; 6 h and 24 h (some 3/48 h)** | GEO (no licence; cite) | NCBI (~45 MB/s) |
| sci-Plex (Srivatsan 2020, GSE139944) | ~650,000 cells, 3 lines (A549, K562, MCF7), 188 compounds | compounds | 4 doses; 24 h | GEO | NCBI |
| Arc Virtual Cell Challenge 2025 | ~300,000 cells, H1 hESC, 300 CRISPRi genes | CRISPRi | single time point | not stated in the sources found | Arc Virtual Cell Atlas (GCS, not measured) |
| Arc Virtual Cell Atlas (incl. Tahoe) | >300 M cells | various | various | CC0 (Atlas page) | GCS (not measured) |
| ZSCAPE (Saunders et al. 2023) | 3.2 M cells, 1,812 zebrafish embryos | 23 genetic perturbations | **19 time points** | GEO | NCBI |
| CZ CELLxGENE Census | >100 M cells | mostly unperturbed | — | CC BY 4.0 | AWS (not measured) |
| DepMap | ~1,100 lines | CRISPR KO, drugs (viability) | single | CC BY 4.0 | not measured |
| Gene/protein encodings: ESM-2 (MIT), STRING (CC BY 4.0), GO (CC BY 4.0), CORUM | — | — | — | open | not measured |

### 1c. Controlled access (recorded, not used for the primary choice)

| resource | content | access route and time |
|---|---|---|
| UK Biobank imaging | organ MRI, retinal images, genotypes | application + fee; typically weeks to months |
| dbGaP / EGA perturbation or patient cohorts | individual-level sequencing | Data Access Committee request; weeks to months |
| RxRx3 full (Recursion) | ~17,000 gene KOs + compounds, >2 million images | rxrx.ai download under Recursion's licence (click-through; not tested here); RxRx3-core above is the open subset |
| Industrial live-cell screens (Incucyte etc.) | time-lapse drug screens | not public |

**What the inventory says.** Open imaging with per-sample action labels is huge for **static**
readouts (JUMP, PERISCOPE, RxRx3, LINCS Cell Painting: hundreds of thousands of actions) and
small for **time-lapse** readouts — with one exception: **MitoCheck** (191,368 two-day movies of
~21,000 gene knockdowns), plus a handful of smaller live screens (idr0002, idr0061, idr0064,
idr0119). Whole-organism behaviour with genotype labels (OWMD, 386 genotypes) is the only
large non-cell option and sits on Zenodo at ~4 MB/s.

---

## 2. Candidates

| ID | candidate | datasets harvested together | alignment |
|---|---|---|---|
| **A** | **Action-conditioned latent world model of genome-scale live-cell RNAi time-lapse (MitoCheck)**, with zero-shot to unseen genes and planning | MitoCheck screenA (+ screenB independent siRNAs); idr0002, idr0061 (other labs, other live readouts) as transfer sets; **gene-action encodings** harvested from Replogle Perturb-seq (K562/RPE1), JUMP CRISPR+ORF and PERISCOPE (A549/HeLa) profiles, ESM-2, STRING/GO | shared **genes** (HGNC symbols) across all sources; HeLa shared with PERISCOPE |
| **B** | Multi-screen morphology world model (JUMP + RxRx3-core + PERISCOPE) with compound/gene planning | cpg0016, RxRx3-core, cpg0021, cpg0004 | shared compounds (InChIKey) and genes; dose as the only trajectory axis |
| **C** | Cross-modal world model co-embedding imaging screens with Perturb-seq/Tahoe/L1000 | cpg0004 LINCS Cell Painting + L1000 (same compounds/doses), JUMP CRISPR + Replogle, Tahoe-100M + JUMP compounds | shared compounds/genes; cell lines partly shared (A549, MCF7) |
| **D** | Whole-organism behavioural world model across genotypes (and drugs) | OWMD videos (386 genotypes) + O'Brien disease-model and drug features | genes via worm-human orthologs; drugs via SMILES |
| **E** | Developmental world model | ZSCAPE (23 perturbations, 19 time points), Zebrahub (no perturbations), Drosophila blastoderm flows | shared developmental time and cell types |
| **F** | Drug-conditioned live-cell world model with dose/combination planning | idr0064 (429 kinase inhibitors, Erk dynamics), idr0119 (drugs x 8 doses + combinations, FUCCI, 96 h); optionally MitoCheck as pretraining dynamics | compounds by SMILES; cell-cycle readouts shared with MitoCheck |
| **G** | Transcriptomic dose–time world model | Tahoe-100M + sci-Plex + L1000 | shared compounds and lines |

---

## 3. Novelty evidence (five closest prior works per candidate)

Searches: arXiv, bioRxiv, OpenReview/ICLR/NeurIPS/ICML pages, Google-indexed proceedings, and
follow-ups of V-JEPA 2(-AC), AlphaCell, MorphoDiff, IMPA, CellFlux, the Virtual Cell Challenge,
MitoCheck, JUMP-CP and RxRx3, run on 2026-10-19. A combination is called **open** only when no
prior work found does all four of: this data type + action-conditioned latent prediction +
zero-shot to unseen actions + planning.

**A — MitoCheck world model** (verdict: **open**)
1. Holmberg, Mechtel, Ouyang 2025, *Adapting Video Diffusion Models for Time-Lapse Microscopy*
   (arXiv 2503.18583): fine-tunes a video diffusion model on ~3,500 MitoCheck movies
   (S-BIAD865), conditioned on four phenotype scores or the first frame. **Does not** condition
   on gene identity (named as future work), predict in a representation space, test unseen
   genes, or plan.
2. Navidi et al. 2025, *MorphoDiff* (ICLR 2025): diffusion generating Cell Painting images
   conditioned on gene (scGPT) or compound embeddings, including unseen perturbations. **Static
   images**, no dynamics, no planning.
3. Failmezger et al. 2013 and Pau et al. 2013 (BMC Bioinformatics): HMM / dynamical
   phenotype modelling of MitoCheck movies for gene fingerprints. **No** learned latent world
   model, no action conditioning, no unseen genes, no planning.
4. AlphaCell (bioRxiv 2026.03.02.709176) and Chreode (arXiv 2605.28111): cell world models
   with action-conditioned transitions in **transcriptomic** latent space (unseen cell types;
   Chreode via GEARS embeddings). No imaging, no planning reported.
5. Assran et al. 2025, *V-JEPA 2 / V-JEPA 2-AC* (arXiv 2506.09985): the recipe (frozen video
   encoder + action-conditioned predictor + latent MPC planning) on robot video. **No biology.**
   Also relevant: PDGrapher (Gonzalez et al., Nat Biomed Eng 2025) plans perturbations
   (inverse design) **in transcriptomic space only**; Lewis & Zueco (ICLR 2026 workshop)
   co-evolve a world model and a perturbation planner on a **synthetic 5-gene** network.

**B — multi-screen morphology world model** (verdict: **partly occupied**)
1. MorphoDiff (above): forward generation for unseen genes/compounds.
2. Palma et al. 2025, IMPA (Nat Commun 2025): style-transfer prediction of
   perturbation morphology for unseen perturbations.
3. CellFlux (Zhang et al. 2025) and CellFluxRL (arXiv 2603.21743): flow matching from control to
   perturbed morphology on BBBC021/RxRx1/JUMP.
4. Fradkin et al. 2024, MolPhenix (NeurIPS 2024): zero-shot phenomics-to-molecule retrieval
   (77% top-1% accuracy) — a retrieval form of "planning".
5. CellPainTR (arXiv 2509.06986), RxRx3-core benchmark (arXiv 2503.20158), CLOOME, CellCLIP
   (NeurIPS 2025): cross-dataset representations and retrieval.
   Open part: latent action-conditioned prediction **across** JUMP + RxRx3 + PERISCOPE with
   explicit planning; but the data have **no time axis** (fails R4), and forward prediction
   for unseen perturbations is already published.

**C — cross-modal co-embedding** (verdict: **KILL — core combination published**)
1. Conrad et al., PETRI (ICLR 2026): unified embeddings from unpaired optical pooled screen
   and Perturb-seq (569 KOs, HepG2), released matched data.
2. Labeled Gromov–Wasserstein OT (arXiv 2405.00838): cross-modality matching/prediction of
   perturbation responses between imaging and Perturb-seq.
3. MultiVCDiff (Cell Reports Methods 2026): one diffusion framework predicting both morphology
   images and transcriptomes for chemical/genetic perturbations.
4. MorphDiff (transcriptome-guided morphology diffusion, 2025): unseen-perturbation morphology
   from L1000 profiles.
5. Way et al. 2022 (Cell Systems): Cell Painting and L1000 for the same compounds are
   complementary.

**D — worm behavioural world model** (verdict: **open**, but data access is slow)
1. Li, Javer, Keaveny, Brown 2017 (bioRxiv 222208): generative RNN of worm posture; notes that
   no generative model yet captures differences between mutant strains. Not genotype-
   conditioned, no unseen genotypes, no planning.
2. Javer et al. 2018, Open Worm Movement Database: data and features, no world model.
3. O'Brien et al. 2024 (eLife 92491): phenotyping and a drug-repurposing screen; no learned
   predictive model.
4. Tierpsy behavioural fingerprints / phenotype classifiers: discriminative only.
5. V-JEPA 2-AC (above): planning in latent space, not in biology.

**E — developmental world model** (verdict: **open in parts, weak on actions**)
1. Cornelis et al. 2026 (arXiv 2610.16236): zero-shot transfer of a continuum foundation model
   to perturbed Drosophila embryos (not action-conditioned, no planning).
2. Saunders et al. 2023 (Nature, ZSCAPE): perturbation atlas, descriptive.
3. Chreode (arXiv 2605.28111): one-step developmental transitions in scRNA latent space.
4. PRESCIENT / dynamic-OT developmental models: trajectory inference, no actions.
5. AlphaCell (above).

**F — drug-conditioned live-cell world model** (verdict: **open**, small)
1. Gross et al. 2023 (Nat Commun): mechanistic phase-specific ODE model of the same idr0119
   data, predicts unseen drug combinations — **but** mechanistic per-drug parameters, no latent
   video model, no unseen drugs.
2. Goglia et al. 2020 (Cell Systems, idr0064): Erk dynamics screen, descriptive.
3. Holmberg et al. 2025 (above): video generation, no drug conditioning.
4. CellFluxRL / IMPA (above): static.
5. PDGrapher (above): planning in transcriptomics.

**G — transcriptomic dose–time world model** (verdict: **KILL — occupied**)
STATE (Arc 2025; transition model over 100 M perturbed cells incl. Tahoe), AlphaCell (2026),
Chreode (2026), CellFlow-type flow-matching models, PDGrapher (planning). The Virtual Cell
Challenge 2025 is itself a benchmark of this space.

---

## 4. Scores

1–5 per criterion; R2 and R6 count double. Max 55.

| ID | R1 labels | R2 scale x2 | R3 actions | R4 time | R5 backbone | R6 white space x2 | R7 feasible | AMI (AC + planning) | informative | **total** | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** | 5 | 4 (8) | 4 | 5 | 5 | 5 (10) | 4 | 5 | 5 | **51** | **GO — winner** |
| F | 5 | 2 (4) | 4 | 5 | 5 | 4 (8) | 5 | 5 | 4 | 45 | GO — **backup** |
| D | 5 | 3 (6) | 3 | 5 | 5 | 5 (10) | 2 | 4 | 4 | 44 | GO — reserve (Zenodo 4 MB/s) |
| B | 5 | 5 (10) | 5 | 2 | 4 | 3 (6) | 4 | 3 | 4 | 43 | no-go (no time axis; forward prediction published) |
| C | 5 | 4 (8) | 5 | 3 | 3 | 1 (2) | 5 | 3 | 3 | 37 | **KILL** (PETRI, LGW-OT, MultiVCDiff) |
| G | 5 | 5 (10) | 5 | 3 | 3 | 1 (2) | 5 | 3 | 3 | 39 | **KILL** (STATE, AlphaCell, Chreode, PDGrapher) |
| E | 4 | 3 (6) | 2 | 5 | 3 | 4 (8) | 4 | 3 | 3 | 38 | no-go (23 actions; no perturbed video) |

Notes on the scores. A scores 4, not 5, on R2 because its video comes from one lab and one cell
line; its multi-dataset harvest is on the **action side** (gene encodings from four independent
modalities) and in transfer sets (idr0002, idr0061, screenB). A scores 4 on R7 because the full
19 TB cannot be fetched in two weeks at ~31 MB/s; a 1–2 TB subset (15,000–30,000 movies) can be
fetched in one to two days. F is the backup because it reuses A's pipeline and has the strongest
planning ground truth (full dose grids and measured combinations), and its data are 3.2 TB on the
same host; its weakness is the small action count. D has an equally open white space but its
data sit on Zenodo at ~4 MB/s.

## 5. Recommendation

**Go with A: an action-conditioned latent world model of genome-scale live-cell RNAi
time-lapse (MitoCheck) on a frozen V-JEPA 2 encoder, with zero-shot to unseen genes through
gene embeddings harvested from Perturb-seq, JUMP/PERISCOPE and protein language models, and
planning by search over gene actions.** Backup: F (drug-conditioned live-cell world model with
dose and combination planning, idr0064 + idr0119), which shares A's code. Reserve: D.

Every outcome is informative: if the action-conditioned predictor does not beat action-agnostic
and position-only controls, that is a quantitative statement about how much of a knockdown's
two-day dynamics is predictable from latent video (the TiJepa forecasting controls, now applied
from the start); if it does but zero-shot fails, the comparison of gene-encoding sources says
which modality carries transferable information about a gene's cell-division phenotype; if
planning works, it is the first demonstration of latent planning over genetic actions on
cell video. Designs: `reports/NEXT_PROJECT_DESIGN.md`.
