# TiJepa

Self-supervised imaging / video modelling of **trained immunity** and innate immune
activation states (monocytes, macrophages, neutrophils, NK cells, dendritic cells),
linked to **iMEMORY**, a multi-omic model of immune disease.

Target states: naive · trained (β-glucan, BCG, oxLDL) · tolerant (LPS pre-exposure) ·
classically activated / polarized (LPS, IFN-γ, IL-4, M1/M2).

---

## The finding this repository is built on

**No public dataset contains imaging or video of trained, tolerant, or innate-memory
immune cells.** A systematic survey of the public record is in
[`reports/Trained immunity imaging datasets.md`](reports/Trained%20immunity%20imaging%20datasets.md):
a Europe PMC full-text search for `"trained immunity"` combined with any image-repository
term returns **0 hits**; the BioImage Archive's 59 macrophage/monocyte/neutrophil studies
include **no** β-glucan, BCG, oxLDL, MDP or LPS-tolerance study; and all 147 IDR projects
and 105 screens contain no innate-immune training study and no innate-immune time-lapse
at all.

So the imaging-plus-omics dataset this project needs has to be generated. TiJepa-0 builds
and honestly evaluates the components such a study would need, using data that exists.

## TiJepa-0 (v0, computational-only)

| Part | Question | Pairing |
|---|---|---|
| **A. Video SSL** | Does JEPA-style video pretraining on open innate-immune microscopy capture activation state and support tracking? | real labels |
| **B. Dynamics → chromatin** | Can single-cell NF-κB dynamics predict features of LPS tolerance? | real, same lab |
| **C. Bridge** | Where do trained and tolerant states fall on shared activation axes? | **hypothesis only** |

Read the paper draft in [`paper/tijepa.md`](paper/tijepa.md) and the honest account of
what ran, what failed and every deviation in
[`reports/RUN_REPORT.md`](reports/RUN_REPORT.md).

### Headline outcomes

- **TiJepa-0** pretrained on ~288k cell-centred clips (~42 GB) from 288 openly licensed
  (CC BY 4.0) human neutrophil migration movies.
- **Part A evaluation is static.** The planned headline benchmark (Dryad
  `doi:10.5061/dryad.m0cfxpp8v`, M0/M1/M2 macrophage movies) sits behind a proof-of-work
  access control and could not be fetched, so no dataset in this run provides *video* of
  innate immune cells with activation-state labels — the data gap restated as an
  experimental limitation. See [`reports/MANUAL_DOWNLOADS.md`](reports/MANUAL_DOWNLOADS.md).
- **Part B is a reported null.** NF-κB dynamics separate naive→LPS from LPS→LPS above
  chance, but the contrast rests on 4 wells and is not significant against a
  well-level permutation null. The dynamics→transcriptome and cross-species analyses are
  null too.
- **Part C is hypothesis-generating only** and pairs no image with any molecular sample.

Nothing in this repository claims that trained immunity can be identified from images.
No trained or tolerant cell was imaged in any dataset used here.

## Reproducing

```bash
pip install -e ".[dev]"
bash scripts/run_all.sh        # preflight -> downloads -> Parts B/C -> pretrain -> eval -> paper
pytest -q && ruff check src/ scripts/ tests/
```

Stages are independent and resumable; pretraining checkpoints every ~30 minutes and
resumes in place. Every number in `paper/` is written by
`scripts/90_make_paper_numbers.py` from `results/*.json` — none is typed by hand, and a
missing result renders as an explicit "not available" rather than a placeholder number.

## Layout

| Path | Contents |
|---|---|
| `src/tijepa/` | package: `io/`, `preprocess/`, `models/`, `train/`, `eval/`, `omics/`, `viz/` |
| `scripts/` | one script per stage, `run_all.sh` |
| `manifests/` | `datasets.yaml`, `models.yaml` — licences read from each record, with sizes and checksums |
| `results/` | every metric as JSON/CSV, produced by code |
| `figures/` | all figures (PNG + PDF), produced by code |
| `paper/` | manuscript template and generated draft |
| `reports/` | run report, manual downloads, weights, and the landscape review |
| `research_notes/` | raw notes behind the landscape review |

## Data licences

Pretraining used **only CC BY 4.0** data. Every dataset's licence was read from its own
record; see `manifests/datasets.yaml`. Datasets excluded on licence grounds
(Zenodo `16319732` CC BY-NC-ND, DeepCell DynamicNuclearNet, the Cellpose training set,
CTMC-v1, LIVECell, Cell Tracking Challenge) were never downloaded.

## Limitations

The pretraining corpus is single-source (one cell type, one modality, one lab); no
activation-labelled video was evaluated; Part B is underpowered and null; Part C is a
cross-species, cross-platform interpolation between three anchor conditions; no QC
filtering was applied to pretraining clips; pretraining used a single seed. The full
list is in `reports/RUN_REPORT.md` §5.

## Licence

Apache-2.0 for code ([`LICENSE`](LICENSE)); CC BY 4.0 for text and figures
([`LICENSE-docs`](LICENSE-docs)). Third-party data keep their own licences.
