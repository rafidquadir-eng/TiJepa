# TiJepa

A self-supervised video model of innate immune cell behaviour (a video joint-embedding
predictive architecture, JEPA), and a public-data audit of how much molecular, functional
and memory state can be read from label-free images and video of innate immune cells.

**What this repository does not show:** trained immunity in images. No public image or
video of trained or tolerised cells exists; the memory-related analyses use the closest
public proxies (acute priming and tolerance, polarisation, restimulation transcriptomes).

## What is here

| | |
|---|---|
| Model | TiJepa-1: ViT-S video JEPA, 16 x 96 x 96 single-channel clips, mixed spatial and future masking, 117,000 steps on 279,647 clips of migrating human neutrophils (ComplexEye, CC BY 4.0); a matched-step spatial-only ablation; TiJepa-0 (earlier corpus) as a historical reference |
| Baselines | ImageNet ViT-S/16, DINOv2 ViT-S/14, V-JEPA 2 ViT-L (frozen, MIT licence), random initialisation (3 draws), frame-shuffled input, hand-crafted features |
| Benchmarks | microglia morphology (BBBC054), label-free immune cell identity (two-photon autofluorescence), NAD(P)H lifetime from the photon image (FLIM), cell re-identification (validation only), detection-level association, latent forecasting with controls, NF-κB ligand decoding, restimulation transcriptomes |
| Manuscripts | `paper/v3/tmlr/` (anonymised, TMLR style) and `paper/v3/biorxiv/` (named, biology-first); figures in `figures/v3/` |
| Report | `reports/RUN_REPORT_v3.md`: compute, decision outcomes, modality coverage map, findings table |

## Headline results

<!-- RESULTS:BEGIN -->
| Test | TiJepa-1 | comparison | reference |
|---|---|---|---|
| Microglia morphology, 3-class BA | 0.595 | V-JEPA 2 0.681 | hand-crafted 0.499 |
| T cell vs neutrophil (mixed fields), BA | 0.714 | TiJepa-0 0.721 | intensity only 0.714 |
| NAD(P)H lifetime within fields, R2 | +0.08 | V-JEPA 2 +0.04 | intensity only +0.08 |
| Re-identification (validation), top-1 | 0.501 | V-JEPA 2 0.667 | raw pixels 0.555 |
| Association, 9-frame gap, embedding only | 0.404 | V-JEPA 2 0.751 | position only 0.883 |
| Forecasting advantage kept on static clips | 0.79 of 0.83 | — | criterion: at most half |
| NF-κB ligand pair, BA (trace JEPA) | 0.286 | random init 0.284 | chance 0.091 |
| β-glucan − untrained TNF after restimulation (log2) | — | GSE212282 -2.09, GSE141656 -1.96 | 6/6 donors |
<!-- RESULTS:END -->

## Reproduce

```bash
python -m venv ~/venv && ~/venv/bin/pip install -r requirements-lock.txt
scripts/run_all_v3.sh            # every v3 result, figure and manuscript
RETRAIN=1 scripts/run_all_v3.sh  # also retrain (about 21 GPU-hours plus the recipe sweep)
```

The frozen corpus (`data/processed/v1`) and the checkpoints are not rebuilt; their
checksums are in `results/v2/corpus_shards_sha256.json` and `reports/BACKUP.md`. Raw data
are downloaded by `scripts/1x_*`; every dataset, licence and checksum is listed in
`manifests/datasets.yaml`, model weights and licences in `manifests/models.yaml`. Scripts
from earlier iterations are in `legacy/` with a map to their replacements.

## Data licences

ComplexEye, S-BIAD2196, the autofluorescence (Zenodo 15657646) and FLIM (Zenodo 14330511)
records and the Tay-lab traces are CC BY 4.0; BBBC054 is CC BY 3.0; GEO records carry no
licence and their source publications are cited. V-JEPA 2 weights are MIT; DINOv2 and the
AugReg ViT are Apache-2.0. Trained TiJepa weights are not published.

## Limitations

One pretraining seed per recipe; a single-source pretraining corpus of one cell type; small
groups in several analyses (six tracking movies, four tolerance wells, three donors per
restimulation dataset); a partially collapsed representation. See the Limitations section
of either manuscript.

## Proposed experiment

The imaging study that would test innate immune memory directly (primary human monocytes,
training/tolerance/polarisation arms, rest-period time-lapse and NAD(P)H imaging, cytokines
at two times after restimulation) is described in the Future-work section of the manuscripts.
