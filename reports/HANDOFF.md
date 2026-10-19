# Handoff (2026-10-19)

State of the repository, the data and the open decisions, written at the end of the TiJepa
project and Phase 0 of the next one. Nothing has been submitted, posted, released or
force-pushed; nothing was pushed to `main`.

## Branches and pull requests

| branch | what it holds | PR |
|---|---|---|
| `main` | original base | — |
| `tijepa-v0` | TiJepa v0–v2 (computational preprint v0 → v2) | #1 open, title still "TiJepa-0: computational preprint v0" (author: squash-merge or retitle) |
| `tijepa-v3` | TiJepa v3: both manuscripts, figures, all v3 results, run report, wipe manifest | #2 open into `main` |
| `next-project` | Phase 0 of the next project: selection report and designs (this file) | none |

All branches are clean and pushed (checked 2026-10-19).

## TiJepa: findings (details in `reports/RUN_REPORT_v3.md` on `tijepa-v3`)

- A video JEPA trained on 279,647 neutrophil clips (TiJepa-1, 117,000 steps) classifies
  microglia morphology far better than hand-crafted features and as well as image models, but
  a frozen general-purpose video model (V-JEPA 2) is better (0.681 vs 0.595 balanced accuracy).
- Its apparent forecasting is positional structure in the latent space (it keeps 0.79 of its
  0.83 advantage over copy-last on static clips); no forecasting claim is made.
- Cell position alone associates cells across frames better than any embedding; label-free
  cell identity and NAD(P)H lifetime are predicted no better than intensity statistics once
  experimental condition is controlled.
- The representation is partially collapsed (effective rank 7.7 of 384); a 5-variant,
  20,000-step recipe sweep found no fix (G-Recipe failed), but the sweep ends before the
  collapse sets in.
- No public dataset pairs imaging of innate immune cells with a memory readout (modality
  coverage map).
- Replicated observation: beta-glucan-trained human macrophages have ~2 log2 lower TNF mRNA 4 h
  after restimulation in two independent datasets (6/6 donors), while the source papers report
  higher TNF-a protein at 24 h.
- Deliverables: `paper/v3/tmlr/tijepa_tmlr.pdf` (anonymised), `paper/v3/biorxiv/tijepa_preprint.pdf`
  (named), `figures/v3/`, `reports/RUN_REPORT_v3.md`.

## Data and backups

- The instance's local data, checkpoints, logs and caches were deleted after SHA-256
  verification of every file against the Lambda persistent filesystem
  (`LambdaFilesystem/transfer/proj2/`, 107 GB; `reports/WIPE_MANIFEST.md` and
  `reports/wipe_evidence/` on `tijepa-v3`). Free local disk: ~480 GB.
- Checkpoints (TiJepa-0, TiJepa-1, the spatial-only ablation, 20 sweep snapshots and 5 resume
  checkpoints, the trace JEPA) are on the persistent filesystem under `tijepa_backups/`,
  `checkpoints/`, `checkpoints_v2/`.
- One untracked local file remains in the checkout, `results/v0/partA/bbbc054_embeddings.npz`
  (40 MB, intentionally not committed because tracked files are kept under 5 MB); an identical
  copy (same SHA-256) is on the persistent filesystem.
- **Google Drive copies are still pending:** the `gdrive` rclone remote on this instance has no
  OAuth token.

## Open author actions

1. Re-authorise rclone (`rclone config reconnect gdrive:`) and copy `tijepa_backups/` to Drive.
2. Add your affiliation (and ORCID) to the bioRxiv manuscript; decide on code availability and an
   anonymised code supplement for TMLR.
3. Read both manuscripts; titles and framing are yours to adjust.
4. Squash-merge or close PR #1; review PR #2.
5. Dryad datasets (`m0cfxpp8v`, `8gtht76q2`) are still not obtained (download needs a browser).

## Next project (Phase 0 done; awaiting your go)

- Recommendation: **GeneWorld** — an action-conditioned latent world model of genome-scale
  live-cell RNAi time-lapse (MitoCheck, 191,368 movies, ~21,000 genes, CC0) on a frozen
  V-JEPA 2 encoder, with zero-shot to unseen genes via gene encodings from Perturb-seq,
  JUMP/PERISCOPE, ESM-2 and STRING, and planning by search over gene actions.
  Backup: **DrugWorld** (idr0064 + idr0119, compound/dose/combination planning).
- Main risk: EBI download throughput (measured 31 MB/s with 64 streams); plan starts with a
  ~1.1 TB subset; Aspera/Globus untested.
- Documents: `reports/NEXT_PROJECT_SELECTION.md`, `reports/NEXT_PROJECT_DESIGN.md`.
- No GPU job is running; the instance is idle.
