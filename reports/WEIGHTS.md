# Model weights

The TiJepa-0 checkpoint is **not committed to git** (PLAN.md §1.3.2 and the repository
`.gitignore` exclude `checkpoints/` and `*.pt`).

| | |
|---|---|
| File | `checkpoints/tijepa0_T1_seed0.pt` |
| Contents | encoder + EMA target encoder + predictor `state_dict`, optimiser state, step count, resolved config |
| Architecture | ViT-S over 2x8x8 tubelets of 16x96x96 clips (1,152 tokens), ~21.3 M encoder parameters |
| Tier | T1 (1x A100-SXM4-40GB, bf16) |
| Seed | 0 (single-GPU run; see `results/preflight.json` for the seed policy) |
| Licence | Apache-2.0, same as the code |
| Training data | ComplexEye human neutrophil movies, Zenodo 10082674, CC BY 4.0 only |

Exact size, step count and GPU-hours are in `results/partA/pretrain_seed0.json`.

## What the author needs to do

The plan asks for weights to be attached to a draft GitHub Release `tijepa-0-v0.1` if
they are **under 2 GB**, otherwise uploaded to Zenodo or Hugging Face.

**This run did not publish them.** PLAN.md §1.4.3 forbids posting anything to Zenodo or
Hugging Face, and creating a GitHub Release publishes an artefact outward, which was not
authorised for this run. The checkpoint is on the instance's local disk only, and that
disk does not persist.

To publish them yourself, from a machine holding the checkpoint:

```bash
ls -lh checkpoints/tijepa0_T1_seed0.pt          # confirm the size

# if under 2 GB — draft release, nothing is public until you publish it
gh release create tijepa-0-v0.1 --draft \
  --title "TiJepa-0 v0.1" \
  --notes "Video-JEPA ViT-S pretrained on ComplexEye neutrophil movies (CC BY 4.0). Apache-2.0." \
  checkpoints/tijepa0_T1_seed0.pt

# if 2 GB or larger — upload to Zenodo or Hugging Face instead and record the DOI here
```

Then record the resulting URL or DOI in this file and in `paper/tijepa.md` §5.

**Reproducing the weights instead of downloading them:** `scripts/run_all.sh` rebuilds
the corpus and retrains from scratch; the exact configuration is stored inside the
checkpoint and mirrored in `results/partA/pretrain_seed0.json`.
