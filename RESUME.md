# Resuming TiJepa v1 pretraining on a new instance

Written 2026-10-16 04:30 UTC, when pretraining was paused on the shared A100 instance so
the project could move to a dedicated GPU. Nothing is lost: the run resumes from the
checkpoint below with optimizer state and step number intact.

## Where the run stopped

| | |
|---|---|
| Run | `tijepa1` (mixed masking, `p_future = 0.5`, 2 future slots), seed 0 |
| Checkpoint step | **14751** |
| Last logged step | 14800 (the 50 steps after the save were discarded by the stop) |
| GPU-hours consumed | **1.51 h** of the 12 GPU-hour budget |
| Clips seen | 1,894,400 (batch 128) |
| Loss | 0.3974 total / 0.3934 spatial / 0.3996 future at step 14800 |
| Effective rank | ~37 of 384 and still climbing — no collapse |
| Stop method | SIGTERM immediately after the periodic save; the trainer has no signal handler, so the save was waited for deliberately |

The matched-step spatial-only ablation (`tijepa1_spatial`, `p_future = 0.0`) had **not**
started. It must run for the same number of steps the main run finishes at.

## Checkpoint

```
checkpoints/tijepa1_T1_seed0.pt          376,538,701 bytes
sha256  2774e60461d2c24b9b8c92b71d211c1cfa4b4e6241010d1c61888f6b7d054b22
```

Contains `model`, `optim` (229 parameter states), `step`, `config`.
`src/tijepa/train/pretrain_video.py` resumes from this path automatically **if the file is
present under `checkpoints/`** — there is no `--resume` flag. Keep the filename exactly as
it is, or training restarts from step 0.

Also keep `checkpoints/pretrain_log_tijepa1_seed0.csv`; the trainer appends to it and the
collapse diagnostics in the paper are read from it.

## Corpus — do not rebuild it

| | |
|---|---|
| Path | `data/processed/v1/` |
| Build ID (`pretrain_index.csv` md5) | **43b19b36849e88d29cefa484066e35a1** |
| Shards | 43 `.npy` files + `pretrain_index.csv` (279,648 lines = header + 279,647 clips) |
| Clips | **279,647** |
| Source movies | **407** ok, 1 excluded for leakage, 18 excluded by QC, 0 failed |
| Size | 41.24 GB nominal / 39 GB on disk |
| Archives | 10 of 21 ComplexEye archives (stopped on wall-clock, ~56 min each) |
| Strides | 1: 211,661 · 3: 61,078 · 9: 6,173 · 27: 735 |
| Static clips | 28,450 (10.17%) |
| Target scale | 0.65 µm/px |
| Generated | 2026-10-16T02:55:40Z (`results/v1/pretrain_corpus.json`) |

Re-running `scripts/32_corpus_v1.py` on the new instance would produce a **different**
corpus — the archive set is wall-clock-bounded, so a different number of archives would
land — and every corpus figure in `paper/tijepa.md` and `reports/RUN_REPORT_v1.md` is
pinned to the build ID above. Verify the md5 after the transfer instead of regenerating.

## Resume command

```bash
cd ~/proj2
/path/to/python3 -u scripts/51_pretrain_v1.py \
    --which both --hours 10.5 --max-steps 150000 \
    --batch-size 128 --num-workers 14 \
    > logs/v1_pretrain.log 2>&1 &
```

**`--hours` is wall-clock from process start, not cumulative**, so it must be reduced by
the time already spent or the run will overshoot the 12 GPU-hour budget the paper reports:
12 − 1.51 = **10.5**. Set `--num-workers` to the new instance's core count minus a couple
(14 suited 30 cores).

`--which both` runs the main objective to its budget and then the spatial-only ablation at
the matched step count. The shard list is frozen inside the trainer, so both runs see the
same corpus.

## After pretraining finishes

In order, from `scripts/`:

1. `62_bbbc054_v1.py --n-cells 9000 --scale-sensitivity` — morphology-class probes
2. `63_tracking_forecast_v1.py` — tracking and latent forecasting
3. `81_figures_v1.py`, `91_make_paper_v1.py`, `92_check_claims.py`, `94_run_report_v1.py` —
   regenerate figures, paper, claim verification and the Section 12 table
4. `99_backup_v1.sh` — back up v1 outputs to Drive, then append to `reports/BACKUP.md`
5. `gh pr edit 1` — title to "TiJepa v1: corrected computational preprint", body from
   `reports/PR_BODY_v1.md`

Standing constraints: never force-push, never push to `main`, never rewrite history;
continue on branch `tijepa-v0` (PR #1 updates itself). Working tree was clean and fully
pushed at the time of the move.

## Git state at the move

```
branch tijepa-v0, clean, nothing unpushed
6945711 chore(v1): draft the PR body
```
