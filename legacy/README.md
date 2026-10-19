# legacy/

Scripts from earlier iterations of this repository that the current pipeline
(`scripts/run_all_v3.sh`) no longer calls. They are kept, unmodified, because results
under `results/v0`–`results/v2` and the reports that cite them were produced by these
scripts; nothing in `results/v3/`, `figures/v3/` or `paper/v3/` depends on them. Paths
inside them still refer to `scripts/`, so run them from a checkout of the commit that
produced the result you want to reproduce.

| legacy script | what it did | replaced by (current pipeline) |
|---|---|---|
| `00_preflight.py` | v0 environment and host checks | `scripts/02_preflight_v2.py` |
| `14_leakage_check.py` | v0 fingerprint leakage check (subset) | `scripts/15_leakage_check_full.py` (all archives); `scripts/71_validation_set_v3.py` (validation set) |
| `20_partB.py`, `21_partB_omics.py` | v0 Part B (tolerance, omics) | `scripts/25_partB_v1.py`, `26_partB_v3.py`, `27_partB_b33_b4_v2.py`, `28_partB_b4_v3.py` |
| `30_preprocess_pretrain.py` | v0 corpus (7 archives, no QC) | `scripts/32_corpus_v1.py` (frozen corpus) |
| `40_partC_omics.py`, `42_partC_v1.py` | v0/v1 Part C resting-state scores | `scripts/45_partC_restim_v2.py`, `46_partC_sample_qc_v3.py`, `47_partC_replication_v3.py` |
| `44_partC_projection_v1.py` | image-to-omics projection (never performed: no stimulus-matched data) | none; stated in the paper |
| `50_pretrain.py`, `51_pretrain_v1.py` | v0/v1 training drivers | `scripts/52_pretrain_v2.py` (TiJepa-1, ablation), `scripts/74_recipe_sweep_v3.py` |
| `53_pretrain_monitor_v2.py` | in-training monitor on a BBBC054 subset | removed: new runs are scored only on the validation set (`72_reid_collapse_v3.py`) |
| `60_partA_eval.py`, `61_tracking_eval.py` | v0 Part A evaluation | `scripts/62_bbbc054_v1.py` (writes v3 with `--out-dir`), `63_tracking_forecast_v1.py`, `70`, `72`, `76` |
| `64_dryad_v1.py` | Dryad file detection | intake recorded in `results/v3/manual_intake.json` |
| `80_figures.py`, `81_figures_v1.py`, `82_figures_v2.py` | earlier figures | `scripts/83_figures_v3.py` (`src/tijepa/viz/figures_v3.py`) |
| `90_make_paper_numbers.py`, `91_make_paper_v1.py`, `95_make_paper_v2.py` | earlier manuscripts | `scripts/97_paper_numbers_v3.py` + `paper/v3/` |
| `92_check_claims.py` | v2 claims check | `scripts/98_check_claims_v3.py` |
| `91_run_report.py`, `94_run_report_v1.py`, `96_run_report_v2.py` | earlier run reports | `scripts/99_run_report_v3.py` |
| `98_backup_loop_v2.sh`, `99_backup_v0.sh`, `99_backup_v1.sh`, `99_finish.sh` | earlier backup/finish helpers | backups recorded in `reports/BACKUP.md` |
| `run_all.sh`, `run_all_v1.sh` | earlier end-to-end runs | `scripts/run_all_v3.sh` |

Script names keep their numeric stage prefix: 0x setup, 1x data, 2x Part B, 3x corpus,
4x Part C, 5x training, 6x–7x Part A evaluation and coverage map, 8x figures, 9x paper,
references and report.
