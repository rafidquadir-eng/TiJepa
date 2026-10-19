#!/usr/bin/env bash
# Reproduce every v3 result, figure and manuscript from raw data, the frozen corpus and
# the checkpoints. Stages run in dependency order; each writes under results/v3/,
# figures/v3/ or paper/v3/. GPU stages run one at a time.
#
#   scripts/run_all_v3.sh            # everything except retraining
#   RETRAIN=1 scripts/run_all_v3.sh  # also retrain TiJepa-1, the ablation and the sweep
#
# Inputs that are NOT rebuilt: data/processed/v1 (the frozen corpus; checksums in
# results/v2/corpus_shards_sha256.json) and checkpoints/*.pt (hashes in reports/BACKUP.md).
# Raw downloads: scripts/10_download.py, 11_download_labelled.py, 12_download_geo.py and
# 13_fetch_complexeye.sh; the v3 additions (two Zenodo records, GSE141656, GSE242947) are
# listed with checksums in manifests/datasets.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-~/venv/bin/python}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "preflight and integrity"
$PY scripts/02_preflight_v2.py

if [ "${RETRAIN:-0}" = "1" ]; then
  log "pretraining TiJepa-1 and the spatial-only ablation (about 21 GPU-hours)"
  $PY scripts/52_pretrain_v2.py --which both
  log "recipe sweep, 5 x 20,000 steps"
  $PY scripts/74_recipe_sweep_v3.py --train
fi

log "Part A"
$PY scripts/15_leakage_check_full.py
$PY scripts/71_validation_set_v3.py
$PY scripts/62_bbbc054_v1.py --n-cells 9000 --scale-sensitivity --out-dir results/v3/partA
$PY scripts/70_forecast_controls_v3.py
$PY scripts/72_reid_collapse_v3.py
$PY scripts/76_association_v3.py
$PY scripts/63_tracking_forecast_v1.py          # point tracking (supplementary, v2 file)
$PY scripts/74_recipe_sweep_v3.py --score

log "coverage-map datasets"
$PY scripts/73_autofluorescence_v3.py
$PY scripts/75_flim_v3.py --extract --analyse

log "Part B"
$PY scripts/25_partB_v1.py
$PY scripts/27_partB_b33_b4_v2.py
$PY scripts/26_partB_v3.py
$PY scripts/28_partB_b4_v3.py

log "Part C"
$PY scripts/45_partC_restim_v2.py
$PY scripts/46_partC_sample_qc_v3.py
$PY scripts/47_partC_replication_v3.py

log "coverage map, figures, references, manuscripts"
$PY scripts/79_coverage_map_v3.py
$PY scripts/83_figures_v3.py
$PY scripts/93_resolve_refs.py
$PY scripts/96_bib_v3.py
$PY scripts/97_paper_numbers_v3.py
bash scripts/84_build_papers_v3.sh
$PY scripts/98_check_claims_v3.py

log "report"
$PY scripts/99_run_report_v3.py
log "done"
