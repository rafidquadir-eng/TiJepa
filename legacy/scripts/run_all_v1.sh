#!/usr/bin/env bash
# Reproduce the TiJepa **v1** run end to end (PLAN_v1.md F0-F9, Section 11.3).
#
# This is what actually ran, with the same flags, caps and budgets. There is no
# `|| true` anywhere: an optional stage checks its own inputs and writes an explicit
# {"status": "skipped", "reason": ...} JSON instead of being allowed to fail quietly.
#
# Two stages are not reproducible from a clean checkout by design and say so:
#   * F0.1/F9 Drive backup needs the author's rclone remote;
#   * F0.3 intake needs the author's manual downloads, which were NOT provided in this
#     run, so the Dryad-dependent analyses report "not provided" rather than numbers.
#
# Pretraining is the long pole: 12 GPU-hours for TiJepa-1, then the spatial-only
# ablation for the SAME number of steps. Everything before it is CPU work and runs
# while nothing else needs the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="/home/ubuntu/vcc/.venv/bin/python3 -u"
NPROC="$(nproc)"
WORKERS=$(( NPROC > 16 ? 14 : NPROC / 2 ))

echo "===== F0  backup, preflight, intake"
bash scripts/99_backup_v0.sh || echo "  (Drive backup needs the author's rclone remote)"
$PY scripts/00_preflight.py

echo "===== F1.1-F1.2  Part B data fixes and the guard"
$PY scripts/12_download_geo.py
$PY scripts/22_partB_rna.py
$PY scripts/23_partB_ligand_codes.py
$PY scripts/24_partB_atac.py

echo "===== F1.3  leakage check (must precede corpus building)"
$PY scripts/11_download_labelled.py --only s2196
$PY scripts/14_leakage_check.py --workers "$WORKERS"

echo "===== F1.4  pretraining corpus, one archive at a time"
# 10 of the 21 ComplexEye archives were processed before the wall-clock cap; see
# results/v1/pretrain_corpus.json and the deviations table in RUN_REPORT_v1.md.
$PY scripts/32_corpus_v1.py --workers "$WORKERS" \
    --max-clips-per-movie 700 --max-clips-total 630000

echo "===== F4  Part B v1"
$PY scripts/25_partB_v1.py
$PY scripts/26_partB_dynamics_omics.py

echo "===== F5  Part C v1 (8.1-8.3)"
$PY scripts/42_partC_v1.py
$PY scripts/43_partC_restim.py

echo "===== F2  pretraining: TiJepa-1 then the matched-step spatial-only ablation"
$PY scripts/51_pretrain_v1.py --which both --hours 12 --max-steps 150000 \
    --batch-size 128 --num-workers "$WORKERS"

echo "===== F3  Part A evaluation"
$PY scripts/62_bbbc054_v1.py --n-cells 9000 --scale-sensitivity
$PY scripts/63_tracking_forecast_v1.py
# F3.1 Dryad and F5 8.4 C3 need the author's manual download; each writes an explicit
# "not provided" status rather than being skipped silently.
$PY scripts/64_dryad_v1.py
$PY scripts/44_partC_projection_v1.py

echo "===== F7  references, figures, paper, claims"
$PY scripts/93_resolve_refs.py
$PY scripts/81_figures_v1.py
$PY scripts/91_make_paper_v1.py
$PY scripts/92_check_claims.py

echo "===== F8  tests and lint"
$PY -m pytest tests/ -q
/home/ubuntu/vcc/.venv/bin/python3 -m ruff check src/ scripts/ tests/

echo "===== F9  report and backup"
$PY scripts/94_run_report_v1.py
bash scripts/99_backup_v1.sh || echo "  (Drive backup needs the author's rclone remote)"

echo "done"
