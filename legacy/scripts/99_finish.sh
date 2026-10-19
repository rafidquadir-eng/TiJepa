#!/usr/bin/env bash
# Run every post-training stage once pretraining has finished.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="python3 -I"
echo "== M6 Part A probes";      $PY scripts/60_partA_eval.py --n-cells 9000
echo "== M6 tracking";           $PY scripts/61_tracking_eval.py
echo "== M6 Part C projection";  $PY scripts/41_partC_projection.py
echo "== M7 figures";            $PY scripts/80_figures.py
echo "== M7 paper numbers";      $PY scripts/90_make_paper_numbers.py
echo "== M7 run report";         $PY scripts/91_run_report.py
echo FINISH_DONE
