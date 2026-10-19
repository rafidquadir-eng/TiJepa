#!/usr/bin/env bash
# Reproduce the TiJepa-0 run end to end. Stages are independent and resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="python3 -I"

echo "== P0 preflight";        $PY scripts/00_preflight.py
echo "== M1 downloads";        $PY scripts/10_download.py --group small
                               $PY scripts/10_download.py --group complexeye --cap-gb 60
                               $PY scripts/11_download_labelled.py --only all
                               $PY scripts/12_download_geo.py
echo "== M1 manifests";        $PY scripts/31_build_manifest.py
echo "== M2 Part B";           $PY scripts/20_partB.py
                               $PY scripts/21_partB_omics.py
echo "== M3 Part C omics";     $PY scripts/40_partC_omics.py
echo "== M4 preprocessing";    $PY scripts/30_preprocess_pretrain.py --workers "$(nproc)"
echo "== M5 pretraining";      $PY scripts/50_pretrain.py --seed 0 --max-hours 8
echo "== M6 evaluation";       $PY scripts/60_partA_eval.py
                               $PY scripts/61_tracking_eval.py
                               $PY scripts/41_partC_projection.py || true
echo "== M7 figures + paper";  $PY scripts/80_figures.py
                               $PY scripts/90_make_paper_numbers.py
echo "done"
