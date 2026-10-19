#!/usr/bin/env bash
# Build both v3 manuscripts with latexmk; fail on undefined references or citations.
set -euo pipefail
cd "$(dirname "$0")/../paper/v3"
for d in tmlr biorxiv; do
  ( cd "$d"
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || {
      tail -40 build.log; echo "build of $d failed"; exit 1; }
    if grep -E "Citation .* undefined|Reference .* undefined|\?\?[a-z]" main.log build.log >/dev/null; then
      grep -E "Citation .* undefined|Reference .* undefined" main.log | head
      echo "$d: undefined citations or references"; exit 1
    fi
    echo "$d: $(pdfinfo main.pdf | awk '/Pages/ {print $2}') pages" )
done
cp biorxiv/main.pdf biorxiv/tijepa_preprint.pdf
cp tmlr/main.pdf tmlr/tijepa_tmlr.pdf
