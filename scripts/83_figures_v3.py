"""Render figures/v3/ from results (src/tijepa/viz/figures_v3.py); a figure whose inputs
do not exist is skipped and reported. Writes results/v3/figures.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.viz import figures_v3 as F  # noqa: E402


def main() -> int:
    out = {}
    for fn in F.ALL:
        paths = fn(REPO)
        out[fn.__name__] = paths or "skipped: inputs absent"
        print(f"  {fn.__name__}: {paths or 'skipped'}")
    (REPO / "results" / "v3" / "figures.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
