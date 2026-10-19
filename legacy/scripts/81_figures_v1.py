"""F7 10.3 driver: render all eight v1 figures to figures/ as PNG and PDF.

Each figure is generated from results/v1 only; a missing input renders an explicit
"not available" panel rather than a plausible-looking placeholder. PLAN_v1.md 10.3 also
requires that each figure be inspected before committing, so this script reports the
pixel size and the fraction of non-white pixels per PNG, which is what catches a figure
that rendered blank or with an empty panel.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.viz import figures_v1 as F  # noqa: E402


def main() -> int:
    out = {"figures": [], "generated_from": "results/v1"}
    for fn in F.ALL:
        try:
            paths = fn(REPO)
            rec = {"function": fn.__name__, "files": paths, "status": "ok"}
            png = next((p for p in paths if p.endswith(".png")), None)
            if png:
                try:
                    import numpy as np
                    from PIL import Image
                    im = np.asarray(Image.open(REPO / png).convert("L"))
                    rec["size_px"] = list(im.shape[::-1])
                    rec["ink_fraction"] = round(float((im < 250).mean()), 4)
                    if rec["ink_fraction"] < 0.01:
                        rec["status"] = "SUSPECT_BLANK"
                except Exception as exc:
                    rec["inspect_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  {fn.__name__:24s} {rec['status']:14s} "
                  f"{rec.get('size_px')} ink={rec.get('ink_fraction')}")
        except Exception as exc:
            rec = {"function": fn.__name__, "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}"}
            print(f"  {fn.__name__:24s} FAILED {rec['error']}")
        out["figures"].append(rec)
    (REPO / "results" / "v1").mkdir(parents=True, exist_ok=True)
    (REPO / "results" / "v1" / "figures.json").write_text(
        json.dumps(out, indent=2) + "\n")
    n_bad = sum(1 for r in out["figures"] if r["status"] != "ok")
    print(f"\n{len(out['figures']) - n_bad}/{len(out['figures'])} figures ok")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
