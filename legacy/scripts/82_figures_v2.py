"""G6: render the v2 figures from results/v2 and record what was drawn.

Each figure is checked for being non-blank; every PNG must still be opened and looked at
before it is committed (PLAN_v2.md §9). A function whose inputs are absent returns no
files and is recorded as omitted, never drawn as a placeholder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from tijepa.viz import figures_v1 as F1  # noqa: E402
from tijepa.viz import figures_v2 as F  # noqa: E402


def main() -> int:
    out = {"figures": [], "generated_from": "results/v2 (Fig 1 and the landscape figure "
                                            "are schematics with no result inputs)"}
    for fn in [F1.fig1_overview, *F.ALL, F1.fig8_landscape]:
        try:
            paths = fn(REPO)
            rec = {"function": fn.__name__, "files": paths,
                   "status": "ok" if paths else "omitted: inputs absent"}
            png = next((p for p in paths if p.endswith(".png")), None)
            if png:
                im = np.asarray(Image.open(REPO / png).convert("L"))
                rec["size_px"] = list(im.shape[::-1])
                rec["ink_fraction"] = round(float((im < 250).mean()), 4)
                if rec["ink_fraction"] < 0.01:
                    rec["status"] = "SUSPECT_BLANK"
        except Exception as exc:
            rec = {"function": fn.__name__, "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}"}
        print(f"  {rec['function']:24s} {rec['status']}")
        out["figures"].append(rec)
    (REPO / "results" / "v2" / "figures.json").write_text(json.dumps(out, indent=2) + "\n")
    bad = [r for r in out["figures"] if r["status"] not in ("ok",)]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
