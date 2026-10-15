"""M7: generate every figure from results/*.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.viz import figures as F  # noqa: E402

if __name__ == "__main__":
    made = {}
    for fn in F.ALL:
        try:
            made[fn.__name__] = fn(REPO)
            print(f"  {fn.__name__}: {made[fn.__name__][0]}")
        except Exception as exc:
            made[fn.__name__] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  {fn.__name__} FAILED: {exc}")
    (REPO / "results" / "figures.json").write_text(json.dumps(made, indent=2) + "\n")
