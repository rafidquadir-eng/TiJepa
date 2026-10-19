"""Record the environment for reproducibility (PLAN.md §3): results/env.json."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKGS = ["numpy", "scipy", "sklearn", "pandas", "matplotlib", "torch", "timm",
        "tifffile", "skimage", "trackpy", "zarr", "yaml", "pytest", "ruff"]


def main() -> int:
    mods = {}
    for m in PKGS:
        try:
            mods[m] = getattr(__import__(m), "__version__", "unknown")
        except Exception:
            mods[m] = "not installed"
    try:
        import torch

        cuda = {"cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                else None}
    except Exception:
        cuda = {}
    env = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip(),
        "git_branch": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": mods,
        **cuda,
        "note": "torch was upgraded from 2.11.0+cu128 to 2.14.1+cu130 mid-run when the "
                "imaging stack was installed; CUDA was re-verified on the A100 before "
                "training. See RUN_REPORT deviation 17.",
    }
    (REPO / "results" / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    print(json.dumps({k: env[k] for k in ("git_sha", "python", "device")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
