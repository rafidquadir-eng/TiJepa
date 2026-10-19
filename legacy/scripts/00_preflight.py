"""P0 / F0.2 preflight: record compute, disk, network reachability and compute tier.

Writes results/v1/preflight.json (v0 wrote results/v0/preflight.json). No data is
downloaded here. Extended for v1 per PLAN_v1.md F0.2: extra hosts, rclone remote
detection, and the v1 pretraining schedule of PLAN_v1.md Section 5.4.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]

HOSTS = [
    "https://zenodo.org",
    "https://datadryad.org",
    "https://ftp.ebi.ac.uk",
    "https://www.ebi.ac.uk",
    "https://ftp.ncbi.nlm.nih.gov",
    "https://www.ncbi.nlm.nih.gov",
    "https://figshare.com",
    "https://ndownloader.figshare.com",
    "https://bbbc.broadinstitute.org",
    "https://data.broadinstitute.org",
    "https://immunemap.org",
    "https://huggingface.co",
    "https://dl.fbaipublicfiles.com",
    "https://github.com",
    "https://pypi.org",
    # added for v1 (PLAN_v1.md F0.2)
    "https://api.crossref.org",
    "https://doi.org",
    "https://www.cellpose.org",
    "https://www.bioconductor.org",
    "https://cloud.r-project.org",
]

# Caps from the run brief (overriding PLAN.md Section 2 for this run).
RAW_VIDEO_CAP_GB = 120.0
MAX_DISK_FRACTION = 0.60


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"ERROR: {exc}"


def gpus() -> list[dict]:
    out = sh(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ]
    )
    if out.startswith("ERROR") or not out:
        return []
    found = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            found.append(
                {
                    "name": parts[0],
                    "memory_total_mib": int(parts[1].split()[0]),
                    "driver_version": parts[2] if len(parts) > 2 else None,
                    "compute_capability": parts[3] if len(parts) > 3 else None,
                }
            )
    return found


def reach(url: str, timeout: int = 15) -> dict:
    t0 = time.time()
    try:
        req = Request(url, method="GET", headers={"User-Agent": "TiJepa-preflight/0.1"})
        with urlopen(req, timeout=timeout) as resp:
            return {
                "url": url,
                "status": resp.status,
                "reachable": 200 <= resp.status < 400,
                "seconds": round(time.time() - t0, 2),
            }
    except Exception as exc:
        return {
            "url": url,
            "status": None,
            "reachable": False,
            "error": f"{type(exc).__name__}: {exc}"[:200],
            "seconds": round(time.time() - t0, 2),
        }


def rclone_remotes() -> dict:
    """Detect rclone remotes by NAME AND TYPE ONLY.

    Never reads, prints or stores rclone config contents, tokens or credentials
    (PLAN_v1.md Section 1 rule 6). `rclone listremotes --long` emits only
    "<name>: <type>" pairs, which is all that is captured here.
    """
    if shutil.which("rclone") is None:
        return {"rclone_installed": False, "remotes": [], "drive_remote": None}
    out = sh(["rclone", "listremotes", "--long"])
    remotes = []
    for line in out.splitlines():
        if ":" not in line:
            continue
        name, _, rtype = line.partition(":")
        remotes.append({"name": name.strip(), "type": rtype.strip()})
    drive = next((r["name"] for r in remotes if r["type"] == "drive"), None)
    if drive is None:
        drive = next(
            (r["name"] for r in remotes
             if any(k in r["name"].lower() for k in ("gdrive", "drive"))), None)
    return {
        "rclone_installed": True,
        "rclone_version": sh(["rclone", "version"]).splitlines()[0]
        if sh(["rclone", "version"]) else None,
        "remotes": remotes,
        "drive_remote": drive,
    }


def v1_schedule(n_gpu: int) -> dict:
    """PLAN_v1.md Section 5.4: pretraining runs and GPU-hour budget by tier."""
    if n_gpu >= 4:
        return {"runs": ["tijepa1_seed0", "tijepa1_seed1", "tijepa1_seed2",
                         "tijepa1_spatial_seed0"],
                "parallel": True, "gpu_hours_per_run": 16,
                "note": "4+ GPUs: three seeds plus the spatial-only ablation in parallel"}
    if n_gpu == 3:
        return {"runs": ["tijepa1_seed0", "tijepa1_seed1", "tijepa1_spatial_seed0"],
                "parallel": True, "gpu_hours_per_run": 16, "note": "3 GPUs"}
    if n_gpu == 2:
        return {"runs": ["tijepa1_seed0", "tijepa1_spatial_seed0"],
                "parallel": True, "gpu_hours_per_run": 16, "note": "2 GPUs"}
    if n_gpu == 1:
        return {"runs": ["tijepa1_seed0", "tijepa1_spatial_seed0"],
                "parallel": False, "gpu_hours_per_run": 12,
                "gpu_hours_total": 24,
                "note": "1 GPU: TiJepa-1 seed 0 for 12 GPU-h, then the spatial-only "
                        "ablation for the SAME NUMBER OF STEPS (~24 GPU-h total). "
                        "Pretraining has a single seed; seed variance in Part A comes "
                        "from random-init draws, cell subsamples and fold assignment "
                        "(PLAN_v1.md F3.5). This must be stated in Methods."}
    return {"runs": [], "parallel": False, "gpu_hours_per_run": 0,
            "note": "no GPU detected"}


def choose_tier(n_gpu: int, min_mem_mib: int) -> tuple[str, str]:
    """Tier selection per the run brief: T1 for 1 GPU, T2 for >=2 GPUs."""
    if n_gpu == 0:
        return "T0", "no CUDA GPU detected"
    if n_gpu == 1:
        return "T1", f"exactly 1 GPU detected ({min_mem_mib} MiB); brief: 1 GPU -> T1"
    return "T2", f"{n_gpu} GPUs detected; brief: >=2 GPUs -> T2"


def main() -> int:
    data_dir = REPO / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(data_dir)

    gpu_list = gpus()
    n_gpu = len(gpu_list)
    min_mem = min((g["memory_total_mib"] for g in gpu_list), default=0)
    tier, tier_reason = choose_tier(n_gpu, min_mem)

    try:
        import torch

        torch_info = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "bf16_supported": bool(
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            ),
        }
    except Exception as exc:
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}

    free_gb = usage.free / 1e9
    disk_cap_gb = free_gb * MAX_DISK_FRACTION
    raw_cap_gb = min(RAW_VIDEO_CAP_GB, disk_cap_gb)

    net = [reach(u) for u in HOSTS]

    # Seed policy for this run: single GPU -> pretrain seed 0 only (brief Section 3).
    if n_gpu >= 3:
        pretrain_seeds, seed_note = [0, 1, 2], "3 parallel pretraining runs, one per GPU"
    elif n_gpu == 2:
        pretrain_seeds, seed_note = [0, 1], "2 parallel pretraining runs"
    else:
        pretrain_seeds, seed_note = (
            [0],
            "single GPU: pretraining uses seed 0 only; seeds 0/1/2 used for all probes "
            "and downstream models (must be stated in Methods)",
        )

    info = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "compute": {
            "cpu_cores_logical": os.cpu_count(),
            "cpu_model": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                None,
            )
            if Path("/proc/cpuinfo").exists()
            else None,
            "ram_total_gb": round(
                int(
                    next(
                        line.split()[1]
                        for line in Path("/proc/meminfo").read_text().splitlines()
                        if line.startswith("MemTotal")
                    )
                )
                / 1e6,
                1,
            )
            if Path("/proc/meminfo").exists()
            else None,
            "gpus": gpu_list,
            "gpu_count": n_gpu,
            **torch_info,
        },
        "disk": {
            "path": str(data_dir),
            "total_gb": round(usage.total / 1e9, 1),
            "used_gb": round(usage.used / 1e9, 1),
            "free_gb": round(free_gb, 1),
            "all_data_local": True,
        },
        "tier": {
            "tier": tier,
            "reason": tier_reason,
            "encoder": "ViT-S",
            "precision": "bf16",
            "raw_video_cap_gb": round(raw_cap_gb, 1),
            "raw_video_cap_source": (
                "brief cap 120 GB" if raw_cap_gb == RAW_VIDEO_CAP_GB else "60% of free disk"
            ),
            "disk_cap_gb": round(disk_cap_gb, 1),
            "pretrain_gpu_hours_budget_per_seed": 24,
            "pretrain_seeds": pretrain_seeds,
            "seed_policy": seed_note,
            "probe_seeds": [0, 1, 2],
        },
        "v1_schedule": v1_schedule(n_gpu),
        "rclone": rclone_remotes(),
        "network": net,
        "unreachable_hosts": [r["url"] for r in net if not r["reachable"]],
        "git_sha": sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "git_branch": sh(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"]),
    }

    out = REPO / "results" / "v1" / "preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in
                      ("tier", "v1_schedule", "unreachable_hosts")}, indent=2))
    print("rclone drive remote detected: "
          f"{info['rclone']['drive_remote'] or 'NONE'}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
