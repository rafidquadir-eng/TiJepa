"""G0 (PLAN_v2.md §3): hardware, integrity and intake records for the v2 continuation.

Writes results/v2/preflight.json, results/v2/corpus_shards_sha256.json (once; later runs
verify against it) and results/v2/manual_intake.json. Exits non-zero on any integrity
mismatch, so nothing downstream starts on a corrupted input. Never prints rclone config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "v2"
CKPT = REPO / "checkpoints" / "tijepa1_T1_seed0.pt"
LOG = REPO / "checkpoints" / "pretrain_log_tijepa1_seed0.csv"
CORPUS = REPO / "data" / "processed" / "v1"

EXPECTED_CKPT_SHA256 = "2774e60461d2c24b9b8c92b71d211c1cfa4b4e6241010d1c61888f6b7d054b22"
EXPECTED_CKPT_BYTES = 376_538_701
EXPECTED_CKPT_STEP = 14751
EXPECTED_INDEX_MD5 = "43b19b36849e88d29cefa484066e35a1"
EXPECTED_SHARDS = 43


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"ERROR: {exc}"


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def read_throughput(path: Path) -> dict:
    """Sequential O_DIRECT read of one shard (bypasses the page cache)."""
    out = subprocess.run(
        ["dd", f"if={path}", "of=/dev/null", "bs=16M", "iflag=direct"],
        capture_output=True, text=True,
    ).stderr.strip().splitlines()
    return {"file": str(path.relative_to(REPO)), "dd": out[-1] if out else ""}


def hardware() -> dict:
    import torch

    gpu = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
              "--format=csv,noheader"])
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    ok = bool(torch.isfinite((a @ a).float()).all())
    df = sh(["df", "--output=source,fstype,avail", str(REPO)]).splitlines()
    return {
        "hostname": socket.gethostname(),
        "gpus": [dict(zip(["name", "memory_total", "driver_version", "compute_capability"],
                          [p.strip() for p in line.split(",")])) for line in gpu.splitlines()],
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bf16_matmul_ok": ok,
        "cpu_logical_cores": os.cpu_count(),
        "cpu_physical_cores": int(sh(["bash", "-c", "lscpu -p=core,socket | grep -v '^#' "
                                      "| sort -u | wc -l"]) or 0),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1),
        "project_fs": df[-1] if df else "",
        "python": platform.python_version(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rclone-status", default="unknown",
                    help="short status of the Drive remote, recorded verbatim (no secrets)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    import torch

    ck_sha = digest(CKPT, "sha256")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ckpt = {
        "file": str(CKPT.relative_to(REPO)), "bytes": CKPT.stat().st_size, "sha256": ck_sha,
        "md5": digest(CKPT, "md5"), "step": int(ck["step"]), "keys": sorted(ck.keys()),
        "config_max_steps": ck["config"]["max_steps"],
    }
    if ck_sha != EXPECTED_CKPT_SHA256 or ckpt["bytes"] != EXPECTED_CKPT_BYTES:
        problems.append("checkpoint hash/size mismatch")
    if ckpt["step"] != EXPECTED_CKPT_STEP or not {"model", "optim", "config"} <= set(ck):
        problems.append("checkpoint step/keys mismatch")
    del ck

    idx_md5 = digest(CORPUS / "pretrain_index.csv", "md5")
    shards = sorted(CORPUS.glob("pretrain_shard_*.npy"))
    if idx_md5 != EXPECTED_INDEX_MD5:
        problems.append("corpus index md5 mismatch")
    if len(shards) != EXPECTED_SHARDS:
        problems.append(f"expected {EXPECTED_SHARDS} shards, found {len(shards)}")

    sha_file = OUT / "corpus_shards_sha256.json"
    current = {p.name: digest(p, "sha256") for p in shards}
    if sha_file.exists():
        recorded = json.loads(sha_file.read_text())["shards"]
        if recorded != current:
            problems.append("shard sha256 differs from corpus_shards_sha256.json")
    else:
        sha_file.write_text(json.dumps({
            "corpus": "data/processed/v1", "pretrain_index_md5": idx_md5,
            "n_shards": len(shards),
            "shards": current,
        }, indent=2) + "\n")

    pre = {
        "phase": "G0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hardware(),
        "checkpoint": ckpt,
        "corpus": {"path": "data/processed/v1", "pretrain_index_md5": idx_md5,
                   "n_shards": len(shards)},
        "storage_read": read_throughput(shards[len(shards) // 2]),
        "storage_decision": "project directory is local ext4 with >1 GB/s direct reads; "
                            "the corpus is read in place (no copy to another disk)",
        "drive_remote": args.rclone_status,
        "integrity_ok": not problems,
        "problems": problems,
    }
    (OUT / "preflight.json").write_text(json.dumps(pre, indent=2) + "\n")
    print(json.dumps({k: pre[k] for k in ("integrity_ok", "problems")}))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
