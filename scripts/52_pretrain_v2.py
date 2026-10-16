"""G1.2-G1.3 driver (PLAN_v2.md §4): TiJepa-1 to a step budget, then the ablation.

  1. `tijepa1` - mixed masking (p_future = 0.5), resumed from its step-14,751 checkpoint
     and run to the planned final step S_total, where LR and EMA finish annealing;
  2. `tijepa1_spatial` - spatial masking only, from scratch, to the **same** S_total with
     the same schedule, seed and frozen shard list.

`--max-hours` is a per-segment safety cap only. `--benchmark` runs a fixed number of
steps from a scratch copy of the checkpoint, writes nothing next to the real checkpoint,
and reports the throughput used to set S_total (G1.2).
"""
from __future__ import annotations

import argparse
import functools
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.train.pretrain_video import PretrainConfig, train  # noqa: E402

OUT = REPO / "results" / "v2" / "partA"
SHARDS = REPO / "data" / "processed" / "v1"
CKPTS = REPO / "checkpoints"

# The segment that ran on the previous instance, before the trainer kept segment records.
# Seconds are the training-loop time logged at step 14,750 in
# checkpoints/pretrain_log_tijepa1_seed0.csv (5,401.9 s), plus one step.
FIRST_SEGMENT = {
    "host": "previous instance (shared A100; hostname not recorded)",
    "gpu_name": "NVIDIA A100-SXM4-40GB",
    "torch_version": "2.11.0+cu128",
    "start_step": 0,
    "end_step": 14751,
    "seconds": 5402.3,
    "gpu_seconds": 5402.3,
    "start_utc": None,
    "end_utc": "2026-10-16T04:25:59Z",
    "note": "seeded from the v1 log; that trainer did not record segments",
}


def config(name: str, p_future: float, a: argparse.Namespace, out_dir: Path) -> PretrainConfig:
    return PretrainConfig(
        shard_dir=str(SHARDS), out_dir=str(out_dir), run_name=name, seed=a.seed,
        max_steps=a.max_steps, max_hours=a.max_hours, batch_size=a.batch_size,
        num_workers=a.num_workers, p_future=p_future,
        prior_segments_if_missing=[FIRST_SEGMENT] if name == "tijepa1" else [],
    )


def gpu_util_sampler(samples: list[float], stop: threading.Event) -> None:
    while not stop.wait(5.0):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            samples.append(float(out.split()[0]))
        except Exception:
            pass


def benchmark(a: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(dir=a.scratch) as tmp:
        tmpd = Path(tmp)
        shutil.copy2(CKPTS / "tijepa1_T1_seed0.pt", tmpd / "tijepa1_T1_seed0.pt")
        start_step = 14751
        cfg = config("tijepa1", 0.5, a, tmpd)
        cfg.max_steps = start_step + a.bench_steps
        cfg.dry_run = True
        cfg.log_every = 50
        util: list[float] = []
        ev = threading.Event()
        th = threading.Thread(target=gpu_util_sampler, args=(util, ev), daemon=True)
        th.start()
        t0 = time.time()
        s = train(cfg)
        ev.set()
        th.join()
        wall = time.time() - t0
    seg = s["segments"][-1]
    steps = s["steps"] - start_step
    # discard warm-up (worker start, cudnn autotune): rate over the loop only
    rate = steps / seg["gpu_seconds"] * 3600
    res = {"bench_steps": steps, "loop_seconds": seg["gpu_seconds"],
           "process_seconds": round(wall, 1), "steps_per_hour": round(rate, 1),
           "gpu_util_mean": round(sum(util[2:]) / max(1, len(util[2:])), 1) if util else None,
           "gpu_util_samples": util, "num_workers": a.num_workers,
           "batch_size": a.batch_size, "gpu": seg["gpu_name"],
           "torch_version": seg["torch_version"], "scratch_copy_deleted": True}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT.parent / "pretrain_benchmark.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "gpu_util_samples"}, indent=2))


def run(name: str, p_future: float, a: argparse.Namespace) -> dict:
    cfg = config(name, p_future, a, CKPTS)
    print(f"\n=== {name}: p_future={p_future}, planned final step {cfg.max_steps}, "
          f"safety cap {cfg.max_hours} h per segment")
    s = train(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pretrain_{name}_seed{a.seed}.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps({k: s.get(k) for k in
                      ("run_name", "steps", "planned_final_step", "annealed", "gpu_hours",
                       "effective_epochs", "stop_reason", "final_lr",
                       "final_ema_momentum", "final_loss_mean_last500")}, indent=2))
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="planned final step S_total (default: results/v2/pretrain_plan.json)")
    ap.add_argument("--max-hours", type=float, default=30.0,
                    help="per-segment safety cap; hitting it ends the run unannealed")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=28)
    ap.add_argument("--which", default="both", choices=["both", "main", "ablation"])
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=500)
    ap.add_argument("--scratch", default=None, help="directory for the benchmark copy")
    a = ap.parse_args()

    if a.benchmark:
        benchmark(a)
        sys.exit(0)
    if a.max_steps <= 0:
        plan = json.loads((REPO / "results" / "v2" / "pretrain_plan.json").read_text())
        a.max_steps = int(plan["S_total"])

    if a.which in ("both", "main"):
        s = run("tijepa1", 0.5, a)
        if s["stop_reason"].startswith("signal_"):
            print("stopped by signal; checkpoint saved, resume with the same command")
            sys.exit(0)
        if not s["annealed"]:
            sys.exit(f"tijepa1 stopped unannealed ({s['stop_reason']}); "
                     "ablation not started")
    if a.which in ("both", "ablation"):
        run("tijepa1_spatial", 0.0, a)
